"""Reuse gaze-independent computation while preserving the full denoiser eye's inputs."""

import argparse
import gc
import json
from dataclasses import fields
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from saccade import chooser as C, model as M, train as TR
from saccade.eval import s1 as S
from . import split_perf as SP
from . import streaming_perf as P

HERE = Path(__file__).resolve().parent
DEV = P.DEV


@torch.no_grad()
def watching(model, cfg, batch, length):
    """Match the existing stateful split with a chunk-batched coarse/record preparation pass."""
    den, per = model.denoiser, model.perception
    assert model.record.fovea == 0, "only the measured P1.5 record is independent of recent looks"
    gen = torch.Generator(device=DEV).manual_seed(20261123)
    frames = torch.randint(256, (length, batch, 63, 63, 3), device=DEV, generator=gen, dtype=torch.uint8)
    fixes = C.CENTRES.to(DEV)[torch.randint(49, (length, batch, 1), device=DEV, generator=gen)]
    actions = torch.randint(17, (length, batch, 1), device=DEV, generator=gen)
    reward = torch.zeros(batch, 1, device=DEV)
    done = torch.zeros(batch, 1, dtype=torch.bool, device=DEV)
    tc = cfg["edm"]["shortcut"]["context_signal"]
    code = torch.full_like(done, M.shortcut_code(tc, den.k_max.bit_length() - 1, den.k_max), dtype=torch.long)
    bp = torch.zeros(batch, 1, cfg["belief"]["width"], device=DEV)
    rec = model.record.zero(batch, DEV)
    carry = model.belief.initial(batch, DEV)
    clock = torch.full((batch, 1, 1), float(den.window), device=DEV)
    frame_index = torch.zeros(1, dtype=torch.long, device=DEV)
    with TR.autocast(DEV, cfg):
        percept = per(frames[0, :, None], fixes[0])
        parts = tuple(x.float() for x in percept.parts())
        ncache = den.new_cache()
        for _ in range(den.window):
            ncache = M.context_pass(den, parts, actions[0], (model.belief.memory(bp), rec[:, None]),
                                    fixes[0], ncache, cfg, gen)
    kv = [(a.contiguous().clone(), b.contiguous().clone()) for a, b in ncache["kv"]]
    buffers = [frame_index, clock, bp, rec, *P.flatten(carry), *P.flatten(kv)]
    initial = [x.clone() for x in buffers]
    inputs = {}

    def restore():
        for dst, src in zip(buffers, initial):
            dst.copy_(src)

    def precompute():
        with TR.autocast(DEV, cfg):
            p, v = C.coarse(per, frames.flatten(0, 1))
            p = p.reshape(length, batch, M.N, -1)
            v = v.reshape(length, batch, -1)
            past, after = model.record.scan(p.transpose(0, 1), done.expand(-1, length), initial[3])
            phys = den.inp[0](p.float()) + den.pos + den.kind[0]
            phys = phys + den.record_read(past.transpose(0, 1)).to(phys.dtype)
            light = (den.inp[2](v.float()) + den.kind[2]).unsqueeze(-2)
            read = model.belief.record_in(after).transpose(0, 1)
        return {"p": p, "v": v, "phys": phys, "light": light, "read": read}

    def take(x):
        return x.index_select(0, frame_index).transpose(0, 1)

    def tick(prepared=False):
        with TR.autocast(DEV, cfg):
            raw, f, a = take(frames), take(fixes).squeeze(1), take(actions).squeeze(1)
            if prepared:
                p, v, phys, light, read = (take(inputs[k]) for k in ("p", "v", "phys", "light", "read"))
                c = M.crop(raw, f).flatten(0, 1).permute(0, 3, 1, 2).float() / 127.5 - 1
                g = M.out_map(per.fovea(c).flatten(2).transpose(1, 2), per.s)[:, None]
                percept = M.Percept(p, g, v, f)
                emb = S.embed(model, percept, read)
                physical = (phys, light)
            else:
                percept = per(raw, f)
                _, after = model.record.scan(percept.p, done, rec)
                emb = model.belief.percept_embed(percept, after)
                physical = None
            p, g, v = (x.float() for x in percept.parts())
            output, nkv = SP.split_denoiser(den, p, v, g, code, a, (model.belief.memory(bp), rec[:, None]),
                                            f, kv, clock, physical)
            y, nc = model.belief(model.belief.tokens(emb, reward, done, a), carry)
            if not prepared:
                rec.copy_(after[:, -1])
            bp.copy_(y[:, -1:])
            for old, new in zip(carry, nc):
                for f in fields(old):
                    getattr(old, f.name).copy_(getattr(new, f.name))
            for (k, v), (nk, nv) in zip(kv, nkv):
                k.copy_(nk)
                v.copy_(nv)
            clock.add_(1)
            frame_index.copy_((frame_index + 1) % length)
            return output, y

    original = M.FunctionalMamba3.forward
    M.FunctionalMamba3.forward = P.fixed_forward
    try:
        setup = P.time_block(precompute, count=2, repeats=3)
        inputs.update(precompute())
        reports = {}
        predictions = []
        for name, prepared in (("full_streaming", False), ("prepared_coarse_record", True)):
            restore()
            graph, out = P.capture(lambda: tick(prepared))
            restore()
            seq = []
            for _ in range(length):
                graph.replay()
                seq.append(tuple(x.clone() for x in P.flatten(out)))
            predictions.append(seq)
            reports[name] = P.time_block(graph.replay, count=length, repeats=4)
            del graph, out
        delta = [P.differences(a, b) for a, b in zip(*predictions)]
    finally:
        M.FunctionalMamba3.forward = original
    return {
        **reports, "prepare_ms_per_chunk": setup["median_wall_ms"], "chunk_frames": length,
        "max_abs_checked_delta_by_frame": delta,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
    }


@torch.no_grad()
def dreaming(model, cfg, batch):
    """Measure denoising plus context with the same frozen record projected once."""
    den = model.denoiser
    input_gen = torch.Generator(device=DEV).manual_seed(20261123)
    actions = torch.randint(17, (batch, 1), device=DEV, generator=input_gen)
    fix = C.CENTRES.to(DEV)[torch.randint(49, (batch, 1), device=DEV, generator=input_gen)]
    record = torch.randn(batch, 1, M.N, model.record.dim, device=DEV, generator=input_gen)
    p = torch.randn(batch, 1, M.N, 16, device=DEV, generator=input_gen)
    g = torch.randn(batch, 1, M.G, 64, device=DEV, generator=input_gen)
    v = torch.randn(batch, 1, 8, device=DEV, generator=input_gen)
    with TR.autocast(DEV, cfg):
        mem = (model.belief.memory(torch.zeros(batch, 1, cfg["belief"]["width"], device=DEV)), record)
        cache = den.new_cache()
        for _ in range(den.window):
            cache = M.context_pass(den, (p, g, v), actions, mem, fix, cache, cfg, input_gen)
    shapes = [x.shape[2:] for x in (p, g, v)]
    original = den.record_read.forward
    gen = torch.Generator(device=DEV)

    def tick(reuse):
        with TR.autocast(DEV, cfg):
            if reuse:
                cached = original(record)
                den.record_read.forward = MethodType(lambda self, x: cached, den.record_read)
            try:
                output = M.sample(den, shapes, actions, mem, fix, cache, cfg, gen)
                next_kv = M.context_pass(den, output, actions, mem, fix, cache, cfg, gen)["kv"]
                return output, next_kv
            finally:
                den.record_read.forward = original

    gen.manual_seed(17)
    ref = tick(False)
    gen.manual_seed(17)
    test = tick(True)
    return {"native": P.time_block(lambda: tick(False), 6, repeats=4),
            "project_record_once": P.time_block(lambda: tick(True), 6, repeats=4),
            "max_abs_delta": P.differences(ref, test)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    model, cfg = P.S2.AI.load(P.S2.P15)
    model.requires_grad_(False).eval()
    report = {}
    for batch in args.batch:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        watch = watching(model, cfg, batch, args.length)
        torch.cuda.empty_cache()
        dream = dreaming(model, cfg, batch)
        report[str(batch)] = {"watching": watch, "dreaming": dream}
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(batch, report[str(batch)], flush=True)


if __name__ == "__main__":
    main()
