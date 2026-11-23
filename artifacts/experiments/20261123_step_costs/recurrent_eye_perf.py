"""Price an eye that reads current periphery and Mamba history without a denoiser call."""

import argparse
import gc
import json
from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

from saccade import chooser as C, model as M, train as TR
from . import streaming_perf as P

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def run(model, cfg, batch, count):
    torch.cuda.reset_peak_memory_stats()
    dev = P.DEV
    rng = torch.Generator(device=dev).manual_seed(20261123)
    torch.manual_seed(20261123)
    carry = model.belief.initial(batch, dev)
    state = model.record.zero(batch, dev)
    bp = torch.zeros(batch, 1, cfg["belief"]["width"], device=dev)
    action = torch.randint(17, (batch, 1), device=dev, generator=rng)
    reward = torch.zeros_like(action, dtype=torch.float32)
    done = torch.zeros_like(action, dtype=torch.bool)
    raw = torch.randint(256, (3, batch, 63, 63, 3), dtype=torch.uint8, device=dev, generator=rng)
    frame = raw[0].clone()
    u = torch.rand(batch, 1, device=dev, generator=rng)
    centres = C.CENTRES.to(dev)
    per = model.perception
    with TR.autocast(dev, cfg):
        initial_p, initial_v = C.coarse(per, frame)
    head = nn.Sequential(nn.Linear(initial_p[0].numel() + initial_v.shape[-1] + bp.shape[-1], 64),
                         nn.GELU(), nn.Linear(64, 49)).to(dev)

    def tick():
        with TR.autocast(dev, cfg):
            p, v = C.coarse(per, frame)
            logits = head(torch.cat((bp[:, 0], p.flatten(1), v), -1)).float()
            prob = .8 * logits.softmax(-1) + .2 / 49
            idx = (u > prob.cumsum(-1)).sum(-1).clamp_max(48)
            fix = centres[idx]
            crop = M.crop(frame, fix).permute(0, 3, 1, 2).float() / 127.5 - 1
            g = M.out_map(per.fovea(crop).flatten(2).transpose(1, 2), per.s)
            percept = M.Percept(p[:, None], g[:, None], v[:, None], fix[:, None])
            _, rs = model.record.scan(model.record.input(percept.p, percept.g, percept.fix), done, state)
            emb = model.belief.percept_embed(percept, rs)
            y, new = model.belief(model.belief.tokens(emb, reward, done, action), carry)
            state.copy_(rs[:, -1])
            bp.copy_(y[:, -1:])
            for dst, src in zip(carry, new):
                for f in fields(dst):
                    getattr(dst, f.name).copy_(getattr(src, f.name))
        return logits, fix, bp

    buffers = [bp, state, *P.flatten(carry)]
    initial = [t.clone() for t in buffers]

    def restore():
        for b, saved in zip(buffers, initial):
            b.copy_(saved)

    original = M.FunctionalMamba3.forward
    try:
        M.FunctionalMamba3.forward = P.fixed_forward
        graph, outputs = P.capture(tick)
        restore()
        eager = []
        for x in raw:
            frame.copy_(x)
            eager.append(tuple(a.clone() for a in tick()))
        restore()
        deltas = []
        for x, expected in zip(raw, eager):
            frame.copy_(x)
            graph.replay()
            deltas.append(P.differences(expected, outputs))
        timing = P.time_block(graph.replay, count)
    finally:
        M.FunctionalMamba3.forward = original
    return {"recurrent_eye_graph": timing, "dynamic_sequence_max_abs_deltas": deltas,
            "readout_params": sum(p.numel() for p in head.parameters()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    model, cfg = P.S2.AI.load(P.S2.P15)
    model.requires_grad_(False).eval()
    result = {}
    for batch in args.batch:
        gc.collect()
        torch.cuda.empty_cache()
        result[str(batch)] = run(model, cfg, batch, args.count)
        print(batch, result[str(batch)], flush=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
