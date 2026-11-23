"""Exact physical/crop split of the frozen denoiser for timing, with causal state carried."""

import argparse
import gc
import json
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from saccade import chooser as C, model as M, train as TR
from . import streaming_perf as P

HERE = Path(__file__).resolve().parent


def time_keys(x):
    return x.transpose(1, 2).flatten(2, 3).unsqueeze(1)


def project(block, token, angle, cond):
    sh1, sc1, g1, sh2, sc2, g2 = cond
    args = block.attn.project(block.n1(token) * (1 + sc1) + sh1, angle)
    return args, (g1, sh2, sc2, g2)


def complete(block, token, q, keys, vals, controls):
    g1, sh2, sc2, g2 = controls
    token = token + g1 * block.attn.attend(q, keys, vals)
    return token + g2 * block.mlp(block.n2(token) * (1 + sc2) + sh2)


def combine(phys, fine):
    return torch.cat((phys[..., :M.N, :], fine, phys[..., M.N:, :]), -2)


def error(left, right):
    result = {}
    for name, a, b in zip(("periphery", "glimpse", "lighting"), left[0], right[0]):
        diff = a.float() - b.float()
        result[name] = {"max": diff.abs().max().item(), "rms": diff.square().mean().sqrt().item()}
    result["kv_max"] = P.differences(left[1], right[1])
    return result


def split_denoiser(den, p, v, crop, code, act, memory, fix, kv, clock):
    """Compute physical tokens first, retaining layer K/V for processing the selected crop."""
    mem, record = memory
    pt = den.inp[0](p) + den.pos + den.kind[0]
    pt = pt + den.record_read(record).to(pt.dtype)
    lt = (den.inp[2](v) + den.kind[2]).unsqueeze(-2)
    token = torch.cat((pt, lt, mem + den.kind[3]), -2)
    grid = torch.arange(M.N, device=p.device)
    sa = torch.cat(((grid % M.GRID)[:, None] * den.f_space, (grid // M.GRID)[:, None] * den.f_space), -1)
    sa = F.pad(sa, (0, 0, 0, 1 + den.K))
    ta = clock * den.f_time
    context = den.cond(code, act)
    layers = []
    temporal = 0
    for block in den.blocks:
        cond = block.mod(F.silu(context)).unsqueeze(2).chunk(6, -1)
        (q, k, u), controls = project(block, token, ta[..., None, :].expand(*token.shape[:-1], -1)
                                     if block.temporal else sa, cond)
        if block.temporal:
            old_k, old_u = kv[temporal]
            pk = old_k.index_select(-2, den.phys)
            pu = old_u.index_select(-2, den.phys)
            keys, vals = time_keys(torch.cat((pk, k), 1)), time_keys(torch.cat((pu, u), 1))
            temporal += 1
        else:
            keys, vals = k, u
        token = complete(block, token, q, keys, vals, controls)
        layers.append((k, u, cond))

    cells = M.window(fix)
    fine = den.inp[1](crop) + den.pos[cells] + den.kind[1]
    fine_angle = torch.cat(((cells % M.GRID)[..., None] * den.f_space,
                            (cells // M.GRID)[..., None] * den.f_space), -1)
    next_kv = []
    temporal = 0
    for block, (pk, pu, cond) in zip(den.blocks, layers):
        (q, k, u), controls = project(block, fine, ta[..., None, :].expand(*fine.shape[:-1], -1)
                                     if block.temporal else fine_angle, cond)
        keys, vals = combine(pk, k), combine(pu, u)
        if block.temporal:
            old_k, old_u = kv[temporal]
            next_k, next_u = torch.cat((old_k, keys), 1), torch.cat((old_u, vals), 1)
            next_kv.append((next_k[:, -den.window:], next_u[:, -den.window:]))
            keys, vals = time_keys(next_k), time_keys(next_u)
            temporal += 1
        fine = complete(block, fine, q, keys, vals, controls)
    out = den.head(combine(token, fine), (context, context))
    return out, next_kv


@torch.no_grad()
def benchmark(model, cfg, batch, count):
    torch.cuda.reset_peak_memory_stats()
    dev = P.DEV
    den = model.denoiser
    gen = torch.Generator(device=dev).manual_seed(20261123)
    act = torch.randint(17, (batch, 1), device=dev, generator=gen)
    fix = C.CENTRES.to(dev)[torch.randint(49, (batch, 1), device=dev, generator=gen)]
    code = torch.full_like(act, M.shortcut_code(cfg["edm"]["shortcut"]["context_signal"], den.k_max.bit_length() - 1, den.k_max))
    frames = torch.randint(0, 256, (3, batch, 1, 63, 63, 3), device=dev, dtype=torch.uint8, generator=gen)
    clock = torch.full((batch, 1, 1), float(den.window), device=dev)
    r = torch.zeros_like(act, dtype=torch.float32)
    done = torch.zeros_like(act, dtype=torch.bool)
    record = model.record.zero(batch, dev)
    carry = model.belief.initial(batch, dev)
    bp = torch.zeros(batch, 1, cfg["belief"]["width"], device=dev)
    with TR.autocast(dev, cfg):
        real = model.perception(frames[0], fix)
        x = tuple(z.float() for z in real.parts())
        cache = den.new_cache()
        for _ in range(den.window):
            cache = M.context_pass(den, x, act, (model.belief.memory(bp), record[:, None]), fix, cache, cfg, gen)
    kv = [(k.contiguous().clone(), v.contiguous().clone()) for k, v in cache["kv"]]

    def tick(raw, split, *, commit=False):
        with TR.autocast(dev, cfg):
            p = model.perception(raw, fix)
            parts = tuple(z.float() for z in p.parts())
            mem = (model.belief.memory(bp), record[:, None])
            if split:
                out, next_kv = split_denoiser(den, parts[0], parts[2], parts[1], code, act, mem, fix, kv, clock)
            else:
                cache = {"t": den.window, "kv": kv}
                out, new = den.step(parts, code, act, mem, fix, cache)
                next_kv = den.commit(cache, new)["kv"]
            _, rs = model.record.scan(model.record.input(p.p, p.g, fix), done, record)
            emb = model.belief.percept_embed(p, rs)
            y, nc = model.belief(model.belief.tokens(emb, r, done, act), carry)
            if commit:
                record.copy_(rs[:, -1])
                bp.copy_(y[:, -1:])
                for old, new in zip(carry, nc):
                    for field in fields(old):
                        getattr(old, field.name).copy_(getattr(new, field.name))
                for (k, v), (nk, nv) in zip(kv, next_kv):
                    k.copy_(nk)
                    v.copy_(nv)
                clock.add_(1)
            return out, next_kv, y, nc

    original = M.FunctionalMamba3.forward
    try:
        M.FunctionalMamba3.forward = P.fixed_forward
        # Validate the novel split against the trained monolithic forward at equal signal/fixation.
        baseline, alternative = tick(frames[0], False), tick(frames[0], True)
        split_delta = error(baseline, alternative)
        # Use a common FP32 math attention path to separate arithmetic from mask-sensitive BF16 kernels.
        mem_fp = (model.belief.memory(bp), record[:, None])
        full_cache = {"t": den.window, "kv": [(k.float(), v.float()) for k, v in kv]}
        with sdpa_kernel(SDPBackend.MATH):
            reference, next_kv = den.step(x, code, act, mem_fp, fix, full_cache)
            fp32_delta = error((reference, den.commit(full_cache, next_kv)["kv"]),
                               split_denoiser(den, x[0], x[2], x[1], code, act, mem_fp, fix, full_cache["kv"], clock))
        del baseline, alternative, reference, next_kv, full_cache, mem_fp
        # Preserve initial buffers when testing several steps against eager current inputs.
        buffers = [bp, record, clock, *P.flatten(carry), *P.flatten(kv)]
        initial = [b.clone() for b in buffers]

        def restore():
            for b, saved in zip(buffers, initial):
                b.copy_(saved)

        inputs = frames[0].clone()
        graph, output = P.capture(lambda: tick(inputs, True, commit=True))
        restore()
        eager = [tuple(t.clone() for t in P.flatten(tick(raw, True, commit=True))) for raw in frames]
        restore()
        comparisons = []
        for raw, ref in zip(frames, eager):
            inputs.copy_(raw)
            graph.replay()
            comparisons.append(P.differences(ref, output))
        timing = P.time_block(graph.replay, count)
    finally:
        M.FunctionalMamba3.forward = original
    return {"split_vs_full_bf16": split_delta, "split_vs_full_fp32_common_math": fp32_delta,
            "eager_vs_graph_dynamic_inputs_deltas": comparisons,
            "split_stateful_graph": timing,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--count", type=int, default=20)
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
        result[str(batch)] = benchmark(model, cfg, batch, args.count)
        print(batch, result[str(batch)], flush=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
