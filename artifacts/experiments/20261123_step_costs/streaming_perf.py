"""Benchmark the proposed eye's frozen P1.5 streaming work with stable shapes."""

import argparse
import gc
import importlib
import json
import os
from dataclasses import fields
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.modules.mamba3 import heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

from saccade import chooser as C, model as M, train as TR

HERE = Path(__file__).resolve().parent
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
DEV = S2.DEV


def fixed_forward(self, inputs, carry, seq_idx=None, backend=None):
    """Same packed kernel as native, with one known contiguous segment per row."""
    assert seq_idx is None and backend is None
    c = self.core
    batch, length, _ = inputs.shape
    z, x, key, query, dt, decay, trap, angles = torch.split(c.in_proj(inputs), [
        c.d_inner, c.d_inner, c.d_state * c.num_bc_heads, c.d_state * c.num_bc_heads,
        c.nheads, c.nheads, c.nheads, c.num_rope_angles], dim=-1)
    z = rearrange(z, "b l (h p) -> b l h p", p=c.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=c.headdim)
    key = c.B_norm(rearrange(key, "b l (r g n) -> b l r g n", r=1, g=c.num_bc_heads))
    query = c.C_norm(rearrange(query, "b l (r g n) -> b l r g n", r=1, g=c.num_bc_heads))
    dt = F.softplus(dt + c.dt_bias)
    adt = torch.clamp(-heavy_tail_activation(decay.float()), max=-c.A_floor) * dt
    angles = angles.unsqueeze(-2).expand(-1, -1, c.nheads, -1).float()
    cu = torch.arange(batch + 1, device=inputs.device, dtype=torch.int32) * length

    def flat(tensor):
        return tensor.reshape(1, batch * length, *tensor.shape[2:])

    def heads(tensor):
        return tensor.reshape(1, batch * length, -1).transpose(1, 2)

    states = (carry.angle, carry.ssm, carry.k, carry.v)
    y, _, ssm, last_k, last_v = mamba3_siso_combined(
        Q=flat(query.squeeze(2)), K=flat(key.squeeze(2)), V=flat(x), ADT=heads(adt), DT=heads(dt), Trap=heads(trap),
        Q_bias=c.C_bias.squeeze(1), K_bias=c.B_bias.squeeze(1), Angles=flat(angles), D=c.D,
        Z=flat(z) if not c.is_outproj_norm else None, chunk_size=c.chunk_size, Input_States=states,
        return_final_states=True, cu_seqlens=cu)
    y = y.reshape(batch, length, -1)
    if c.is_outproj_norm:
        y = c.norm(y, z.flatten(-2))
    _, angle = angle_dt_fwd(flat(angles).to(torch.bfloat16).float(), heads(dt).float().contiguous(),
                            init_state=carry.angle.float().contiguous(), chunk_size=c.chunk_size,
                            return_output_state=True, cu_seqlens=cu)
    return c.out_proj(y.to(x.dtype)), M.Mamba3Carry(angle, ssm.float(), last_k, last_v)


def time_block(fn, count, repeats=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    gpu = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        tick = perf_counter()
        start.record()
        for _ in range(count):
            fn()
        stop.record()
        stop.synchronize()
        times.append((perf_counter() - tick) * 1e3 / count)
        gpu.append(start.elapsed_time(stop) / count)
    return {"median_wall_ms": float(np.median(times)), "median_stream_ms": float(np.median(gpu))}


def capture(fn):
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def flatten(outputs):
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if hasattr(outputs, "__dataclass_fields__"):
        outputs = tuple(getattr(outputs, field.name) for field in fields(outputs))
    return [x for y in outputs for x in flatten(y)]


def differences(left, right):
    a, b = flatten(left), flatten(right)
    assert len(a) == len(b)
    return max((x.float() - y.float()).abs().max().item() for x, y in zip(a, b))


@torch.no_grad()
def benchmark(model, cfg, batch, count, graphs):
    torch.cuda.reset_peak_memory_stats()
    rng = torch.Generator(device=DEV).manual_seed(20261123)
    frames = torch.randint(256, (batch, 1, 63, 63, 3), dtype=torch.uint8, device=DEV, generator=rng)
    fix = C.CENTRES.to(DEV)[torch.randint(49, (batch, 1), device=DEV, generator=rng)]
    act = torch.randint(17, (batch, 1), device=DEV, generator=rng)
    r = torch.zeros_like(act, dtype=torch.float32)
    done = torch.zeros_like(act, dtype=torch.bool)
    state = model.record.zero(batch, DEV)
    carry = model.belief.initial(batch, DEV)
    den = model.denoiser
    code = torch.full_like(act, M.shortcut_code(cfg["edm"]["shortcut"]["context_signal"],
                                                den.k_max.bit_length() - 1, den.k_max))
    with TR.autocast(DEV, cfg):
        percept = model.perception(frames, fix)
        parts = tuple(x.float() for x in percept.parts())
        memory = (model.belief.memory(torch.randn(batch, 1, cfg["belief"]["width"], device=DEV, generator=rng)),
                  state[:, None])
        cache = den.new_cache()
        for _ in range(den.window):
            cache = M.context_pass(den, parts, act, memory, fix, cache, cfg, rng)
        _, warm_record = model.record.scan(model.record.input(percept.p, percept.g, fix), done, state)
        emb = model.belief.percept_embed(percept, warm_record)
        _, carry = model.belief(model.belief.tokens(emb, r, done, act), carry)
        state = warm_record[:, -1].contiguous()

    def world():
        with TR.autocast(DEV, cfg):
            kept = []
            out, kv = den.step(parts, code, act, memory, fix, cache, kept)
            glimpse = den.glimpse_step(parts[1], code, act, fix, cache, kept)
            return out, glimpse, den.commit(cache, kv)["kv"]

    def belief(p):
        _, nxt = model.record.scan(model.record.input(p.p, p.g, fix), done, state)
        emb = model.belief.percept_embed(p, nxt)
        return model.belief(model.belief.tokens(emb, r, done, act), carry)

    def step():
        with TR.autocast(DEV, cfg):
            p = model.perception(frames, fix)
            return world(), belief(p)

    baseline = step()
    result = {"eager_native": time_block(step, count), "denoiser_full_plus_crop_eager": time_block(world, count)}
    original = M.FunctionalMamba3.forward
    try:
        M.FunctionalMamba3.forward = fixed_forward
        fixed = step()
        result["fixed_shape_max_abs_delta"] = differences(baseline, fixed)
        result["eager_fixed_shape"] = time_block(step, count)
        if graphs:
            graph, output = capture(step)
            graph.replay()
            result["captured_max_abs_delta"] = differences(fixed, output)
            result["cuda_graph"] = time_block(graph.replay, count)
            del graph, output
    finally:
        M.FunctionalMamba3.forward = original
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 32, 64])
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--eager-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    results = {}
    for b in args.batch:
        gc.collect()
        torch.cuda.empty_cache()
        results[str(b)] = benchmark(model, cfg, b, args.count, not args.eager_only)
        print(b, results[str(b)], flush=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
