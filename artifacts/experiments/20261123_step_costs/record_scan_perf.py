"""Fuse P1.5's fixed, gaze-independent spatial record update for a matched benchmark."""

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from saccade import model as M
from . import streaming_perf as P

HERE = Path(__file__).resolve().parent


@triton.jit
def record_kernel(PX, FIRST, STATE, A, B, BEFORE, AFTER,
                  channels: tl.constexpr, modes: tl.constexpr, n: tl.constexpr, steps: tl.constexpr,
                  size: tl.constexpr, px_b: tl.constexpr, px_t: tl.constexpr, px_n: tl.constexpr,
                  px_c: tl.constexpr, st_b: tl.constexpr, st_n: tl.constexpr, st_d: tl.constexpr,
                  first_b: tl.constexpr, first_t: tl.constexpr, block: tl.constexpr):
    lane = tl.program_id(0) * block + tl.arange(0, block)
    half: tl.constexpr = channels * modes
    width: tl.constexpr = half * 2
    mode = lane % modes
    channel = lane // modes % channels
    cell = lane // half % n
    row = lane // (half * n)
    j = lane % half
    valid = lane < size
    ar, ai = tl.load(A + 2 * mode), tl.load(A + 2 * mode + 1)
    br, bi = tl.load(B + 2 * mode), tl.load(B + 2 * mode + 1)
    offset = row * st_b + cell * st_n + j * st_d
    sr = tl.load(STATE + offset, valid, 0)
    si = tl.load(STATE + offset + half * st_d, valid, 0)
    for t in range(steps):
        reset = tl.load(FIRST + row * first_b + t * first_t, valid, 0)
        sr = tl.where(reset, 0., sr)
        si = tl.where(reset, 0., si)
        out = (((row * steps + t) * n + cell) * width) + j
        tl.store(BEFORE + out, sr, valid)
        tl.store(BEFORE + out + half, si, valid)
        x = tl.load(PX + row * px_b + t * px_t + cell * px_n + channel * px_c, valid, 0).to(tl.float32)
        nr = (ar * sr - ai * si) + br * x
        ni = (ar * si + ai * sr) + bi * x
        sr, si = nr, ni
        tl.store(AFTER + out, sr, valid)
        tl.store(AFTER + out + half, si, valid)


NATIVE_SCAN = M.Record.scan


@torch.no_grad()
def fused_scan(self, p, first, state):
    if self.fovea or not p.is_cuda:
        return NATIVE_SCAN(self, p, first, state)
    b, steps, n, channels = p.shape
    before = torch.empty((b, steps, n, self.dim), dtype=torch.float32, device=p.device)
    after = torch.empty_like(before)
    size = b * n * channels * self.a.numel()
    record_kernel[(triton.cdiv(size, 128),)](p, first, state, torch.view_as_real(self.a), torch.view_as_real(self.b),
                                           before, after, channels, self.a.numel(), n, steps, size,
                                           *p.stride(), *state.stride(), *first.stride(), 128,
                                           enable_fp_fusion=False)
    return before, after


def benchmark(model, batch, steps):
    torch.cuda.reset_peak_memory_stats()
    dev = P.DEV
    rng = torch.Generator(device=dev).manual_seed(20261123)
    rec = model.record
    x = torch.randn(batch, steps, M.N, rec.channels, device=dev, dtype=torch.bfloat16, generator=rng)
    state = 0.1 * torch.randn(batch, M.N, rec.dim, device=dev, generator=rng)
    first = torch.rand(batch, steps, device=dev, generator=rng) < 0.02
    native = lambda: NATIVE_SCAN(rec, x, first, state)
    fused = lambda: fused_scan(rec, x, first, state)
    a, b = native(), fused()
    error = [{"max": float((y - z).abs().max()), "rms": float((y - z).square().mean().sqrt())} for y, z in zip(a, b)]
    del a, b
    return {"native": P.time_block(native, 5, repeats=4), "fused": P.time_block(fused, 5, repeats=4),
            "errors_before_and_after": error, "has_resets": bool(first.any().item()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    model, _ = P.S2.AI.load(P.S2.P15)
    model.requires_grad_(False).eval()
    out = {}
    for batch, steps in ((8, 72), (2, 264), (64, 8), (64, 1)):
        out[f"{batch}x{steps}"] = benchmark(model, batch, steps)
        print(batch, steps, out[f"{batch}x{steps}"], flush=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
