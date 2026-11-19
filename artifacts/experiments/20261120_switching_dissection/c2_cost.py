"""C2's cost (PLAN.md, "C2's cost and cache"): what a world-eye would add to P1's run, and what a cache would save.
Everything is timed on P1.5 checkpoint 10, on its own training streams, at the trainer's shapes (16 streams, 8 prefix
frames plus 64 or 256).
- **The eye's estimate:** per target frame, one periphery evaluation at signal 0 (the sampler's first step) and the
  frame's context pass, sequential over the chunk. The memory tokens and record come from a belief / record pass that
  the trainer already makes; computed here, not timed.
- **Cache read:** fp16 periphery estimates from a memmap, one contiguous run per stream.
- **Cache fill:** the same eye pass offline at 64 streams; time per frame.

Projections use the run's own measured update times (2.284 s at 64, 8.581 s at 256; 12,500 and 3,125 updates).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c2_cost SCRATCH_DIR
"""
import dataclasses, importlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P15 = ROOT / "artifacts/experiments/20261116_p15/run/ckpt_10.pt"
AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
S = importlib.import_module("saccade.eval.s1")
WORLD_SEED = importlib.import_module("artifacts.experiments.20261116_p15.train").WORLD_SEED
BASE = {64: (2.284, 12500), 256: (8.581, 3125)}


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def inputs(model, cfg, chunk, dev):
    """Perception, record (denoiser read) and predicted beliefs for every frame of the chunk (untimed)."""
    c = TR.tensors(chunk, dev)
    with TR.autocast(dev, cfg):
        parts = [model.perception(c["frames"][:, a:a + 64], c["fixation"][:, a:a + 64]) for a in range(0, c["frames"].shape[1], 64)]
        P = M.Percept(*(torch.cat(z, 1) for z in zip(*[p.parts() for p in parts])), c["fixation"])
        rins, bps = [], []
        for r in torch.arange(P.p.shape[0], device=dev).split(2):           # two streams at a time, as the trainer's
            Pr = M.Percept(P.p[r], P.g[r], P.v[r], P.fix[r])                  # 256-step microbatches (memory)
            rin, rst = model.record.scan(Pr.p, c["first"][r], model.record.zero(len(r), dev))
            y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(Pr, rst), c["reward"][r], c["terminated"][r], c["action"][r]),
                                model.belief.initial(len(r), dev))
            rins.append(rin.to(torch.bfloat16)); bps.append(torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)); del rst, rin
    return c, P, torch.cat(rins), torch.cat(bps)


@torch.no_grad()
def eye_pass(model, cfg, c, P, rin, bp, prefix, dev):
    """The world-eye's zero-signal periphery estimate of every target frame, sequentially; returns [B, S, 81, 16]."""
    den, k = model.denoiser, cfg["edm"]["shortcut"]["k_max"]
    _, e, _ = M.shortcut_ladder(cfg)
    B, L = P.p.shape[:2]
    g = torch.Generator(device=dev).manual_seed(7)
    code = torch.full((B, 1), M.shortcut_code(0, e, k), device=dev)
    cache, out = den.new_cache(), []
    with TR.autocast(dev, cfg):
        for t in range(L):
            mem = S.cond(model.belief.memory(bp[:, t:t + 1]), rin[:, t:t + 1])
            if t >= prefix:
                z = (torch.randn(P.p[:, t:t + 1].shape, device=dev, generator=g), torch.zeros_like(P.g[:, t:t + 1]),
                     torch.randn(P.v[:, t:t + 1].shape, device=dev, generator=g))
                o, _ = den.step(z, code, c["prev_action"][:, t:t + 1], mem, P.fix[:, t:t + 1], cache)
                out.append(o[0])
            cache = M.context_pass(den, tuple(x[:, t:t + 1].float() for x in P.parts()), c["prev_action"][:, t:t + 1], mem,
                                   P.fix[:, t:t + 1], cache, cfg, g)
    return torch.cat(out, 1)


def main(scratch):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=True)
    model, cfg = AI.load(P15)
    model.requires_grad_(False).eval()
    dev = next(model.parameters()).device
    d, prefix = cfg["data"], cfg["data"]["context_prefix"]
    ck = torch.load(P15, map_location="cpu", weights_only=False)
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], prefix, seed=WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    res = {"baseline": BASE}
    for Sx in (64, 256):
        times = []
        for rep in range(4):                                              # 1 warm-up, 3 timed
            chunk = streams.next(Sx)
            c, P, rin, bp = inputs(model, cfg, chunk, dev)
            torch.cuda.reset_peak_memory_stats()
            t0 = sync(); est = eye_pass(model, cfg, c, P, rin, bp, prefix, dev); t1 = sync()
            assert est.shape[1] == Sx
            if rep:
                times.append(t1 - t0)
        res[f"eye_{Sx}"] = {"sec_per_update": float(np.mean(times)), "runs": times, "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                            "fraction_of_update": float(np.mean(times)) / BASE[Sx][0]}
        print(Sx, res[f"eye_{Sx}"], flush=True)
    # offline fill: 64 streams (a 64-step chunk repeated 4 times along the batch; 256-step rows at 64 streams exceed 6 GB)
    chunk = streams.next(64)
    c, P, rin, bp = inputs(model, cfg, chunk, dev)
    rep4 = lambda x: x.repeat(4, *([1] * (x.dim() - 1)))
    c4 = {"prev_action": rep4(c["prev_action"])}
    P4 = M.Percept(*(rep4(x) for x in P.parts()), rep4(P.fix))
    eye_pass(model, cfg, c4, P4, rep4(rin), rep4(bp), prefix, dev)          # warm-up at this shape
    t0 = sync(); eye_pass(model, cfg, c4, P4, rep4(rin), rep4(bp), prefix, dev); t1 = sync()
    res["fill_64_streams"] = {"sec_per_frame": (t1 - t0) / (64 * 64), "in_loop_sec_per_frame_256": res["eye_256"]["sec_per_update"] / (16 * 256),
                              "in_loop_sec_per_frame_64": res["eye_64"]["sec_per_update"] / (16 * 64)}
    print("fill", res["fill_64_streams"], flush=True)
    # cache read: a memmap of fp16 periphery estimates, one contiguous run of S frames per stream
    n = 400_000
    mm = np.lib.format.open_memmap(scratch / "cache.npy", mode="w+", dtype=np.float16, shape=(n, 81, 16))
    for a in range(0, n, 50_000):
        mm[a:a + 50_000] = np.float16(0.5)                                # real pages (content irrelevant to timing)
    mm.flush(); del mm
    mm = np.load(scratch / "cache.npy", mmap_mode="r")
    rng = np.random.default_rng(1)
    for Sx in (64, 256):
        ts = []
        for _ in range(20):
            starts = rng.integers(0, n - Sx, 16)
            t0 = time.perf_counter(); x = torch.as_tensor(np.stack([mm[s:s + Sx] for s in starts])).to(dev, non_blocking=False); t1 = sync()
            ts.append(t1 - t0)
        res[f"cache_read_{Sx}"] = {"sec_per_update": float(np.median(ts))}
        print("read", Sx, res[f"cache_read_{Sx}"], flush=True)
    (scratch / "cache.npy").unlink()
    # projections
    fresh_h = (res["eye_64"]["sec_per_update"] * BASE[64][1] + res["eye_256"]["sec_per_update"] * BASE[256][1]) / 3600
    base_h = (BASE[64][0] * BASE[64][1] + BASE[256][0] * BASE[256][1]) / 3600
    frames = 25_600_000; passes = 10
    fill_h_per_pass = res["fill_64_streams"]["sec_per_frame"] * frames / passes / 3600
    read_h = (res["cache_read_64"]["sec_per_update"] * BASE[64][1] + res["cache_read_256"]["sec_per_update"] * BASE[256][1]) / 3600
    res["projection_hours"] = {"baseline_run": base_h, "fresh_eye_extra": fresh_h,
                               "cache": {f"refresh_every_{kk}_passes": {"extra": read_h + fill_h_per_pass * passes / kk,
                                                                        "saved_vs_fresh": fresh_h - (read_h + fill_h_per_pass * passes / kk)}
                                         for kk in (1, 2, 5, 10)}}
    print(json.dumps(res["projection_hours"], indent=1), flush=True)
    (HERE / "c2_cost.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
