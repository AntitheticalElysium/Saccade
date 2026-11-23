"""What one awake, watched and dreamed step costs. PLAN.md (2026-11-23).

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_step_costs.step_costs
    JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_step_costs.step_costs --craftax
"""
import argparse
import importlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def timed(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return float(np.median(out) * 1e3)                                      # ms


@torch.no_grad()
def world(B, reps):
    from saccade import chooser as C, model as M, train as TR
    from saccade.eval import s1 as S
    S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    dev, den = S2.DEV, model.denoiser
    g = torch.Generator(device=dev).manual_seed(0)
    frames = torch.randint(0, 256, (B, 63, 63, 3), dtype=torch.uint8, device=dev)
    fix = C.CENTRES.to(dev)[torch.randint(49, (B,), device=dev)]
    with TR.autocast(dev, cfg):
        P = model.perception(frames[:, None], fix[:, None])
    x = tuple(t.float() for t in (P.p, P.g, P.v))                          # [B, 1, ...]
    dims = [tuple(t.shape[2:]) for t in x]
    pa = torch.randint(17, (B, 1), device=dev)
    r, d, a = torch.zeros(B, device=dev), torch.zeros(B, dtype=torch.bool, device=dev), torch.randint(17, (B,), device=dev)
    carry = model.belief.initial(B, dev)
    rs = model.record.zero(B, dev)
    bp = torch.randn(B, model.belief.memory_out.in_features, device=dev)
    with TR.autocast(dev, cfg):
        mem = S.cond(model.belief.memory(bp)[:, None], rs[:, None])
        cache = den.new_cache()
        for _ in range(cfg["data"]["context_prefix"]):                      # a full 8-frame cache
            cache = M.context_pass(den, x, pa, mem, fix[:, None], cache, cfg, g)
    k = den.k_max
    tc = cfg["edm"]["shortcut"]["context_signal"]
    code = torch.full((B, 1), M.shortcut_code(tc, k.bit_length() - 1, k), device=dev)
    res = {}

    def world_step():
        with TR.autocast(dev, cfg):
            _, new = den.step(x, code, pa, mem, fix[:, None], cache)
            den.commit(cache, new)
    res["world_step_ms"] = timed(world_step, reps)

    kept = []
    with TR.autocast(dev, cfg):
        den.step(x, code, pa, mem, fix[:, None], cache, kept)

    def glimpse_only():
        with TR.autocast(dev, cfg):
            den.glimpse_step(x[1], code, pa, fix[:, None], cache, kept)
    res["glimpse_tokens_ms"] = timed(glimpse_only, reps)

    def encode():
        with TR.autocast(dev, cfg):
            model.perception(frames[:, None], fix[:, None])
    res["glimpse_encoder_ms"] = timed(encode, reps)

    def belief_step():
        with TR.autocast(dev, cfg):
            reads, _, _ = S.history(model, x[0][:, 0], state=rs)
            emb = S.embed(model, M.Percept(x[0][:, 0], x[1][:, 0], x[2][:, 0], fix), reads)
            model.belief(model.belief.tokens(emb[:, None], r[:, None], d[:, None], a[:, None]), carry)
    res["record_belief_ms"] = timed(belief_step, reps)

    newfix = C.CENTRES.to(dev)[torch.randint(49, (B,), device=dev)][:, None]

    def dream():
        with TR.autocast(dev, cfg):
            parts, fx = M.sample(den, dims, pa, mem, fix[:, None], cache, cfg, g, choose=lambda p_, v_: newfix)
            parts = tuple(t[:, 0].float() for t in parts)
            reads0, _, _ = S.history(model, parts[0], state=rs)
            h = model.outcome.features(bp, S.embed(model, M.Percept(*parts, fx[:, 0]), reads0)).float()
            rr = model.outcome.sample_reward(h, g)
            torch.sigmoid(model.outcome.cont_logit(h, rr).float())
            emb = S.embed(model, M.Percept(*parts, fx[:, 0]), reads0)
            model.belief(model.belief.tokens(emb[:, None], rr[:, None], d[:, None], a[:, None]), carry)
            M.context_pass(den, tuple(t[:, None] for t in parts), pa, mem, fx, cache, cfg, g)
    res["dream_frame_ms"] = timed(dream, reps)
    res["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    return res


def craftax(reps):
    import jax
    import jax.numpy as jnp
    RP = importlib.import_module("artifacts.eda.replay")
    _, _, reset, step, frame = RP.env_and_render()
    key = jax.random.PRNGKey(0)
    _, s = reset(key)
    for i in range(20):                                                    # compile and warm up
        _, s, _, _, _ = step(jax.random.fold_in(key, i), s, jnp.int32(i % 17))
        np.asarray(frame(s))
    out = []
    for i in range(reps):
        t0 = time.perf_counter()
        _, s, _, d, _ = step(jax.random.fold_in(key, 100 + i), s, jnp.int32(i % 17))
        np.asarray(frame(s))
        out.append(time.perf_counter() - t0)
        if bool(d):
            _, s = reset(jax.random.fold_in(key, 10 ** 6 + i))
    return {"craftax_step_render_ms": float(np.median(out) * 1e3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craftax", action="store_true")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    if args.craftax:
        res = craftax(max(args.reps, 200))
        (HERE / "craftax.json").write_text(json.dumps(res, indent=1) + "\n")
    else:
        res = {f"batch_{B}": world(B, args.reps) for B in (1, 16, 64)}
        (HERE / "step_costs.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
