"""Drift rows, DEFAR's simulated inference step (PLAN.md): P1.5 continued from checkpoint 10 by C14's exact recipe plus
drift rows; checkpoint 11 in defar_run/. C14's run is the matched control. The package is untouched: the trainer's
bootstrap_rows and shortcut_loss are wrapped here.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261122_defar.defar check      # exactness, cost, memory
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261122_defar.defar train
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261122_defar.defar readings
"""
import copy, importlib, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from saccade import model as M, train as TR

HERE = Path(__file__).resolve().parent
C14 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c14_lr")
SRC, RUN, CTRL = C14.SRC, HERE / "defar_run", C14.RUN / "ckpt_11.pt"
DRIFT = {"fraction": 0.25, "seed": 2606, "step": 2, "beta": 1.0}   # step: grid steps of 1 / k_max (2 -> 1/4, the sampler's)
ORIG_LOSS, ORIG_ROWS = M.shortcut_loss, TR.bootstrap_rows
CLAMP = [None]                                                     # the sampler's clamp, set from the config


def rows_with_drift(cfg, world_seed, update):
    """bootstrap_rows' own draw (1 = bootstrap) plus drift rows (2) among the rest, by their own generator."""
    b = ORIG_ROWS(cfg, world_seed, update)
    out = b.astype(np.int8)
    k = round(DRIFT["fraction"] * len(b))
    out[np.random.default_rng([DRIFT["seed"], world_seed, update]).choice(np.flatnonzero(~b), k, replace=False)] = 2
    return out


def two_stream(den, xc, xq, code_c, code_q, prev_action, memory, fix, episode, valid):
    """Denoiser.estimate's two streams, with gradients and the trainer's activation checkpointing: stream c is the chunk
    in its context role (as forward() sees it), stream q each frame's own input, whose temporal window reads c's keys /
    values for the earlier frames and its own for the frame itself. Returns (c's outputs, q's outputs)."""
    (tc, ang), (tq, _) = den.embed(xc, memory, fix), den.embed(xq, memory, fix)
    cc, cq = den.conds(code_c, prev_action), den.conds(code_q, prev_action)
    win, ang_t = den._window(tc.shape[1], episode, valid, tc.device)
    ang_t = ang_t.expand(tc.shape[0], -1, -1)
    for blk in den.blocks:
        def f(a, b, blk=blk):
            if blk.temporal:
                a2, kv = blk(a, cc, ang_t, win, None, den.phys)
                return a2, blk(b, cq, ang_t, win, None, den.phys, ctx=kv)[0]
            return blk(a, cc, ang, None, None, den.phys)[0], blk(b, cq, ang, None, None, den.phys)[0]
        tc, tq = checkpoint(f, tc, tq, use_reentrant=False) if den.checkpoint and den.training and torch.is_grad_enabled() else f(tc, tq)
    return den.head(tc, cc), den.head(tq, cq)


def drift_rows(den, x0, pa, memory, fix, episode, valid, sc, levels, noise):
    """The drift rows' per-frame loss [B, T]: the flow term (stream c, the ordinary inputs) plus beta x the drift term
    (stream q). For frames at tau1 >= step / k_max: the real frame noised to tau0 = tau1 - step / k_max with the same
    noise, one no-gradient finest-code evaluation, one sampler step of step / k_max to tau1, then the finest-code estimate
    there against the real frame, ramp-weighted in x-space, physical group only."""
    k, ds, b = sc["k_max"], DRIFT["step"], M._bcast
    tau = [s.float() / k for s in M._parts(tuple(l[0] for l in levels))]            # p, g, v
    zt = tuple((1 - b(t, x)) * n + b(t, x) * x for x, n, t in zip(x0, noise, tau))
    code = tuple(M.shortcut_code(s_, e_, k) for s_, e_ in levels)                    # (physical, glimpse)
    s1 = levels[0][0]; can = s1 >= ds
    s0 = torch.where(can, s1 - ds, s1); t0 = s0.float() / k
    z0 = [((1 - b(t0, x)) * n + b(t0, x) * x).to(z.dtype) for x, n, z in ((x0[0], noise[0], zt[0]), (x0[2], noise[2], zt[2]))]
    with torch.no_grad():
        mem = tuple(m.detach() for m in memory) if isinstance(memory, tuple) else memory.detach()
        e0 = two_stream(den, zt, (z0[0], zt[1], z0[1]), code, (M.shortcut_code(s0, levels[0][1], k), code[1]), pa, mem, fix, episode, valid)[1]
        move = lambda z, xh: z.float() + (xh.float().clamp(-CLAMP[0], CLAMP[0]) - z.float()) / b(1 - t0, z) * (ds / k)
        zd = [torch.where(b(can, z), move(z, xh), zz.float()).to(zz.dtype) for z, xh, zz in ((z0[0], e0[0], zt[0]), (z0[1], e0[2], zt[2]))]
    oc, oq = two_stream(den, zt, (zd[0], zt[1], zd[1]), code, code, pa, memory, fix, episode, valid)
    w = [sc["ramp"][0] * t + sc["ramp"][1] for t in tau]
    n_el = sum(t[0, 0].numel() for t in x0)                                          # shortcut_loss's per-frame mean
    per_frame = lambda terms: sum(e.flatten(2).sum(-1) for e in terms) / n_el
    flow = per_frame([b(wt, x) * (o.float() - x) ** 2 for o, x, wt in zip(oc, x0, w)])
    drift = per_frame([b(w[0], x0[0]) * (oq[0].float() - x0[0]) ** 2, b(w[2], x0[2]) * (oq[2].float() - x0[2]) ** 2])
    STATS.append((float(drift.detach()[can].mean()) if can.any() else float("nan"), float(flow.detach().mean())))
    return flow + DRIFT["beta"] * drift * can


STATS = []


def shortcut_loss(den, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
    """shortcut_loss with drift rows (bootstrap codes: 0 flow, 1 bootstrap, 2 drift). Levels and noise are drawn exactly
    as shortcut_loss draws them, so the training RNG runs as in C14; non-drift rows go to shortcut_loss itself."""
    if bootstrap.dtype == torch.bool:
        return ORIG_LOSS(den, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels, noise)
    boot, drift = bootstrap == 1, bootstrap == 2
    k = sc["k_max"]; B, T = x0[0].shape[:2]; dev = x0[0].device
    levels = levels or [M.shortcut_levels((B, T), k, boot, dev) for _ in range(2)]
    noise = noise or tuple(torch.randn_like(t) for t in x0)

    def part(m):
        s = lambda a: a[m]
        mem = tuple(s(z) for z in memory) if isinstance(memory, tuple) else s(memory)
        return (tuple(s(z) for z in x0), s(prev_action), mem, s(fix), s(episode), s(valid), sc), \
            [(s(a), s(c)) for a, c in levels], tuple(s(z) for z in noise)
    loss = torch.zeros(B, T, device=dev)
    if (~drift).any():
        args, lv, nz = part(~drift)
        loss = loss.index_put(((~drift).nonzero()[:, 0],), ORIG_LOSS(den, *args, boot[~drift], start, lv, nz))
    if drift.any():
        args, lv, nz = part(drift)
        loss = loss.index_put((drift.nonzero()[:, 0],), drift_rows(den, *args[:-1], sc, lv, nz))
    return loss


def install(cfg):
    CLAMP[0] = cfg["perception"]["periphery"]["tanh_clamp"]
    M.shortcut_loss, TR.bootstrap_rows = shortcut_loss, rows_with_drift


def check():
    """(1) two_stream with q = c equals forward(); (2) a drift row's flow term equals shortcut_loss's on the same row, levels
    and noise; (3) C14's recipe for a few updates with and without drift rows: seconds per update and peak memory."""
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
    model, cfg = AI.load(SRC / "ckpt_10.pt"); den = model.denoiser; den.train()
    install(cfg); dev = torch.device("cuda"); sc = cfg["edm"]["shortcut"]
    g = torch.Generator(device=dev).manual_seed(0)
    B, T = 2, 24
    r = lambda *s: torch.randn(*s, device=dev, generator=g)
    x0 = (r(B, T, 81, 16).tanh(), r(B, T, 9, 64).tanh(), r(B, T, 8).tanh())
    mem = (r(B, T, den.K, den.W), r(B, T, 81, den.record_read[0].in_features))
    pa = torch.randint(0, 17, (B, T), device=dev, generator=g); fix = torch.randint(0, 49, (B, T), device=dev, generator=g)
    ep, va = torch.zeros(B, T, dtype=torch.long, device=dev), torch.ones(B, T, dtype=torch.bool, device=dev)
    levels = [M.shortcut_levels((B, T), sc["k_max"], torch.zeros(B, dtype=torch.bool, device=dev), dev) for _ in range(2)]
    noise = tuple(torch.randn(t.shape, device=dev, generator=g) for t in x0)
    out = {}
    with torch.no_grad(), TR.autocast(dev, cfg):
        code = tuple(M.shortcut_code(s_, e_, sc["k_max"]) for s_, e_ in levels)
        zt = tuple((1 - M._bcast(t, x)) * n + M._bcast(t, x) * x for x, n, t in
                   zip(x0, noise, [s.float() / sc["k_max"] for s in M._parts(tuple(l[0] for l in levels))]))
        f = den(zt, code, pa, mem, fix, ep, va)
        c, q = two_stream(den, zt, zt, code, code, pa, mem, fix, ep, va)
        out["two_stream_c_vs_forward"] = max(float((a.float() - b_.float()).abs().max()) for a, b_ in zip(f, c))
        out["two_stream_q_vs_forward"] = max(float((a.float() - b_.float()).abs().max()) for a, b_ in zip(f, q))
        orig = ORIG_LOSS(den, x0, pa, mem, fix, ep, va, sc, torch.zeros(B, dtype=torch.bool, device=dev), True, levels, noise)
        DRIFT_beta, DRIFT["beta"] = DRIFT["beta"], 0.0
        mine = drift_rows(den, x0, pa, mem, fix, ep, va, sc, levels, noise)
        DRIFT["beta"] = DRIFT_beta
        out["flow_term_vs_shortcut_loss"] = float((orig - mine).abs().max() / orig.abs().max())
    print(json.dumps(out), flush=True)
    del model, den; torch.cuda.empty_cache()
    for arm in ("control", "drift"):                                  # C14's recipe, a few updates, from a scratch copy
        run = Path(os.environ["DEFAR_SCRATCH"]) / f"check_{arm}"
        shutil.rmtree(run, ignore_errors=True); run.mkdir()
        shutil.copyfile(SRC / "ckpt_10.pt", run / "latest.pt")
        for fn in ("gaze.bin", "chooser.jsonl", "log.jsonl"):
            shutil.copyfile(SRC / fn, run / fn)
        ck = torch.load(SRC / "ckpt_10.pt", map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(ck["cfg"]); total, frac = cfg["p1"]["frames"], cfg["p1"]["checkpoint_every_fraction"]
        f5 = TR.lr_factor(5 * total * frac, cfg)
        cfg["p1"] = dict(cfg["p1"], frames=int(round(total * 1.1)), checkpoint_every_fraction=1 / 11)
        M.shortcut_loss, TR.bootstrap_rows = ORIG_LOSS, ORIG_ROWS
        if arm == "drift":
            install(cfg)
        lr0, vr0 = TR.lr_factor, TR.validate_resume
        TR.lr_factor = lambda frames, cfg_: f5; TR.validate_resume = lambda *a, **k: None
        torch.cuda.reset_peak_memory_stats(); STATS.clear()
        TR.p1(cfg, ck["world_seed"], run, updates=ck["state"]["update"] + 50, log_every=25)   # the 2nd log: steady state
        TR.lr_factor, TR.validate_resume = lr0, vr0
        logs = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()][-1]
        print(arm, json.dumps({k: logs.get(k) for k in ("update", "sec_per_update", "peak_gib", "g1")}),
              "drift/flow per frame", np.nanmean([s[0] for s in STATS]) if STATS else None, np.mean([s[1] for s in STATS]) if STATS else None,
              flush=True)
    M.shortcut_loss, TR.bootstrap_rows = ORIG_LOSS, ORIG_ROWS


def train():
    """C14's train() with drift rows installed and its run directory moved here."""
    ck = torch.load(SRC / "ckpt_10.pt", map_location="cpu", weights_only=False)
    install(ck["cfg"])
    C14.RUN = RUN
    C14.train()
    (HERE / "drift_stats.json").write_text(json.dumps({"calls": len(STATS), "drift_term_mean": float(np.nanmean([x[0] for x in STATS])),
                                                     "flow_term_mean": float(np.mean([x[1] for x in STATS]))}, indent=1) + "\n")


if __name__ == "__main__":
    {"check": check, "train": train}[sys.argv[1]]()
