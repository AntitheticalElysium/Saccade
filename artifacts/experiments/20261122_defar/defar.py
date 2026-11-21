"""The drift-row fine-tune, DEFAR's simulated inference step (PLAN.md): P1.5 continued from checkpoint 10 by the
higher-learning-rate continuation's exact recipe, plus drift rows; checkpoint 11 in defar_run/. The package is untouched:
the trainer's bootstrap_rows and shortcut_loss are wrapped here.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261122_defar.defar check      # exactness, cost, memory
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261122_defar.defar train
"""
import copy, importlib, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from saccade import model as M, train as TR

HERE = Path(__file__).resolve().parent
C14 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c14_lr")
SRC, RUN, CONT = C14.SRC, HERE / "defar_run", C14.RUN / "ckpt_11.pt"   # CONT: the continuation's checkpoint 11
DRIFT = {"fraction": 0.25, "seed": 2606, "dest": (4, 6), "beta": 1.0}   # dest: destination signal indices (k_max = 8)
ORIG_LOSS, ORIG_ROWS = M.shortcut_loss, TR.bootstrap_rows
CFG = {}                                                           # the sampler's clamp, step and step code, from install()
TERMS = [HERE / "drift_terms.jsonl"]


def rows_with_drift(cfg, world_seed, update):
    """bootstrap_rows' own draw (1 = bootstrap) plus drift rows (2) among the rest, by their own generator, at most one per
    micro-batch of the 256-step chunks (the trainer's np.array_split of the streams), so no micro-batch carries two
    drift rows' two-stream passes at once."""
    b = ORIG_ROWS(cfg, world_seed, update)
    out = b.astype(np.int8)
    k = round(DRIFT["fraction"] * len(b))
    rng = np.random.default_rng([DRIFT["seed"], world_seed, update])
    groups = [g[~b[g]] for g in np.array_split(np.arange(len(b)), cfg["data"]["accumulate_at_256"])]
    for i in rng.choice([i for i, g in enumerate(groups) if len(g)], k, replace=False):
        out[rng.choice(groups[i])] = 2
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
    (stream q). Every frame's query goes to a destination level, 4/8 on even frames and 6/8 on odd ones. The real frame
    is noised to one sampler step below it (2/8, 4/8) with the frame's own noise; one no-gradient evaluation by the
    deployed quarter-step output and the sampler's own step (with its clamp) give the drifted state; then the finest
    output there against the real frame, ramp-weighted in x-space, physical group only."""
    k, e_max, b = sc["k_max"], sc["k_max"].bit_length() - 1, M._bcast
    tau = [s.float() / k for s in M._parts(tuple(l[0] for l in levels))]            # p, g, v
    zt = tuple((1 - b(t, x)) * n + b(t, x) * x for x, n, t in zip(x0, noise, tau))
    code = tuple(M.shortcut_code(s_, e_, k) for s_, e_ in levels)                    # (physical, glimpse)
    B, T = levels[0][0].shape
    lo, hi = DRIFT["dest"]
    s1 = torch.where(torch.arange(T, device=zt[0].device) % 2 == 0, lo, hi).expand(B, T)   # destination, no randomness
    s0 = s1 - round(CFG["d"] * k); t0, t1 = s0.float() / k, s1.float() / k
    z0 = [((1 - b(t0, x)) * n + b(t0, x) * x).to(z.dtype) for x, n, z in ((x0[0], noise[0], zt[0]), (x0[2], noise[2], zt[2]))]
    with torch.no_grad():
        mem = tuple(m.detach() for m in memory) if isinstance(memory, tuple) else memory.detach()
        e0 = two_stream(den, zt, (z0[0], zt[1], z0[1]), code, (M.shortcut_code(s0, torch.full_like(s0, CFG["e_step"]), k), code[1]),
                        pa, mem, fix, episode, valid)[1]
        move = lambda z, xh: z.float() + (xh.float().clamp(-CFG["clamp"], CFG["clamp"]) - z.float()) / b(1 - t0, z) * CFG["d"]
        zd = [move(z, xh).to(z.dtype) for z, xh in ((z0[0], e0[0]), (z0[1], e0[2]))]
    oc, oq = two_stream(den, zt, (zd[0], zt[1], zd[1]), code, (M.shortcut_code(s1, torch.full_like(s1, e_max), k), code[1]),
                        pa, memory, fix, episode, valid)
    w, w1 = [sc["ramp"][0] * t + sc["ramp"][1] for t in tau], sc["ramp"][0] * t1 + sc["ramp"][1]
    n_el = sum(t[0, 0].numel() for t in x0)                                          # shortcut_loss's per-frame mean
    per_frame = lambda terms: sum(e.flatten(2).sum(-1) for e in terms) / n_el
    flow = per_frame([b(wt, x) * (o.float() - x) ** 2 for o, x, wt in zip(oc, x0, w)])
    drift = per_frame([b(w1, x0[0]) * (oq[0].float() - x0[0]) ** 2, b(w1, x0[2]) * (oq[2].float() - x0[2]) ** 2])
    if TERMS[0] is not None:                                                         # the in-run reading (PLAN.md)
        with open(TERMS[0], "a") as fh:
            fh.write(json.dumps({"drift_4": float(drift.detach()[:, 0::2].mean()), "drift_6": float(drift.detach()[:, 1::2].mean()),
                                 "flow": float(flow.detach().mean())}) + "\n")
    return flow + DRIFT["beta"] * drift


def shortcut_loss(den, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
    """shortcut_loss with drift rows (bootstrap codes: 0 flow, 1 bootstrap, 2 drift). Levels and noise are drawn exactly
    as shortcut_loss draws them, so the training RNG runs as in the continuation; non-drift rows go to shortcut_loss itself."""
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
    _, e_step, d = M.shortcut_ladder(cfg)
    CFG.update(clamp=cfg["perception"]["periphery"]["tanh_clamp"], e_step=e_step, d=d)
    M.shortcut_loss, TR.bootstrap_rows = shortcut_loss, rows_with_drift


def check():
    """(1) two_stream with q = c equals forward(); (2) a drift row's flow term equals shortcut_loss's on the same row, levels
    and noise; (3) the continuation's recipe with drift rows for 50 updates from a scratch copy: seconds per update and
    peak memory (the continuation itself logged 9.27 s and 3.23 GiB), and the drift terms it writes."""
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    scratch = Path(os.environ["DEFAR_SCRATCH"])
    AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
    model, cfg = AI.load(SRC / "ckpt_10.pt"); den = model.denoiser; den.train()
    install(cfg); dev = torch.device("cuda"); sc = cfg["edm"]["shortcut"]
    TERMS[0] = None
    g = torch.Generator(device=dev).manual_seed(0)
    B, T = 2, 24
    r = lambda *s: torch.randn(*s, device=dev, generator=g)
    x0 = (r(B, T, 81, 16).tanh(), r(B, T, 9, 64).tanh(), r(B, T, 8).tanh())
    mem = (r(B, T, den.K, den.W), r(B, T, 81, den.record_read[0].in_features))
    pa = torch.randint(0, 17, (B, T), device=dev, generator=g); fix = torch.randint(1, 8, (B, T, 2), device=dev, generator=g)   # window centres (row, col)
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
        beta, DRIFT["beta"] = DRIFT["beta"], 0.0
        mine = drift_rows(den, x0, pa, mem, fix, ep, va, sc, levels, noise)
        DRIFT["beta"] = beta
        out["flow_term_vs_shortcut_loss"] = float((orig - mine).abs().max() / orig.abs().max())
    print(json.dumps(out), flush=True)
    del model, den; torch.cuda.empty_cache()
    run = scratch / "check_drift"
    shutil.rmtree(run, ignore_errors=True); run.mkdir()
    shutil.copyfile(SRC / "ckpt_10.pt", run / "latest.pt")
    for fn in ("gaze.bin", "chooser.jsonl", "log.jsonl"):
        shutil.copyfile(SRC / fn, run / fn)
    ck = torch.load(SRC / "ckpt_10.pt", map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ck["cfg"]); total, frac = cfg["p1"]["frames"], cfg["p1"]["checkpoint_every_fraction"]
    f5 = TR.lr_factor(5 * total * frac, cfg)
    cfg["p1"] = dict(cfg["p1"], frames=int(round(total * 1.1)), checkpoint_every_fraction=1 / 11)
    TERMS[0] = scratch / "check_drift_terms.jsonl"; TERMS[0].unlink(missing_ok=True)
    TR.lr_factor = lambda frames, cfg_: f5; TR.validate_resume = lambda *a, **k: None
    torch.cuda.reset_peak_memory_stats()
    TR.p1(cfg, ck["world_seed"], run, updates=ck["state"]["update"] + 50, log_every=25)   # the 2nd log: steady state
    log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()][-1]
    terms = [json.loads(l) for l in TERMS[0].read_text().splitlines()]
    print("drift rows", json.dumps({k: log.get(k) for k in ("update", "sec_per_update", "peak_gib", "g1")}),
          "calls", len(terms), "drift 4/8, 6/8, flow:", *(round(float(np.mean([t[k] for t in terms])), 6) for k in ("drift_4", "drift_6", "flow")),
          flush=True)


def train():
    """The continuation's train() with drift rows installed and its run directory moved here; drift terms logged."""
    ck = torch.load(SRC / "ckpt_10.pt", map_location="cpu", weights_only=False)
    install(ck["cfg"])
    C14.RUN = RUN
    C14.train()


if __name__ == "__main__":
    {"check": check, "train": train}[sys.argv[1]]()
