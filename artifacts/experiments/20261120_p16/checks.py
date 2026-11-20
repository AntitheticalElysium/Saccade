"""P1.6 pre-launch checks (PLAN.md), GPU:

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.checks units       record groups, routes, adapters
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.checks throughput  run-scale cost and memory
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.checks eye         the world-eye: fill, trainer, cost
    python -m artifacts.experiments.20261120_p16.checks criteria                           the gate (checks.json passed)
"""
import copy
import importlib
import time
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
P16 = importlib.import_module("artifacts.experiments.20261120_p16.train")
DEV = torch.device("cuda", 0)
OUT = {}


def rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-12))


def units():
    cfg = P16.recipe()
    torch.manual_seed(1)
    RFT = importlib.import_module("artifacts.experiments.20261116_temporal_memory.record_ft")
    rec = M.Record(cfg).to(DEV)
    B, L = 2, 30
    p, g = torch.randn(B, L, 81, 16, device=DEV), torch.randn(B, L, 9, 64, device=DEV)
    fix = torch.randint(1, 8, (B, L, 2), device=DEV)
    first = torch.zeros(B, L, dtype=torch.bool, device=DEV); first[:, 0] = True; first[1, 11] = True
    a = rec.scan(rec.input(p, g, fix), first, rec.zero(B)); b = RFT.record(p, first, torch.zeros(B, 81, 2, device=DEV))
    d0 = b[0].shape[-1]
    res = {"record_dim": rec.dim, "periphery_group_equals_test2": max(rel(a[0][..., :d0], b[0]), rel(a[1][..., :d0], b[1]))}
    seen = torch.zeros(B, 81, dtype=torch.bool, device=DEV)
    for t in range(L):
        seen[torch.arange(B, device=DEV)[:, None], M.window(fix[:, t])] = True
    res["fovea_zero_where_never_looked"] = bool((a[1][:, -1][~seen][:, d0:] == 0).all()) and bool((a[1][:, -1][seen][:, d0:] != 0).any())
    model = M.M0(cfg).to(DEV)
    model.denoiser.checkpoint = True
    for m in (*model.denoiser.out, model.denoiser.final_mod, model.denoiser.record_read[2]):   # zero-initialised outputs
        torch.nn.init.normal_(m.weight, std=0.02)                       # pass no gradient at update 0 (P1.5's units)
    for blk in model.denoiser.blocks:
        for n, p_ in blk.named_parameters():
            if p_.abs().sum() == 0 and p_.dim() == 2:
                torch.nn.init.normal_(p_, std=0.02)
    split = D.load_split(cfg)
    P = cfg["data"]["context_prefix"]
    streams = D.Streams(split["train"][:40], 4, P, seed=3, targets_offsets=D.frame_offsets(split["train"]))
    tmp = Path(tempfile.mkdtemp())
    ctl = C.Controller(model, cfg, tmp, DEV)
    chunk = streams.next(32)
    ctl.prepare(chunk, streams)
    rows = [1, 3]
    carry, cache = model.belief.initial(4, DEV), torch.zeros(4, P + 1, cfg["belief"]["width"], device=DEV)
    boot = TR.bootstrap_rows(cfg, 1, 0)
    with torch.random.fork_rng(devices=[DEV.index]):
        out = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], None, cfg, boot[rows], 0,
                           None, rec.zero(2), TR.gaze_rows(ctl, rows, 32))
    terms = out[0]
    res["terms_finite"] = all(bool(torch.isfinite(v)) for v in terms.values())
    res["encoder_norms"] = TR.encoder_grad_norms(model, terms)
    blocked = ("reward", "cont", "reward_forecast", "cont_forecast", "gaze_forecast")
    res["route_by_role_ok"] = all(res["encoder_norms"][k] == 0.0 for k in blocked) and all(res["encoder_norms"][k] > 0 for k in ("g1", "tsp", "rec"))
    w_read, w_in = model.denoiser.record_read[0].weight, model.belief.record_in[0].weight
    gr = torch.autograd.grad(terms["g1"], (w_read, w_in), retain_graph=True)
    gw = torch.autograd.grad(terms["reward"], w_in, retain_graph=True)[0]
    res["fovea_columns_grad"] = {"denoiser_read_g1": float(gr[0][:, d0:].norm()), "belief_in_g1": float(gr[1][:, d0:].norm()),
                                 "belief_in_reward": float(gw[:, d0:].norm())}
    res["record_state_shape"] = list(out[3].shape)
    shutil.rmtree(tmp)
    # adapters: s1.encode's record (history with the glimpse), s1.before and s1.window against one contiguous scan
    L_ = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 2)
    model.eval()
    enc = S.encode(model, L_, cfg)
    diffs = []
    for e in range(2):
        x = model.record.input(*(enc[e][k].to(DEV).float() if k != "fix" else enc[e][k].to(DEV) for k in ("p", "g", "fix")))[None]
        T1 = x.shape[1]
        st = torch.zeros(1, T1, dtype=torch.bool, device=DEV); st[:, 0] = True
        rin, _ = model.record.scan(x, st, model.record.zero(1, DEV))
        ts = [t for t in (9, 70, 130) if t < T1]
        w = S.window(model, enc, [(e, t) for t in ts], 8)
        diffs += [rel(w[i, -1], rin[0, t]) for i, t in enumerate(ts)] + [rel(w[i, 0], rin[0, t - 8]) for i, t in enumerate(ts)]
    res["adapter_window_vs_contiguous"] = max(diffs)
    OUT["units"] = res
    print(json.dumps(res, indent=1), flush=True)


def _model(cfg):
    torch.manual_seed(1)
    model = M.M0(cfg).to(DEV)
    for m in (*model.denoiser.out, model.denoiser.final_mod, model.denoiser.record_read[2]):   # zero-initialised outputs
        torch.nn.init.normal_(m.weight, std=0.02)                       # would make every estimate 0
    for blk in model.denoiser.blocks:
        for n, p_ in blk.named_parameters():
            if p_.abs().sum() == 0 and p_.dim() == 2:
                torch.nn.init.normal_(p_, std=0.02)
    return model.eval()


@torch.no_grad()
def eye():
    """P1.6's world-eye. (1) The fill equals forward() on each frame's own window from the same draws, looks, record and
    belief (first frames, segment boundaries, beyond the window). (2) Segmentation does not change it. (3) The trainer
    refreshes at the listed checkpoints, then scores S with the cache, and a resumed controller reopens it (digest
    checked, a changed file refused). (4) The fill's throughput on training episodes, projected to the whole split."""
    cfg = P16.recipe(); res = {}
    model = _model(cfg)
    split = D.load_split(cfg); eps = split["train"][:3]; off = D.frame_offsets(eps)
    tmp = Path(tempfile.mkdtemp()); seed = 7
    looks = np.full(int(off[-1]), -1, np.int16)
    C.world_errors(model, cfg, eps, off, looks, tmp / "a.npy", seed, segment=64)
    C.world_errors(model, cfg, eps, off, looks, tmp / "b.npy", seed, segment=4096)
    A, B = np.load(tmp / "a.npy").astype(np.float32), np.load(tmp / "b.npy").astype(np.float32)
    res["episodes_T"] = [int(e.T) for e in eps]
    res["segmentation_rel"] = float(np.abs(A - B).max() / np.abs(B).max())
    sc, P = cfg["edm"]["shortcut"], cfg["data"]["context_prefix"]
    k, tc = sc["k_max"], sc["context_signal"]
    pc = cfg["perception"]
    dims = ((M.N, pc["periphery"]["token_dim"]), (M.G, pc["fovea"]["token_dim"]), (pc["lighting"]["dim"],))
    rng, worst = np.random.default_rng(seed), []
    for i, ep in enumerate(eps):                                         # the reference, frames < 300 (causal)
        fix = D.fixation_prior(rng, (ep.T + 1,))
        gen = torch.Generator(device=DEV).manual_seed(seed * 1_000_003 + i)
        nc = [torch.randn((1, ep.T + 1, *d), device=DEV, generator=gen) for d in dims]
        nq = [torch.randn((1, ep.T + 1, *d), device=DEV, generator=gen) for d in dims[::2]]
        T1 = min(ep.T + 1, 300)
        c = TR.tensors(D.gather(eps, [[(i, 1, t, *map(int, fix[t])) for t in range(T1)]]), DEV)
        with TR.autocast(DEV, cfg):
            X = model.perception(c["frames"], c["fixation"])
            rin, rst = model.record.scan(model.record.input(X.p, X.g, X.fix), c["first"], model.record.zero(1, DEV))
            y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(X, rst), c["reward"], c["terminated"], c["action"]), model.belief.initial(1, DEV))
            y = y[:, 1::2]; bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], 1)
            for t in [t for t in (0, 1, 5, 8, 9, 63, 64, 65, 127, 128, 200, 299) if t < T1]:
                a = max(0, t - P); sl = lambda z: z[:, a:t + 1]
                xs = tuple(torch.cat([(1 - tc / k) * nz[:, a:t] + tc / k * z[:, a:t].float(), q], 1)
                           for nz, z, q in zip(nc, X.parts(), (nq[0][:, t:t + 1], torch.zeros_like(X.g[:, t:t + 1], dtype=torch.float), nq[1][:, t:t + 1])))
                code = torch.full((1, t + 1 - a), M.shortcut_code(tc, k.bit_length() - 1, k), device=DEV); code[:, -1] = M.shortcut_code(0, k.bit_length() - 1, k)
                o = model.denoiser(xs, code, sl(c["prev_action"]), (model.belief.memory(sl(bp)), sl(rin)), sl(X.fix), sl(c["episode"]), sl(c["valid"]))
                ref = (X.p[0, t].float() - o[0][0, -1].float()).square().sum(-1).cpu().numpy()
                worst.append(float(np.abs(A[off[i] + t] - ref).max() / max(np.abs(ref).max(), 1e-12)))
    res["fill_vs_window_forward_rel"] = max(worst)
    # (3) the trainer: refreshes after checkpoints 1 and 2 of a short run on 40 episodes
    c2 = copy.deepcopy(cfg)
    c2["p1"] = dict(c2["p1"], frames=16 * 64 * 8, checkpoint_every_fraction=0.25)
    c2["fixation"]["world_eye"] = dict(c2["fixation"]["world_eye"], refresh_after=[1, 2])
    run = tmp / "run"
    torch.cuda.reset_peak_memory_stats()
    with torch.enable_grad():
        TR.p1(c2, P16.WORLD_SEED, run, n_episodes=40)
    res["run_peak_gib_with_fills"] = torch.cuda.max_memory_allocated() / 2 ** 30
    log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    res["refreshes_logged"] = [r["world_eye_refresh"] for r in log if "world_eye_refresh" in r]
    res["refresh_seconds_40_episodes"] = [r["seconds"] for r in log if "world_eye_refresh" in r]
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    st = ck["fixation_controller"]; res["latest_eye"] = st["eye"]["file"] if st["eye"] else None
    tr = np.fromfile(run / "gaze.bin", dtype=C.TRACE)
    per = 16 * 64                                                         # rows per 64-step update; refresh 1 after update 2
    match = []
    for name, rows in (("eye_cache_01.npy", tr[2 * per:4 * per]),):
        cache = np.load(run / name).astype(np.float32); ok = rows["target"] >= 0
        S = cache[rows["target"][ok]][:, C.WIN.numpy()].sum(-1)
        match.append(float((S.argmax(-1) == rows["S"][ok]).mean()))
    before = tr[:2 * per]; okb = before["target"] >= 0
    Sb = np.load(run / "eye_cache_01.npy").astype(np.float32)[before["target"][okb]][:, C.WIN.numpy()].sum(-1)
    res["trace_S_is_cache_argmax_after_refresh"] = match[0]
    res["trace_S_is_cache_argmax_before_refresh"] = float((Sb.argmax(-1) == before["S"][okb]).mean())
    ctl = C.Controller(model, c2, run, DEV); ctl.load_state(st)
    res["resumed_controller_has_eye"] = ctl.eye is not None
    data = (run / st["eye"]["file"]).read_bytes(); (run / st["eye"]["file"]).write_bytes(data[:-2] + bytes([data[-2] ^ 1, data[-1]]))
    try:
        C.Controller(model, c2, run, DEV).load_state(st); res["changed_cache_refused"] = False
    except RuntimeError:
        res["changed_cache_refused"] = True
    # (4) throughput on 60 training episodes
    eps60 = split["train"][:60]; off60 = D.frame_offsets(eps60)
    torch.cuda.synchronize(); t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    C.world_errors(model, cfg, eps60, off60, np.full(int(off60[-1]), -1, np.int16), tmp / "c.npy", seed)
    torch.cuda.synchronize(); dt = time.time() - t0
    frames_all = int(D.frame_offsets(split["train"])[-1])
    res["fill"] = {"frames": int(off60[-1]), "seconds": dt, "ms_per_frame": 1000 * dt / int(off60[-1]), "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                   "split_frames": frames_all, "projected_hours_per_fill": dt / int(off60[-1]) * frames_all / 3600}
    shutil.rmtree(tmp)
    OUT["eye"] = res
    print(json.dumps(res, indent=1), flush=True)


def throughput(updates=30):
    """Run-scale cost: the trainer itself for `updates` updates at 64 and 256 steps (route by role on, the recipe);
    seconds per update and peak GiB from the trainer's own log."""
    res = {}
    for steps in (64, 256):
        tmp = Path(tempfile.mkdtemp())
        torch.cuda.reset_peak_memory_stats()
        TR.p1(P16.check_recipe(P16.recipe()), P16.WORLD_SEED, tmp, updates=updates, steps=steps, checkpoints=False)
        log = [json.loads(l) for l in (tmp / "log.jsonl").read_text().splitlines()]
        res[f"steps_{steps}"] = {k: log[-1][k] for k in ("sec_per_update", "peak_gib", "g1")}
        print(json.dumps(res), flush=True)
        shutil.rmtree(tmp)
    OUT["throughput"] = res


def criteria():
    r = json.loads((HERE / "checks.json").read_text())
    u, t = r["units"], r["throughput"]
    p15 = json.loads((HERE.parent / "20261116_p15/checks.json").read_text())["throughput"]
    hours = (t["steps_64"]["sec_per_update"] * 12500 + t["steps_256"]["sec_per_update"] * 3125) / 3600   # P1.5's updates
    hours15 = (p15["route_True_steps_64"]["sec_per_update"] * 12500 + p15["route_True_steps_256"]["sec_per_update"] * 3125) / 3600
    c = {"periphery_group_exact": u["periphery_group_equals_test2"] < 1e-5, "fovea_writes_only_where_looked": u["fovea_zero_where_never_looked"],
         "terms_finite": u["terms_finite"], "route_by_role_ok": u["route_by_role_ok"],
         "fovea_columns_reached": all(v > 0 for v in u["fovea_columns_grad"].values()),
         "adapters_match": u["adapter_window_vs_contiguous"] < 1e-2,
         "peak_under_budget": max(v["peak_gib"] for v in t.values()) < 4.5,
         "projected_hours": hours, "projected_hours_p15_same_estimate": hours15,
         "time_within_17h_estimate": hours * (15.4 / hours15) < 17}           # scaled by P1.5's measured run / its estimate
    if "eye" in r:                                                       # the world-eye (P1.6, 11-20 evening)
        e = r["eye"]
        c.update(eye_fill_matches_window_forward=e["fill_vs_window_forward_rel"] < 3e-2, eye_segmentation_invariant=e["segmentation_rel"] < 3e-2,
                 eye_refreshes_at_listed=e["refreshes_logged"] == [1, 2], eye_scores_with_cache=e["trace_S_is_cache_argmax_after_refresh"] > 0.99,
                 eye_resume_digest=e["resumed_controller_has_eye"] and e["changed_cache_refused"],
                 eye_peak_under_budget=max(e["run_peak_gib_with_fills"], e["fill"]["peak_gib"]) < 5.0,
                 projected_hours_with_fills=hours * (15.4 / hours15) + 2 * e["fill"]["projected_hours_per_fill"])
        c["eye_time_within_19h"] = c["projected_hours_with_fills"] < 19
    passed = all(v for k, v in c.items() if not k.startswith("projected"))
    OUT.update(criteria=c, passed=passed)
    print(json.dumps(c, indent=1), "passed", passed, flush=True)


if __name__ == "__main__":
    {"units": units, "throughput": throughput, "eye": eye, "criteria": criteria}[sys.argv[1]]()
    f = HERE / "checks.json"
    prev = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**prev, **OUT}, indent=1) + "\n")
