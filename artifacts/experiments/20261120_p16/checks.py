"""P1.6 pre-launch checks (PLAN.md), GPU:

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.checks units       record groups, routes, adapters
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.checks throughput  run-scale cost and memory
    python -m artifacts.experiments.20261120_p16.checks criteria                           the gate (checks.json passed)
"""
import importlib
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
    passed = all(v for k, v in c.items() if not k.startswith("projected"))
    OUT.update(criteria=c, passed=passed)
    print(json.dumps(c, indent=1), "passed", passed, flush=True)


if __name__ == "__main__":
    {"units": units, "throughput": throughput, "criteria": criteria}[sys.argv[1]]()
    f = HERE / "checks.json"
    prev = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**prev, **OUT}, indent=1) + "\n")
