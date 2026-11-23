"""Does P1.5 remember places beyond the frame? PLAN.md (2026-11-23).

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_place_memory.place_memory
"""
import argparse
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from saccade.eval import gates as G, s1 as S

HERE = Path(__file__).resolve().parent
B = importlib.import_module("artifacts.experiments.20261119_s2.s2b")
S2 = B.S2
RR, CC = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
BUCKETS = ((2, 16), (17, 64), (65, 256), (257, 10 ** 9))


def real_terrain(R, en):
    with torch.no_grad():
        p, v = en["p"].to(S2.DEV).float()[None], en["v"].to(S2.DEV).float()[None]
        return B.terrain(R, p, v)[0]                                        # [T + 1, 7, 9]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    B.H = 1
    model, cfg, R, head, pol = B.setup()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    eps = np.flatnonzero(L["ep_eps"] < 0.99)[:8 if args.smoke else 120]
    rng = np.random.default_rng(20261123)
    rows, plans = [], []
    for e in eps:
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(int)
        ts = [t for t in range(8, T) if (pos[t + 1] != pos[t]).any()]
        for t in sorted(rng.choice(ts, min(len(ts), 6 if args.smoke else 40), replace=False)) if ts else []:
            rows.append((int(e), int(L["ep_shard"][e]), int(L["ep_slot"][e]), int(t), T, 1))
            plans.append([int(L["t_action"][so[e] - e + t])])
    rows, plans = np.array(rows, np.int32), np.array(plans, np.int64)
    sub = {e: i for i, e in enumerate(eps)}
    enc = S.encode(model, S2.sublabels(L, eps), cfg)
    grids = G.episode_grids(L)                                              # indexed by the labels' episode index
    out = B.rollout(model, cfg, R, head, pol, enc, sub, grids, rows, plans, mode="free", mf=4, seed=7, gaze="recorded")
    dream = {w: out["m_terr"][w][:, 1] for w in range(len(rows))}          # [4, 7, 9] per start
    rterr = {e: real_terrain(R, enc[sub[e]]) for e in eps}
    rec = {k: [] for k in ("episode", "group", "context", "cls", "agree")}
    by_ep = {}
    for w, r in enumerate(rows):
        by_ep.setdefault(int(r[0]), []).append((int(r[3]), w))
    for e, starts in by_ep.items():                                         # one sweep per episode
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(int)
        real_e, last, k = rterr[e], {}, 0
        starts = sorted(starts)
        for s in range(T + 1):
            for i in range(7):
                for j in range(9):
                    last[(pos[s][0] + RR[i, j], pos[s][1] + CC[i, j])] = s
            while k < len(starts) and starts[k][0] == s:
                t, w = starts[k]; k += 1
                now = {(pos[t][0] + RR[i, j], pos[t][1] + CC[i, j]) for i in range(7) for j in range(9)}
                d = pos[t + 1] - pos[t]
                real = real_e[t + 1]
                for i in range(7):
                    for j in range(9):
                        cell = (pos[t + 1][0] + RR[i, j], pos[t + 1][1] + CC[i, j])
                        if cell in now or real[i, j] < 0:
                            continue
                        ii, jj = i - d[0], j - d[1]
                        if not (0 <= ii < 7 and 0 <= jj < 9) or real[ii, jj] < 0:
                            continue
                        gap = (t + 1 - last[cell]) if cell in last else None
                        group = "first" if gap is None else next(f"{lo}-{hi}" for lo, hi in BUCKETS if lo <= gap <= hi)
                        rec["episode"].append(int(e)); rec["group"].append(group)
                        rec["context"].append("predictable" if real[i, j] == real[ii, jj] else "surprising")
                        rec["cls"].append(int(real[i, j])); rec["agree"].append(float((dream[w][:, i, j] == real[i, j]).mean()))
    z = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(HERE / ("cells_smoke.npz" if args.smoke else "cells.npz"), **z)
    res = summarise(z, rng)
    (HERE / ("place_memory_smoke.json" if args.smoke else "place_memory.json")).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


def matched(z, idx, ctx, group):
    """Class-matched revisit minus first-sighting agreement in a context, weighted by the revisit's class counts."""
    m = idx[(z["context"][idx] == ctx)]
    a, b = m[z["group"][m] == group], m[z["group"][m] == "first"]
    num = den = 0.0
    for c in np.unique(z["cls"][a]):
        ac, bc = a[z["cls"][a] == c], b[z["cls"][b] == c]
        if len(ac) and len(bc):
            num += len(ac) * (z["agree"][ac].mean() - z["agree"][bc].mean()); den += len(ac)
    return num / den if den else None


def summarise(z, rng, reps=1000):
    eps = np.unique(z["episode"]); idx = {e: np.flatnonzero(z["episode"] == e) for e in eps}
    allidx = np.arange(len(z["agree"]))
    groups = ["first"] + [f"{lo}-{hi}" for lo, hi in BUCKETS]
    boots = [np.concatenate([idx[e] for e in rng.choice(eps, len(eps))]) for _ in range(reps)]
    res = {"cells": int(len(allidx)), "episodes": int(len(eps)), "agreement": {}, "matched_minus_first": {}}
    for ctx in ("surprising", "predictable"):
        for g in groups:
            m = allidx[(z["context"] == ctx) & (z["group"] == g)]
            vals = [z["agree"][b[(z["context"][b] == ctx) & (z["group"][b] == g)]].mean() for b in boots]
            res["agreement"][f"{ctx}/{g}"] = {"n": int(len(m)), "mean": float(z["agree"][m].mean()) if len(m) else None,
                                              "ci95": np.nanquantile(vals, [.025, .975]).tolist() if len(m) else None}
        for g in groups[1:]:
            point = matched(z, allidx, ctx, g)
            vals = [v for v in (matched(z, b, ctx, g) for b in boots) if v is not None]
            res["matched_minus_first"][f"{ctx}/{g}"] = {"mean": point, "ci95": np.quantile(vals, [.025, .975]).tolist() if vals else None}
    return res


if __name__ == "__main__":
    main()
