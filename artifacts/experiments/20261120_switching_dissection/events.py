"""Stage A of the switching dissection (PLAN.md), on CPU from S2's saved grids, in minutes.
- Cell rates per transition, by depth: edge arrivals; interior appearances split into jumps (a same-kind vanishing
  within 3 cells in the same transition) and spawns; vanishings split the same way.
- Mob survival under a greedy 1-cell tracker: the start's own mobs, and model-born mobs by birth type.
- Episode-cluster bootstrap, paired by start against P1.5.

    python -m artifacts.experiments.20261120_switching_dissection.events
"""
import json, os
from pathlib import Path

import numpy as np

from saccade.eval import gates as G, oracle as O

HERE = Path(__file__).resolve().parent
S2 = HERE.parent / "20261119_s2"
BANK = HERE.parent / "20261114_s2_harness/dev_bank_h16.npz"
KINDS = G.KINDS
BUCKETS = {"1": [0], "2-4": [1, 2, 3], "5-8": [4, 5, 6, 7], "9-16": list(range(8, 16))}   # transition j: state j -> j + 1
EDGE = np.zeros((7, 9), bool); EDGE[[0, 6], :] = True; EDGE[:, [0, 8]] = True
WORLDS = tuple(os.environ["EV_WORLDS"].split(",")) if os.environ.get("EV_WORLDS") else \
    ("simulator", "P1.5", "sw_rec2_2000", "sw_rec2_4000", "sw_rec_2000")             # the K sweep: EV_WORLDS, EV_N
NS = int(os.environ.get("EV_N", 0)) or None                               # smoke: the first EV_N starts
REF = os.environ.get("EV_REF", "P1.5")                                    # the paired reference world


def load(name):
    if name == "simulator":
        o = np.load(S2 / "oracle.npz")
        return o["rd_grid"][:NS, 8:16], o["rd_pos"][:NS, 8:16].astype(int), o["lab_done"][:NS, 8:16]
    z = np.load(S2 / ("model.npz" if name == "P1.5" else f"model_{name}.npz"))
    return z["m_grid"][:NS], z["m_pos"][:NS].astype(int), z["m_done"][:NS]


def dil(m, r):
    for _ in range(r):
        m = G._dilate(m)
    return m


def rates(grid, pos, done):
    """Per start and transition index: cells of each event class per kind -> {kind: {event: (N, 16)}}, valid (N, 16)."""
    N, F = grid.shape[:2]
    valid = ~np.concatenate([np.zeros((N, F, 1), bool), np.cumsum(done, 2) > 0], 2)[:, :, :16]
    out = {k: {e: np.zeros((N, 16)) for e in ("edge", "spawn", "jump", "vanish_paired", "vanish_unpaired", "count")} for k in KINDS}
    births = np.zeros((len(KINDS), N, F, 16, 7, 9), np.int8)               # 1 edge, 2 spawn, 3 jump, at state j + 1
    for j in range(16):
        g0, g1 = grid[:, :, j].reshape(-1, 7, 9), grid[:, :, j + 1].reshape(-1, 7, 9)
        d = np.clip((pos[:, :, j + 1] - pos[:, :, j]).reshape(-1, 2), -1, 1)
        v = valid[:, :, j].reshape(-1)
        ones = np.ones_like(g0, bool)
        inview = G._shift(ones, -d, 1)
        for k, kn in enumerate(KINDS, 1):
            m0, m1 = g0 == k, g1 == k
            new = m1 & ~G._shift(G._dilate(m0), d, 0)
            van = m0 & ~G._shift(G._dilate(m1), -d, 0) & inview
            near_van = G._shift(dil(van, 3), d, 0)                        # a vanishing within 3 cells, in t + 1 cells
            near_new = G._shift(dil(new, 3), -d, 0)                        # an appearance within 3 cells, in t cells
            cells = {"edge": new & EDGE, "spawn": new & ~EDGE & ~near_van, "jump": new & ~EDGE & near_van,
                     "vanish_paired": van & near_new, "vanish_unpaired": van & ~near_new, "count": m1}
            for e, c in cells.items():
                out[kn][e][:, j] = (c.sum((1, 2)) * v).reshape(N, F).sum(1)
            typ = cells["edge"] * 1 + cells["spawn"] * 2 + cells["jump"] * 3
            births[k - 1, :, :, j] = (typ * v[:, None, None]).reshape(N, F, 7, 9)
    return out, valid.sum(1).astype(float), births                         # transitions per start and index


def track(grid, pos, done, k, w, s0, horizon):
    """Greedy 1-cell tracker of one kind-k mob at world cell w seen at state s0 -> (step of death or 0, steps observed)."""
    n = O.n_alive({"done": done, "pos": pos})
    for s in range(s0 + 1, min(s0 + horizon, n - 1) + 1):
        rel = w - pos[s]
        if abs(rel[0]) > 2 or abs(rel[1]) > 3:                              # its neighbourhood leaves the view: censored
            return 0, s - 1 - s0
        r0, c0 = rel[0] + 3, rel[1] + 4
        win = grid[s][r0 - 1:r0 + 2, c0 - 1:c0 + 2] == k
        if not win.any():
            return s - s0, s - s0
        ii, jj = np.nonzero(win)
        b = np.argmin(np.abs(ii - 1) + np.abs(jj - 1))
        w = pos[s] + np.array([r0 - 1 + ii[b] - 3, c0 - 1 + jj[b] - 4])
    return 0, min(s0 + horizon, n - 1) - s0


def survival(grid, pos, done, births):
    """Start mobs: deaths / at risk per step after state 0 (N, 16) per kind. Model-born mobs (born at state >= 1, by
    type, from rates' masks): deaths / at risk at 1..4 steps after birth, per start (N, 4)."""
    N, F = grid.shape[:2]
    start = {kn: (np.zeros((N, 16)), np.zeros((N, 16))) for kn in KINDS}
    born = {(kn, t): (np.zeros((N, 4)), np.zeros((N, 4))) for kn in KINDS for t in ("edge", "spawn", "jump")}
    for n in range(N):
        for f in range(F):
            g, p, dn = grid[n, f], pos[n, f], done[n, f]
            for k, kn in enumerate(KINDS, 1):
                for i, j in zip(*np.nonzero(g[0] == k)):
                    if not (1 <= i <= 5 and 1 <= j <= 7):
                        continue
                    died, seen = track(g, p, dn, k, p[0] + np.array([i - 3, j - 4]), 0, 16)
                    start[kn][1][n, :seen] += 1
                    if died:
                        start[kn][0][n, died - 1] += 1
                for s, i, j in zip(*np.nonzero(births[k - 1, n, f])):
                    t = ("edge", "spawn", "jump")[births[k - 1, n, f, s, i, j] - 1]
                    died, seen = track(g, p, dn, k, p[s + 1] + np.array([i - 3, j - 4]), s + 1, 4)
                    born[(kn, t)][1][n, :seen] += 1
                    if died:
                        born[(kn, t)][0][n, died - 1] += 1
    return start, born


def ratio_ci(num, den, ep, rng, ref=None):
    """sum(num)/sum(den) with an episode bootstrap SE; with ref = (num, den) of P1.5 also the paired difference."""
    est = float(num.sum() / max(den.sum(), 1e-12))
    u, inv = np.unique(ep, return_inverse=True)
    agg = lambda x: np.bincount(inv, weights=x, minlength=len(u))
    a, b = agg(num), agg(den)
    idx = [rng.integers(0, len(u), len(u)) for _ in range(500)]
    se = float(np.std([a[i].sum() / max(b[i].sum(), 1e-12) for i in idx]))
    if ref is None:
        return est, se, None, None
    ra, rb = agg(ref[0]), agg(ref[1])
    diffs = [a[i].sum() / max(b[i].sum(), 1e-12) - ra[i].sum() / max(rb[i].sum(), 1e-12) for i in idx]
    return est, se, est - float(ref[0].sum() / max(ref[1].sum(), 1e-12)), float(np.std(diffs))


def main():
    b = np.load(BANK)
    ep = b["rows"][b["sample"]][:NS, 0]
    rng = np.random.default_rng(20261120)
    R, V, SV = {}, {}, {}
    for w in WORLDS:
        g, p, d = load(w)
        R[w], V[w], births = rates(g, p, d)
        SV[w] = survival(g, p, d, births)
        print("loaded", w, flush=True)
    res = {"rates": {}, "start_hazard": {}, "born_death_within": {}}
    print("cells per transition (x1000), by depth; diff vs P1.5 (SE)")
    for kn in KINDS:
        for e in ("edge", "spawn", "jump", "vanish_paired", "vanish_unpaired"):
            for bk, js in BUCKETS.items():
                row = {}
                for w in WORLDS:
                    num, den = R[w][kn][e][:, js].sum(1), V[w][:, js].sum(1)
                    ref = (R[REF][kn][e][:, js].sum(1), V[REF][:, js].sum(1)) if w != REF else None
                    row[w] = ratio_ci(num, den, ep, rng, ref)
                res["rates"][f"{kn}|{e}|{bk}"] = row
                print(f"  {kn:9s} {e:16s} {bk:5s}", " | ".join(
                    f"{w}: {x[0] * 1e3:6.2f}" + (f" ({x[2] * 1e3:+.2f}±{x[3] * 1e3:.2f})" if x[2] is not None else "")
                    for w, x in row.items()), flush=True)
    print("start mobs: hazard per step (deaths / at risk), by depth")
    for kn in KINDS:
        for bk, js in BUCKETS.items():
            row = {}
            for w in WORLDS:
                dth, rk = SV[w][0][kn]
                ref = (SV[REF][0][kn][0][:, js].sum(1), SV[REF][0][kn][1][:, js].sum(1)) if w != REF else None
                row[w] = ratio_ci(dth[:, js].sum(1), rk[:, js].sum(1), ep, rng, ref) + (float(rk[:, js].sum()),)
            res["start_hazard"][f"{kn}|{bk}"] = row
            print(f"  {kn:9s} {bk:5s}", " | ".join(
                f"{w}: {x[0]:.3f}" + (f" ({x[2]:+.3f}±{x[3]:.3f})" if x[2] is not None else "") + f" n={int(x[4])}"
                for w, x in row.items()), flush=True)
    print("model-born mobs: death within 1 / 2 / 4 steps of birth (product of per-step hazards), births at risk")
    for kn in KINDS:
        for t in ("edge", "spawn", "jump"):
            row = {}
            for w in WORLDS:
                dth, rk = SV[w][1][(kn, t)]
                h = dth.sum(0) / np.maximum(rk.sum(0), 1)
                cum = 1 - np.cumprod(1 - h)
                row[w] = [float(cum[0]), float(cum[1]), float(cum[3]), float(rk[:, 0].sum())]
            res["born_death_within"][f"{kn}|{t}"] = row
            print(f"  {kn:9s} {t:6s}", " | ".join(f"{w}: {x[0]:.2f}/{x[1]:.2f}/{x[2]:.2f} n={int(x[3])}" for w, x in row.items()), flush=True)
    (HERE / ("events.json" if NS is None else f"events_{os.environ.get('EV_TAG', 'smoke')}.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
