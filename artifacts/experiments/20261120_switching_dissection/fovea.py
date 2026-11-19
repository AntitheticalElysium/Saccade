"""Stage A4 of the switching dissection (PLAN.md): step-1 deletion of start mobs inside versus outside the last real
frame's glimpse window (a seeded uniform fixation, rebuilt exactly as S.encode drew it), and, observationally, the
next-step hazard of tracked mobs inside versus outside each generated frame's chosen window; by day and night.

    python -m artifacts.experiments.20261120_switching_dissection.fovea
"""
import json

import numpy as np

from saccade import data as D
from saccade.eval import oracle as O

EV = __import__("importlib").import_module("artifacts.experiments.20261120_switching_dissection.events")
S2 = EV.S2
SEED = 0                                                                  # cfg p1.monitor.fixed_fixation_seed


def real_fixations():
    """The fixation of each S2 start's last real frame (state 0), as S.encode drew it."""
    import importlib
    S = importlib.import_module("saccade.eval.s1")
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    b = np.load(EV.BANK)
    rows = b["rows"][b["sample"]]
    fx = {}
    for e in np.unique(rows[:, 0]):
        rng = np.random.default_rng([SEED, int(L["ep_shard"][e]), int(L["ep_slot"][e])])
        fx[int(e)] = np.array([D.fixation_prior(rng, ()) for _ in range(int(L["ep_T"][e]) + 1)])
    return np.array([fx[int(e)][int(t)] for e, t in zip(rows[:, 0], rows[:, 3])]), rows[:, 0]


def inside(fix, i, j):
    return abs(i - fix[0]) <= 1 and abs(j - fix[1]) <= 1


def main():
    fx0, ep = real_fixations()
    o = np.load(S2 / "oracle.npz")
    night = o["lab_light"][:, 0, 0] < 0.5
    rng = np.random.default_rng(20261120)
    res = {}
    for world in ("P1.5", "sw_rec2_4000"):
        z = np.load(S2 / ("model.npz" if world == "P1.5" else f"model_{world}.npz"))
        grid, pos, done, mfix = z["m_grid"], z["m_pos"].astype(int), z["m_done"], z["m_fix"]
        N, F = grid.shape[:2]
        # step 1: [kind, glimpsed, night, ring] -> per-start (deaths, at risk)
        s1 = {}
        deep = {}
        for n in range(N):
            for f in range(F):
                g, p, dn = grid[n, f], pos[n, f], done[n, f]
                na = O.n_alive({"done": dn, "pos": p})
                for k, kn in enumerate(EV.KINDS, 1):
                    if kn == "arrows":
                        continue
                    for i, j in zip(*np.nonzero(g[0] == k)):
                        if not (1 <= i <= 5 and 1 <= j <= 7):
                            continue
                        w = p[0] + np.array([i - 3, j - 4])
                        ring = i in (1, 5) or j in (1, 7)
                        for s in range(1, min(16, na - 1) + 1):            # greedy 1-cell tracker, path kept
                            rel = w - p[s]
                            if abs(rel[0]) > 2 or abs(rel[1]) > 3:
                                break
                            r0, c0 = rel[0] + 3, rel[1] + 4
                            prev = w - p[s - 1]; pi_, pj = prev[0] + 3, prev[1] + 4    # its cell at state s - 1
                            glimpsed = inside(fx0[n], pi_, pj) if s == 1 else inside(mfix[n, f, s - 2], pi_, pj)
                            win = g[s][r0 - 1:r0 + 2, c0 - 1:c0 + 2] == k
                            key = (kn, bool(glimpsed), bool(night[n]))
                            tab = s1 if s == 1 else deep
                            kk = key + ((bool(ring),) if s == 1 else ())
                            tab.setdefault(kk, (np.zeros(N), np.zeros(N)))
                            tab[kk][1][n] += 1
                            if not win.any():
                                tab[kk][0][n] += 1
                                break
                            ii, jj = np.nonzero(win)
                            bb = np.argmin(np.abs(ii - 1) + np.abs(jj - 1))
                            w = p[s] + np.array([r0 - 1 + ii[bb] - 3, c0 - 1 + jj[bb] - 4])
        out = {}
        print(f"== {world}: step 1 (last real frame's seeded glimpse); hazard (SE), at risk")
        for kn in ("zombies", "cows", "skeletons"):
            for nt in (False, True):
                for rg in (False, True):
                    a = [s1.get((kn, gl, nt, rg)) for gl in (True, False)]
                    if any(x is None for x in a):
                        continue
                    (h1, se1, _, _), (h0, se0, _, _) = (EV.ratio_ci(x[0], x[1], ep, rng) for x in a)
                    out[f"step1|{kn}|{'night' if nt else 'day'}|{'ring' if rg else 'inner'}"] = {
                        "glimpsed": [h1, se1, float(a[0][1].sum())], "not": [h0, se0, float(a[1][1].sum())]}
                    print(f"  {kn:9s} {'night' if nt else 'day':5s} {'ring ' if rg else 'inner'} glimpsed {h1:.3f} ({se1:.3f}) n={int(a[0][1].sum()):5d}"
                          f" | not {h0:.3f} ({se0:.3f}) n={int(a[1][1].sum()):5d} | diff {h1 - h0:+.3f}", flush=True)
        print(f"== {world}: steps 2-16 (chooser's fixation of the previous generated frame; observational)")
        for kn in ("zombies", "cows", "skeletons"):
            for nt in (False, True):
                a = [deep.get((kn, gl, nt)) for gl in (True, False)]
                if any(x is None for x in a):
                    continue
                (h1, se1, _, _), (h0, se0, _, _) = (EV.ratio_ci(x[0], x[1], ep, rng) for x in a)
                out[f"deep|{kn}|{'night' if nt else 'day'}"] = {"glimpsed": [h1, se1, float(a[0][1].sum())], "not": [h0, se0, float(a[1][1].sum())]}
                print(f"  {kn:9s} {'night' if nt else 'day':5s} glimpsed {h1:.3f} ({se1:.3f}) n={int(a[0][1].sum()):6d}"
                      f" | not {h0:.3f} ({se0:.3f}) n={int(a[1][1].sum()):6d} | diff {h1 - h0:+.3f}", flush=True)
        res[world] = out
    (EV.HERE / "fovea.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
