"""Stage A2 of the switching dissection (PLAN.md): after a tracked mob dies (no same-kind mob within 1 cell of its
last position, still in view), is it redrawn within 1 cell of that position at the next one or two steps? From S2's
saved grids, every mob present at state 0 (inner cells), per world; episode bootstrap paired against P1.5.

    python -m artifacts.experiments.20261120_switching_dissection.resurrect
"""
import json

import numpy as np

from saccade.eval import oracle as O

EV = __import__("importlib").import_module("artifacts.experiments.20261120_switching_dissection.events")


def near(grid_s, pos_s, w, k):
    """Same-kind mob within 1 cell of world cell w in the frame at pos_s (None if w's neighbourhood is out of view)."""
    rel = w - pos_s
    if abs(rel[0]) > 2 or abs(rel[1]) > 3:
        return None
    r0, c0 = rel[0] + 3, rel[1] + 4
    return bool((grid_s[r0 - 1:r0 + 2, c0 - 1:c0 + 2] == k).any())


def deaths(grid, pos, done):
    """Per start: (deaths at step 1, recovered of those, deaths at steps 2-14, recovered of those) per kind."""
    N, F = grid.shape[:2]
    out = {kn: np.zeros((N, 4)) for kn in EV.KINDS}
    for n in range(N):
        for f in range(F):
            g, p, dn = grid[n, f], pos[n, f], done[n, f]
            na = O.n_alive({"done": dn, "pos": p})
            for k, kn in enumerate(EV.KINDS, 1):
                for i, j in zip(*np.nonzero(g[0] == k)):
                    if not (1 <= i <= 5 and 1 <= j <= 7):
                        continue
                    died, _ = EV.track(g, p, dn, k, p[0] + np.array([i - 3, j - 4]), 0, 16)
                    if not died or died + 2 > na - 1:
                        continue
                    w = _last(g, p, k, p[0] + np.array([i - 3, j - 4]), died)
                    back = [near(g[s], p[s], w, k) for s in (died + 1, died + 2)]
                    if back[0] is None:
                        continue
                    col = 0 if died == 1 else 2
                    if died > 14:
                        continue
                    out[kn][n, col] += 1
                    out[kn][n, col + 1] += any(b for b in back if b is not None)
    return out


def _last(g, p, k, w, died):
    """The tracker's last position before the death step (replays EV.track's greedy updates)."""
    for s in range(1, died):
        rel = w - p[s]; r0, c0 = rel[0] + 3, rel[1] + 4
        win = g[s][r0 - 1:r0 + 2, c0 - 1:c0 + 2] == k
        ii, jj = np.nonzero(win)
        b = np.argmin(np.abs(ii - 1) + np.abs(jj - 1))
        w = p[s] + np.array([r0 - 1 + ii[b] - 3, c0 - 1 + jj[b] - 4])
    return w


def main():
    b = np.load(EV.BANK)
    ep = b["rows"][b["sample"]][:EV.NS, 0]
    rng = np.random.default_rng(20261120)
    D = {}
    for w in EV.WORLDS:
        D[w] = deaths(*EV.load(w))
        print("loaded", w, flush=True)
    res = {}
    print("deleted start mobs redrawn within 1 cell at the next 1-2 steps: rate (diff vs P1.5 +- SE), deaths")
    for kn in EV.KINDS:
        for lab, col in (("died at step 1", 0), ("died at steps 2-14", 2)):
            row = {}
            for w in EV.WORLDS:
                ref = (D["P1.5"][kn][:, col + 1], D["P1.5"][kn][:, col]) if w != "P1.5" else None
                row[w] = EV.ratio_ci(D[w][kn][:, col + 1], D[w][kn][:, col], ep, rng, ref) + (float(D[w][kn][:, col].sum()),)
            res[f"{kn}|{lab}"] = row
            print(f"  {kn:9s} {lab:18s}", " | ".join(
                f"{w}: {x[0]:.3f}" + (f" ({x[2]:+.3f}+-{x[3]:.3f})" if x[2] is not None else "") + f" n={int(x[4])}"
                for w, x in row.items()), flush=True)
    (EV.HERE / ("resurrect.json" if EV.NS is None else "resurrect_smoke.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
