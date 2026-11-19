"""Stage B (1), A3 of the switching dissection (PLAN.md), on CPU from S2's saved grids. Step-1 deletion of isolated
start mobs, split four ways:
- by the simulator's own motion of that mob over its 16 label futures;
- by whether the world's camera moved at step 1;
- by ring (next to the view edge, or inner);
- by day or night.

The simulator's half B, read through the reader, is the instrument floor.

    python -m artifacts.experiments.20261120_switching_dissection.deletions [K-sweep tags, e.g. K1_smoke K8_smoke]
"""
import json, sys

import numpy as np

EV = __import__("importlib").import_module("artifacts.experiments.20261120_switching_dissection.events")
S2 = EV.S2


def world(name):
    if name == "simulator_B":
        o = np.load(S2 / "oracle.npz")
        return o["rd_grid"][:, 8:16], o["rd_pos"][:, 8:16].astype(int)
    z = np.load(S2 / ("model.npz" if name == "P1.5" else f"model_{name}.npz"))
    return z["m_grid"], z["m_pos"].astype(int)


def units(o, n_starts):
    """Isolated inner start mobs where reader and labels agree at state 0, with the simulator's label motion at step 1:
    (start, kind, i, j, P(stays in its cell), P(gone: no same kind within 1 cell), P(camera moved), night)."""
    lab, lpos, light = o["lab_grid"][:n_starts], o["lab_pos"][:n_starts].astype(int), o["lab_light"][:n_starts, 0, 0]
    rd0 = o["rd_grid"][:n_starts, 0, 0]
    out = []
    for n in range(n_starts):
        g0 = lab[n, 0, 0]
        for k in range(1, 5):
            for i, j in zip(*np.nonzero(g0 == k)):
                if not (1 <= i <= 5 and 1 <= j <= 7) or rd0[n, i, j] != k:
                    continue
                if ((g0[max(0, i - 2):i + 3, max(0, j - 2):j + 3] == k).sum()) > 1:
                    continue
                w = lpos[n, 0, 0] + np.array([i - 3, j - 4])
                stay = gone = moved_cam = 0
                for f in range(16):
                    rel = w - lpos[n, f, 1]; r, c = rel[0] + 3, rel[1] + 4
                    win = lab[n, f, 1][r - 1:r + 2, c - 1:c + 2] == k
                    stay += lab[n, f, 1][r, c] == k
                    gone += not win.any()
                    moved_cam += bool((lpos[n, f, 1] != lpos[n, f, 0]).any())
                out.append((n, k, i, j, stay / 16, gone / 16, moved_cam / 16, light[n] < 0.5))
    return out


def deleted(grid, pos, u):
    """Per unit: (deleted, camera moved) in each future of this world at step 1, where its neighbourhood stays in view."""
    n, k, i, j = u[:4]
    res = []
    for f in range(grid.shape[1]):
        w = pos[n, f, 0] + np.array([i - 3, j - 4])
        rel = w - pos[n, f, 1]
        if abs(rel[0]) > 2 or abs(rel[1]) > 3:
            continue
        r, c = rel[0] + 3, rel[1] + 4
        res.append((not (grid[n, f, 1][r - 1:r + 2, c - 1:c + 2] == k).any(), bool((pos[n, f, 1] != pos[n, f, 0]).any())))
    return res


def main(extra):
    o = np.load(S2 / "oracle.npz")
    names = ["simulator_B", "P1.5", "sw_rec2_4000", *extra]
    n_starts = min(len(world(x)[0]) for x in names)
    U = units(o, n_starts)
    print(f"{len(U)} isolated inner start mobs over {n_starts} starts", flush=True)
    rows = {}
    for name in names:
        g, p = world(name)
        rows[name] = [deleted(g, p, u) for u in U]
    res = {}

    def table(label, key):
        print(label)
        for kn_i, kn in enumerate(EV.KINDS, 1):
            if kn == "arrows":
                continue
            cells = {}
            for name in names:
                acc = {}
                for u, r in zip(U, rows[name]):
                    if u[1] != kn_i:
                        continue
                    for d, cam in r:
                        b = key(u, cam)
                        if b is None:
                            continue
                        acc.setdefault(b, [0, 0]); acc[b][0] += d; acc[b][1] += 1
                cells[name] = {b: (v[0] / v[1], v[1]) for b, v in sorted(acc.items())}
            res[f"{label}|{kn}"] = cells
            bins = sorted({b for c in cells.values() for b in c})
            for b in bins:
                print(f"  {kn:9s} {b:22s}", " | ".join(f"{nm}: {c[b][0]:.3f} (n={c[b][1]})" if b in c else f"{nm}: -" for nm, c in cells.items()))

    stay_bin = lambda s: "stays 1" if s == 1 else "stays [0.5,1)" if s >= 0.5 else "stays (0,0.5)" if s > 0 else "stays 0"
    table("by the simulator's P(stays in its cell)", lambda u, cam: stay_bin(u[4]) if u[5] == 0 else None)
    table("by this world's camera move at step 1", lambda u, cam: "camera moved" if cam else "camera still")
    table("by ring", lambda u, cam: "edge ring" if u[2] in (1, 5) or u[3] in (1, 7) else "inner")
    table("by light", lambda u, cam: "night" if u[7] else "day")
    table("all", lambda u, cam: "all")
    (EV.HERE / f"deletions{'_' + '_'.join(extra) if extra else ''}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
