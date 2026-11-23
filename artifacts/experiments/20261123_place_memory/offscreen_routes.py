"""Test whether all-coarse or gaze-gated place writes support off-screen water routes."""

import json
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
S0 = EXPERIMENTS.parent / "eda/s0"
SEED = 20261123
FIRST = 8
BUDGET = 32
REPLICATES = 4
SIDE = 64
WATER = 3
WALKABLE = (2, 6, 7, 13)
RR, CC = np.indices((7, 9))
DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
FIX = [(r, c) for r in range(1, 8) for c in range(1, 8)]


def route(terrain, pos, water=None):
    """Shortest labelled static route to an adjacent square; unknown terrain blocks search."""
    water = terrain == WATER if water is None else water
    if not water.any():
        return BUDGET + 1
    goals = np.zeros_like(water)
    goals[:, 1:] |= water[:, :-1]
    goals[:, :-1] |= water[:, 1:]
    goals[1:] |= water[:-1]
    goals[:-1] |= water[1:]
    r, c = map(int, pos)
    passable = np.isin(terrain, WALKABLE)
    to_visit = deque([(r, c, 0)])
    passable[r, c] = False
    while to_visit:
        y, x, dist = to_visit.popleft()
        if goals[y, x]:
            return dist
        if dist == BUDGET:
            continue
        for dy, dx in DELTAS:
            i, j = y + dy, x + dx
            if 0 <= i < SIDE and 0 <= j < SIDE and passable[i, j]:
                passable[i, j] = False
                to_visit.append((i, j, dist + 1))
    return BUDGET + 1


def study(labels, selected, probs):
    offsets = np.r_[0, np.cumsum(labels["ep_T"] + 1)]
    rng = np.random.default_rng(SEED)
    results = []
    for i, ep in enumerate(selected):
        n = int(labels["ep_T"][ep])
        sl = slice(offsets[ep], offsets[ep + 1])
        p = probs["p"][probs["episode"] == i]
        assert len(p) == max(0, n + 1 - FIRST)
        assert np.array_equal(probs["night"][probs["episode"] == i], labels["s_light"][sl][FIRST:] < .5)
        if n < FIRST or labels["ep_eps"][ep] >= .99:
            continue
        p = np.concatenate((np.full((FIRST, 49), 1 / 49), p), axis=0)
        p /= p.sum(-1, keepdims=True)
        actual = labels["s_view"][sl]
        pos = labels["s_pos"][sl].astype(int)
        target = (labels["s_body"][sl, 2] <= 3) & ~labels["s_sleep"][sl].astype(bool)
        land = np.zeros((SIDE, SIDE), np.int8)
        gaze_maps = np.zeros((2, REPLICATES, SIDE, SIDE), np.int8)
        rands = rng.random((2, REPLICATES, n + 1, 1))
        old_eye = np.minimum((rands[0] > np.cumsum(p, axis=-1)).sum(-1), 48)
        uniform = np.minimum((49 * rands[1, ..., 0]).astype(int), 48)
        for t in range(n):
            rr, cc = pos[t, 0] + RR - 3, pos[t, 1] + CC - 4
            show = (rr >= 0) & (rr < SIDE) & (cc >= 0) & (cc < SIDE)
            seen = actual[t]
            land[rr[show], cc[show]] = seen[show]
            for g, windows in enumerate((old_eye, uniform)):
                for draw in range(REPLICATES):
                    rf, cf = FIX[windows[draw, t]]
                    eye = show & (np.abs(RR - rf) <= 1) & (np.abs(CC - cf) <= 1)
                    gaze_maps[g, draw, rr[eye], cc[eye]] = seen[eye]

            if t < FIRST or t % 4 or not target[t] or (seen == WATER).any():
                continue
            row = {
                "episode": int(ep), "t": t, "all_coarse": route(land, pos[t]),
                "old_eye_only": [], "uniform_eye_only": [],
                "old_eye_targets_full_path": [], "uniform_eye_targets_full_path": [],
            }
            for g, name in enumerate(("old_eye", "uniform_eye")):
                for draw in range(REPLICATES):
                    trace = gaze_maps[g, draw].copy()
                    trace[rr[show], cc[show]] = seen[show]
                    row[f"{name}_only"].append(route(trace, pos[t]))
                    # Match route coverage to isolate which water targets were remembered.
                    row[f"{name}_targets_full_path"].append(route(land, pos[t], trace == WATER))
            results.append(row)
    return results


def sources():
    template = EXPERIMENTS / "20261119_s2/dream_outcome_audit.json"
    selected = json.loads(template.read_text())["recorded_frames"]["dev_episodes"]
    with np.load(S0 / "s0_dev_labels.npz") as data:
        labels = {k: data[k] for k in ("ep_T", "ep_eps", "s_pos", "s_light", "s_view", "s_body", "s_sleep")}
    with np.load(EXPERIMENTS / "20261123_eye_drives/drives.npz") as data:
        probs = {k: data[k] for k in ("episode", "p", "night")}
    return labels, selected, probs


def summarize(rows):
    groups = ("old_eye_only", "uniform_eye_only", "old_eye_targets_full_path", "uniform_eye_targets_full_path")
    full = np.array([r["all_coarse"] for r in rows])
    episodes = np.array([r["episode"] for r in rows])
    rng = np.random.default_rng(SEED)
    results = {}
    for limit in (16, 32):
        transferable = full <= limit
        out = {"all_coarse_reachable": int(transferable.sum()), "all_coarse_fraction": float(transferable.mean())}
        ids, inverse = np.unique(episodes[transferable], return_inverse=True)
        counts = np.bincount(inverse)
        boot = rng.integers(len(ids), size=(4000, len(ids)))
        for name in groups:
            d = np.array([r[name] for r in rows])
            out[f"{name}_reachable_fraction"] = float((d <= limit).mean())
            retained = (d[transferable] <= limit).mean(1)
            weights = np.bincount(inverse, weights=retained)
            fractions = weights[boot].sum(1) / counts[boot].sum(1)
            out[f"{name}_retains_given_full"] = float(retained.mean())
            out[f"{name}_retains_given_full_ci95"] = np.quantile(fractions, (.025, .975)).tolist()
        results[limit] = out
    return {"states": len(rows), "episodes": len({r["episode"] for r in rows}), "by_limit": results}


def main():
    rows = study(*sources())
    result = {"seed": SEED, "replicates": REPLICATES, "summary": summarize(rows), "rows": rows}
    (HERE / "offscreen_routes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
