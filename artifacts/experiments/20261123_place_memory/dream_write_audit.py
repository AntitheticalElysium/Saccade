"""Count unsupported overwrites of real place memories in saved model rollouts."""

import json
from pathlib import Path

import numpy as np

from .offscreen_routes import EXPERIMENTS, RR, CC, S0, SIDE, WALKABLE

HERE = Path(__file__).resolve().parent
RUN = EXPERIMENTS / "20261119_s2"
STABLE = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
EDITS = (5, 7, 8, 9, 10)
DIRECTIONS = np.array([(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)])


def locations(pos):
    y, x = pos[..., 0, None] + RR.reshape(-1) - 3, pos[..., 1, None] + CC.reshape(-1) - 4
    valid = (y >= 0) & (y < SIDE) & (x >= 0) & (x < SIDE)
    return np.where(valid, y * SIDE + x, 0), valid


def remembered(rows):
    with np.load(S0 / "s0_dev_labels.npz") as z:
        offsets = np.r_[0, np.cumsum(z["ep_T"] + 1)]
        positions, views = z["s_pos"].astype(int), z["s_view"]
    maps = np.zeros((len(rows), SIDE**2), dtype=np.int8)
    origins = []
    for i, row in enumerate(rows):
        episode, t = int(row[0]), int(row[3])
        sl = slice(offsets[episode] + max(0, t - 64), offsets[episode] + t + 1)
        keys, use = locations(positions[sl])
        for k, m, view in zip(keys, use, views[sl].reshape(-1, 63)):
            maps[i, k[m]] = view[m]
        origins.append(positions[offsets[episode] + t])
    return maps, np.stack(origins)


def constrain(pos, actions):
    step = np.diff(pos.astype(int), axis=-2)
    effect = DIRECTIONS[np.where(actions <= 4, actions, 0)]
    accept = (step == effect).all(-1, keepdims=True)
    return np.concatenate((pos[:, :1], pos[:, :1] + np.cumsum(np.where(accept, step, 0), axis=-2)), axis=1)


def score(anchors, origin, plans, pose, terrain, done):
    draws = terrain.shape[1]
    n = len(anchors) * draws
    h = terrain.shape[2] - 1
    plans = np.repeat(plans[:, :h], draws, axis=0)
    pose = pose.reshape(n, h + 1, 2)
    np.testing.assert_array_equal(pose[:, 0], np.repeat(origin, draws, axis=0))
    pos = constrain(pose, plans)
    keys, valid = locations(pos)
    read = terrain.reshape(n, h + 1, 63)
    before_edit = ~np.maximum.accumulate(np.isin(plans, EDITS), axis=1)
    alive = np.c_[np.ones(n, bool), ~np.maximum.accumulate(done.reshape(n, h)[:, :-1], axis=1)]
    maps = np.repeat(anchors, draws, axis=0)
    touched = np.zeros_like(maps, dtype=bool)
    disagreement = np.zeros_like(touched)
    row = np.arange(n)[:, None]
    counts = dict(exposed=0, changed=0, passability_change=0, reentries=0, changed_reentries=0,
                  water_exposed=0, water_changed=0)
    for t in range(1, h + 1):
        k = keys[:, t]
        anc = maps[row, k]
        p = read[:, t]
        eligible = (before_edit[:, t - 1] & alive[:, t - 1])[:, None]
        seen = eligible & valid[:, t] & (p > 1) & np.isin(anc, STABLE)
        diff = seen & (anc != p)
        dy, dx = (pos[:, t] - pos[:, t - 1]).T
        oy, ox = RR.reshape(1, -1) + dy[:, None], CC.reshape(1, -1) + dx[:, None]
        entering = (oy < 0) | (oy > 6) | (ox < 0) | (ox > 8)
        counts["exposed"] += int(seen.sum())
        counts["changed"] += int(diff.sum())
        counts["passability_change"] += int((diff & (np.isin(anc, WALKABLE) != np.isin(p, WALKABLE))).sum())
        counts["reentries"] += int((seen & entering).sum())
        counts["changed_reentries"] += int((diff & entering).sum())
        counts["water_exposed"] += int((seen & (anc == 3)).sum())
        counts["water_changed"] += int((diff & (anc == 3)).sum())
        i, j = np.nonzero(seen)
        touched[i, k[i, j]] = True
        disagreement[i, k[i, j]] = diff[i, j]
    counts["last_write_disagrees"] = int(disagreement.sum())
    counts["unique_slots_touched"] = int(touched.sum())
    return counts


def main():
    with np.load(RUN / "bank_competent.npz") as b:
        select = b["kind"] == 0
        rows, plans = b["rows"][select], b["plans"][select]
    anchors, origin = remembered(rows)
    result = {"starts": len(rows), "draws_per_start": 8}
    for arm, file, key, draws in (("real_render", "oracle_competent.npz", "rd", slice(8, 16)),
                                   ("dream", "model_competent.npz", "m", slice(None))):
        with np.load(RUN / file) as data:
            pos = data[f"{key}_pos"][select, draws]
            terr = data[f"{key}_terr"][select, draws]
            done = data["lab_done" if key == "rd" else "m_done"][select, draws]
        result[arm] = score(anchors, origin, plans, pos, terr, done)
    (HERE / "dream_write_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
