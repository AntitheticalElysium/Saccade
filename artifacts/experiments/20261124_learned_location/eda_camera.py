"""Can the agent learn its own camera model from its data? (EDA for the plan's review, 11-24; no model trained.)

1. Geometry from correspondence. For each transition with a sensed move δ (the declared motion sensor: the
   simulator displacement), the mean L1 distance between screen cell k's coarse token at t and every screen cell k'
   at t + 1 (all 81 cells, HUD and the player's cell included). A cell showing the world should match the cell one move
   back (o_k' = o_k - δ); a cell fixed to the sensor (status rows, the player's sprite) should match itself. The
   offset table is then solved from the matches by least squares, up to a constant, and compared with the renderer's.
   By day (light >= 0.5) and in the dark. Panel episodes 0-22 (the diagnosis episodes).
2. Codex's counts (11-24), positions only, all 120 panel episodes: revisit lookups (reading 2's definition) with at
   least 25 earlier sightings of the same world cell under write-every-frame; first sightings whose world address had
   already been given to a status-row token under the plan's offsets (rows 7-8 as rows +4, +5).

    JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261124_learned_location.eda_camera
"""
import importlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")
HERE = Path(__file__).resolve().parent
ROW, COL = torch.arange(81) // 9, torch.arange(81) % 9
OR, OC = ROW - 3, COL - 4                                              # the renderer's offsets (map rows 0-6)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def true_match(k, d):
    """The screen cell showing, after move d, what screen cell k showed before; None if it left view or k is a status
    row (no world projection)."""
    if ROW[k] > 6:
        return None
    r, c = int(OR[k]) - d[0] + 3, int(OC[k]) - d[1] + 4
    return r * 9 + c if 0 <= r <= 6 and 0 <= c <= 8 else None


@torch.no_grad()
def geometry(data):
    S = {(cond, d): torch.zeros(81, 81, dtype=torch.float64) for cond in ("day", "dark") for d in MOVES}
    N = defaultdict(int)
    for r in data:
        p, pos, light = r["p"].float(), r["pos"], r["light"].numpy()
        dl = (pos[1:] - pos[:-1]).tolist()
        for t, d in enumerate(dl):
            d = tuple(d)
            if d not in MOVES:
                continue
            cond = "day" if light[t] >= 0.5 and light[t + 1] >= 0.5 else "dark"
            S[cond, d] += torch.cdist(p[t][None], p[t + 1][None], p=1)[0].double()
            N[cond, d] += 1
    out = {}
    for cond in ("day", "dark"):
        right = wrong = fixed_self = fixed_n = 0
        rel = []
        margins = []
        for d in MOVES:
            D = S[cond, d] / max(N[cond, d], 1)
            for k in range(81):
                best = int(D[k].argmin()); srt = D[k].sort().values
                tm = true_match(k, d)
                if tm is None:
                    if ROW[k] > 6 or k == 31:
                        fixed_n += 1; fixed_self += int(best == k)
                    continue
                if k == 31 or tm == 31:                                   # the player's cell: covered by the sprite
                    continue
                ok = best == tm; right += int(ok); wrong += int(not ok)
                margins.append(float((srt[1] - srt[0]) / srt[0]))
                rel.append((k, best, d))
        # offsets from the matches: o_k - o_best = d, least squares with o_31 = 0, for cells the matches reach
        idx = sorted({k for k, _, _ in rel} | {b for _, b, _ in rel} | {31})
        pos_of = {k: i for i, k in enumerate(idx)}
        A = np.zeros((2 * len(rel) + 2, 2 * len(idx))); y = np.zeros(2 * len(rel) + 2)
        for i, (k, b, d) in enumerate(rel):
            for a in range(2):
                A[2 * i + a, 2 * pos_of[k] + a] = 1; A[2 * i + a, 2 * pos_of[b] + a] = -1; y[2 * i + a] = d[a]
        for a in range(2):
            A[-2 + a, 2 * pos_of[31] + a] = 1
        sol = np.linalg.lstsq(A, y, rcond=None)[0].reshape(-1, 2)
        truth = np.stack([OR[idx].numpy(), OC[idx].numpy()], 1)
        map_cells = [i for i, k in enumerate(idx) if ROW[k] <= 6]
        exact = float((np.round(sol[map_cells]) == truth[map_cells]).all(1).mean())
        out[cond] = {"transitions": {f"{d}": N[cond, d] for d in MOVES},
                     "world_cells_matched_one_move_back": right / max(right + wrong, 1), "pairs": right + wrong,
                     "match_margin_median": float(np.median(margins)) if margins else None,
                     "sensor_fixed_cells_matching_themselves": fixed_self / max(fixed_n, 1),
                     "offset_table_recovered_exactly": exact, "map_cells_in_table": len(map_cells),
                     "offset_error_max": float(np.abs(sol[map_cells] - truth[map_cells]).max())}
    return out


def counts(data):
    dup = n_rev = hud_hit = n_first = 0
    for r in data:
        pos = r["pos"].numpy(); T = r["T"]
        seen = defaultdict(int); hud = set()
        for t in range(T + 1):
            if t > 0:
                prev = {(pos[t - 1][0] + int(OR[k]), pos[t - 1][1] + int(OC[k])) for k in range(63)}
                for k in range(63):
                    w = (pos[t][0] + int(OR[k]), pos[t][1] + int(OC[k]))
                    if w in prev:
                        continue
                    if seen[w]:
                        n_rev += 1; dup += int(seen[w] >= 25)
                    else:
                        n_first += 1; hud_hit += int(w in hud)
            for k in range(81):
                w = (pos[t][0] + int(OR[k]), pos[t][1] + int(OC[k]))
                if ROW[k] <= 6:
                    seen[w] += 1
                else:
                    hud.add(w)
    return {"revisit_lookups": n_rev, "share_with_25_or_more_earlier_sightings": dup / n_rev,
            "first_sightings": n_first, "share_whose_address_already_held_a_status_row_token": hud_hit / n_first}


def main():
    data = torch.load(LM.DATA / "tokens_panel.pt")
    res = {"geometry_episodes_0_22": geometry(data[:23]), "codex_counts_all_120": counts(data)}
    (HERE / "eda_camera.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
