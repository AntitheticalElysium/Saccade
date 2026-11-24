"""Remove single remembered cells from the known-route water diagnostic."""

import json
from collections import deque
from pathlib import Path

import numpy as np

from .offscreen_routes import BUDGET, CC, DELTAS, RR, S0, SIDE, WALKABLE, WATER, route

HERE = Path(__file__).resolve().parent


def shortest_path(terrain, origin):
    start = tuple(map(int, origin))
    prev = {start: None}
    q = deque([(start, 0)])
    while q:
        pos, d = q.popleft()
        y, x = pos
        for dy, dx in DELTAS:
            end = y + dy, x + dx
            r, c = end
            if r < 0 or r >= SIDE or c < 0 or c >= SIDE:
                continue
            if terrain[end] == WATER:
                path = [pos]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return end, path[::-1]
            if d < BUDGET and terrain[end] in WALKABLE and end not in prev:
                prev[end] = pos
                q.append((end, d + 1))
    raise AssertionError("archived route is missing")


def compare(terrain, origin, visible):
    water, path = shortest_path(terrain, origin)
    remembered_path = [p for p in path if not visible[p]]
    if not remembered_path:
        return False, None, 0
    w = terrain.copy()
    w[water] = 0
    water_pivotal = route(w, origin) > BUDGET
    pivotal = None
    for pos in remembered_path:
        v = terrain.copy()
        v[pos] = 0
        if route(v, origin) > BUDGET:
            pivotal = int(terrain[pos])
            break
    return water_pivotal, pivotal, len(remembered_path)


def main():
    rows = json.loads((HERE / "offscreen_routes.json").read_text())["rows"]
    chosen = {}
    for r in rows:
        if r["all_coarse"] <= BUDGET:
            chosen.setdefault(r["episode"], {})[r["t"]] = r["all_coarse"]
    with np.load(S0 / "s0_dev_labels.npz") as l:
        lengths, pos, view = l["ep_T"], l["s_pos"].astype(int), l["s_view"]
    offsets = np.r_[0, np.cumsum(lengths + 1)]
    output = []
    for e, times in chosen.items():
        seen = np.zeros((SIDE, SIDE), dtype=np.int8)
        for t in range(max(times) + 1):
            g = offsets[e] + t
            origin = pos[g]
            y, x = origin[0] + RR - 3, origin[1] + CC - 4
            m = (y >= 0) & (y < SIDE) & (x >= 0) & (x < SIDE)
            seen[y[m], x[m]] = view[g][m]
            if t in times:
                assert route(seen, origin) == times[t]
                visible = np.zeros_like(seen, dtype=bool)
                visible[y[m], x[m]] = True
                water, cell, length = compare(seen, origin, visible)
                output.append({"episode": int(e), "t": int(t), "water_pivotal": water,
                               "first_pivotal_path_class": cell, "path_cells_offscreen": length})
    test = [r for r in output if r["path_cells_offscreen"] > 0]
    both = [r for r in test if r["first_pivotal_path_class"] is not None and not r["water_pivotal"]]
    result = {"qualifying_routes": len(output), "routes_with_remembered_walkable_cell": len(test),
              "at_least_one_pivotal_remembered_walkable_cell": sum(r["first_pivotal_path_class"] is not None for r in test),
              "pivotal_walkable_cell_but_chosen_water_cell_redundant": len(both),
              "examples_by_first_pivotal_class": {i: sum(r["first_pivotal_path_class"] == i for r in both) for i in WALKABLE},
              "rows": output}
    (HERE / "path_entry_value.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
