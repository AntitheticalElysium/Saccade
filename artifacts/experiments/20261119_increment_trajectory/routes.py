"""Which input carries the faced-tile edit when the coal digit fails? (follow-up to trajectory.py, 11-19)

trajectory.py: P1.5's first-call coal digit contrast formed at checkpoint 3 (0.15, faced tile already 0.87) and crept
+0.014 per checkpoint after; F4's arrived late and together with the tile (checkpoint 3: digit 0.05 / tile 0.13;
checkpoint 5: 0.85 / 0.82), through the direct action (Codex's crossed routes, checkpoint 5: direct alone moves the tile
0.59 and the digit 0.27, the belief alone 0.25 / 0.11). HUD-Change learned the tile early through the post-action belief
(checkpoint 3: belief alone 0.40, direct alone 0.24) and never the coal digit (both inputs 0.02).

Hypothesis (route competition): when the belief carries the map edit early, the denoiser never builds its own
action-and-faced-block route, and the rare coal digit, which F4 drew through that route, has no route left. Competing
explanation: the direct route forms anyway and the digit fails for another reason (the record's copy, Codex; run-level
chance in an abrupt transition).

Codex's localize.py functions unchanged, its bank, seeds, 16 histories x 4 paired draws per cohort, first denoiser call,
all four conditions (direct action x post-action belief, NOOP/DO each): P1.5 at checkpoints 3, 5, 10, F4 at checkpoint 4
(between its 3 and 5). Writes routes_{world}_{KK}.json.

Read, fixed now:
- Supports: P1.5's coal tile at checkpoint 3 moves more with the belief alone than with the direct action alone (as
  HUD-Change); F4 at checkpoint 4 moves its coal tile and digit together, the direct action alone at least as much as the
  belief alone.
- Refutes: P1.5's coal tile at checkpoint 3 is carried by the direct action alone (>= 0.5 and >= the belief alone) while
  its coal digit stays <= 0.2 under both inputs: the direct route formed without carrying the digit.
- Neither input alone moving the tile (both < 0.3, both inputs together > 0.8) = a conjunction: inconclusive on the route.

    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python -m artifacts.experiments.20261119_increment_trajectory.routes WORLD:STEP [...]
"""
import importlib, json, sys

import numpy as np
import torch

TJ = importlib.import_module("artifacts.experiments.20261119_increment_trajectory.trajectory")
LZ, HERE, COHORTS, CAP = TJ.LZ, TJ.HERE, TJ.COHORTS, TJ.CAP

CONDITIONS = [(LZ.NOOP, 0), (LZ.DO, 0), (LZ.NOOP, 1), (LZ.DO, 1)]     # localize's F4/HUD conditions
NAMES = ["N/N", "D/N", "N/D", "D/D"]                                  # direct action / post-action belief


@torch.no_grad()
def run(world, step):
    LZ.M.FunctionalMamba3.forward = LZ.mamba3_cpu
    model, cfg, sha = LZ.load(world, step)
    bank = np.load(LZ.HERE / "bank.npz")
    rows, count = {k: [] for k in COHORTS}, {k: 0 for k in COHORTS}
    for i, (item, row, ep) in enumerate(zip(bank["kind"], bank["rows"], bank["endpoints"])):
        if item not in count or count[item] >= CAP:
            continue
        count[item] += 1
        target = LZ.CV.MINING[item.split("_")[0]][2]
        hist, a, r, d, fix, future = LZ.history_frames(model, cfg, row, ep)
        reads, rec = LZ.record_tape(model, hist.p, target)
        prev, bp = LZ.belief_tape(model, hist, a, r, d, reads)
        first, _ = LZ.physical(model, cfg, hist, a, fix, prev, bp, rec, None, CONDITIONS,
                               torch.Generator().manual_seed(6111903 + i), True)
        vals = LZ.read_positions(first, future, [target, int(row[-1])]).reshape(len(CONDITIONS), LZ.DRAWS, 2)
        rows[item].append({"episode": int(row[0]), "t": int(row[1]), "move": vals.mean(1).tolist()})
    out = {"sha256": sha, "world": world, "step": step, "conditions": NAMES, "rows": rows}
    (HERE / f"routes_{world}_{step:02d}.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in rows.items():
        m = np.array([x["move"] for x in v]).mean(0)
        print(world, step, k, "digit", {n: round(float(m[j, 0] - m[0, 0]), 3) for j, n in enumerate(NAMES) if j},
              "tile", {n: round(float(m[j, 1] - m[0, 1]), 3) for j, n in enumerate(NAMES) if j}, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    for spec in sys.argv[1:]:
        w, s = spec.split(":")
        run(w, int(s))
