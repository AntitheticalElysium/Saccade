"""When does P1.5's denoiser learn, or fail to learn, the coal digit increment? (user 11-19: the cheap frozen read.)

Codex's localize.py (20261119_increment_localization), its functions unchanged, native condition only (no cut):
P1.5 checkpoints 1-10 on CPU, its bank's cohorts coal_1 (the regression), stone_2 (P1.5's 2->3 dip), stone_1 and
wood_unlock (references), 16 histories x 4 paired draws each, its seeds. Per history and checkpoint: the first
denoiser call's move along the true NOOP->DO step (0 = the NOOP successor's token, 1 = the DO successor's) for the
target digit cell and the faced tile, under both inputs DO minus both NOOP. Checkpoint 10 must reproduce Codex's
p15_10_paired.json native first-call values. F4's and HUD-Change's trajectories (Codex's files) are read beside it.

Read, fixed now: the coal digit contrast across checkpoints 6-10 (the 256-step half, lowest learning rates) rising
by more than its paired bootstrap interval = still learning when the schedule ended (a delayed transition the
schedule cut off); within it = flat (something holds the copy in place).

    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python -m artifacts.experiments.20261119_increment_trajectory.trajectory STEP [STEP ...]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

LZ = importlib.import_module("artifacts.experiments.20261119_increment_localization.localize")
HERE = Path(__file__).resolve().parent
COHORTS, CAP = ("coal_1", "stone_2", "stone_1", "wood_unlock"), 16


@torch.no_grad()
def run(step):
    LZ.M.FunctionalMamba3.forward = LZ.mamba3_cpu
    model, cfg, sha = LZ.load("p15", step)
    bank = np.load(LZ.HERE / "bank.npz")
    conditions = [(LZ.NOOP, 0), (LZ.DO, 1)]                              # localize's P1.5 conditions
    rows, count = {k: [] for k in COHORTS}, {k: 0 for k in COHORTS}
    for i, (item, row, ep) in enumerate(zip(bank["kind"], bank["rows"], bank["endpoints"])):
        if item not in count or count[item] >= CAP:
            continue
        count[item] += 1
        target = LZ.CV.MINING[item.split("_")[0]][2]
        hist, a, r, d, fix, future = LZ.history_frames(model, cfg, row, ep)
        reads, rec = LZ.record_tape(model, hist.p, target)
        prev, bp = LZ.belief_tape(model, hist, a, r, d, reads)
        first, _ = LZ.physical(model, cfg, hist, a, fix, prev, bp, rec, None, conditions,
                               torch.Generator().manual_seed(6111903 + i), True)    # localize's seed; first call only
        vals = LZ.read_positions(first, future, [target, int(row[-1])]).reshape(len(conditions), LZ.DRAWS, 2)
        rows[item].append({"episode": int(row[0]), "t": int(row[1]), "move": vals.mean(1).tolist()})
    out = {"sha256": sha, "step": step, "conditions": ["N/N", "D/D"], "rows": rows}
    (HERE / f"p15_{step:02d}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(step, {k: round(float(np.mean([x["move"][1][0] - x["move"][0][0] for x in v])), 3) for k, v in rows.items()}, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    for s in sys.argv[1:]:
        run(int(s))
