"""P1.6 (user 11-20): P1.5's recipe plus G1, the foveal record ("write where you look"), and the world-eye (user 11-20:
"Bundle the world-eye into P1.6, two refreshes are fine"; PLAN.md). From scratch, P1.5's schedule and world seed;
resumable; --updates stops early, --steps fixes the chunk length (the throughput checks)."""
import argparse
import importlib
import os
from pathlib import Path

from saccade import train as TR

HERE = Path(__file__).resolve().parent
P15 = importlib.import_module("artifacts.experiments.20261116_p15.train")
F61 = P15.F61                                                            # the worker's interface (run.py)
WORLD_SEED = P15.WORLD_SEED
FOVEA_MODES = 8          # the first 8 modes of the record's own S4D basis: 2 x 64 x 8 = 1,024 features per cell, the
                         # periphery record's size (its measured memory fits, P1.5 checks: 3.08 GiB); recency resolved to
                         # ~4.6 steps (mode 7 turns 7 pi / 32 = 0.69 rad per step) inside the basis's 64-step decay

WORLD_EYE = {"refresh_after": [5, 6],   # 20261120_switching_dissection C11: the rule declared before the run picked the
                                        # two fills after passes 5 and 6 (entities 0.479 vs chooser 0.466 vs fresh 0.481
                                        # over the ten passes, zombies equal); a cache is as good as the world that filled
                                        # it (C11's rows flat), and early worlds make poor eyes
             "seed": 2026112006}        # the fills' own draws (fixation prior where a frame has no look; noise)


def recipe(route_by_role=True):
    cfg = P15.check_recipe(P15.recipe(route_by_role))
    cfg["record"] = dict(cfg["record"], fovea_modes=FOVEA_MODES)
    cfg["fixation"] = dict(cfg["fixation"], world_eye=dict(WORLD_EYE))
    return cfg


def check_recipe(cfg):
    assert cfg in (recipe(), recipe(False)), "P1.6 recipe differs from P1.5 plus the foveal record and the world-eye"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--no-checkpoints", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    TR.p1(check_recipe(recipe()), WORLD_SEED, args.out, updates=args.updates, steps=args.steps,
          checkpoints=not args.no_checkpoints)


if __name__ == "__main__":
    main()
