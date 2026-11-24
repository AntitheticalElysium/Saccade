"""Dream throughput by batch. PLAN.md, second reading (2026-11-23).

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_step_costs.dream_scaling
"""
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

SC = importlib.import_module("artifacts.experiments.20261123_step_costs.step_costs")
HERE = Path(__file__).resolve().parent


def main():
    res = {}
    for B in (64, 128, 256):
        torch.cuda.reset_peak_memory_stats()
        r = SC.world(B, 20)
        res[f"batch_{B}"] = {"dream_frame_ms": r["dream_frame_ms"], "dream_ms_per_stream": r["dream_frame_ms"] / B,
                             "peak_gib": r["peak_gib"]}
        print(B, res[f"batch_{B}"], flush=True)
    (HERE / "dream_scaling.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
