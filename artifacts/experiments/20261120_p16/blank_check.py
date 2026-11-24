"""The added test's premise (PLAN, 11-21): under blank_fovea the record's fovea group is exactly zero and its periphery
group is untouched; without it the fovea group carries the glimpse. One DEV episode through S1's encode.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.blank_check CHECKPOINT
"""
import importlib, json, sys

import numpy as np
import torch

from saccade import model as M
from saccade.eval import s1 as S

E = importlib.import_module("artifacts.experiments.20261120_p16.evaluation")
AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")

if __name__ == "__main__":
    model, cfg = AI.load(sys.argv[1]); model.requires_grad_(False).eval()
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 1)
    n0 = M.Record.size(cfg, coarse=True)
    with torch.no_grad():
        a = S.encode(model, L, cfg)[0]
        E.blank_fovea()
        b = S.encode(model, L, cfg)[0]
    res = {"fovea_group_intact_max": float(a["record_every"][..., n0:].abs().max()),
           "fovea_group_blanked_max": float(b["record_every"][..., n0:].abs().max()),
           "periphery_group_change": float((a["record_every"][..., :n0] - b["record_every"][..., :n0]).abs().max()),
           "mamba_read_change": float((a["reads"].float() - b["reads"].float()).abs().max())}
    res["passed"] = res["fovea_group_intact_max"] > 0 and res["fovea_group_blanked_max"] == 0 and res["periphery_group_change"] == 0 \
        and res["mamba_read_change"] > 0
    print(json.dumps(res), flush=True)
    assert res["passed"], res
