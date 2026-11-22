"""Instrument check for RESULTS.md (11-23): does P1.5's coarse view determine the health digit, or was the 9.9% miss
the coarse reader's weakness? Exact renderer-atlas codebook read of the health cell's periphery token (the method of
20261104_learned_fine_candidates/hud_codebook.py and 20261122_rule_change/evidence_eda.py, here for health, cell 63),
on every frame of the same 160 DEV episodes, against the true health label.

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261122_dream_commitment.coarse_health_codebook
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from saccade import model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
AUD = importlib.import_module("artifacts.experiments.20261119_s2.dream_outcome_audit")
CC = importlib.import_module("artifacts.experiments.20261122_dream_commitment.commitment_checks")
HEALTH_CELL = 63


@torch.no_grad()
def codebook(model, atlas, p, v):
    """Nearest rendered health glyph (0-9) in periphery-token space, through the frozen encoder with the frame's v."""
    per = model.perception
    glyphs = torch.from_numpy(atlas).to(p.device).permute(0, 3, 1, 2).float()
    glyphs = F.adaptive_avg_pool2d(glyphs, per.acuity).round().flatten(1) / 127.5 - 1
    h = glyphs[None].expand(len(p), -1, -1)
    for layer, film in zip(per.p_layers, per.film):
        gain, shift = film(v)[:, None].chunk(2, -1)
        h = F.gelu(layer(h)) * (1 + gain) + shift
    codes = M.out_map(per.p_out(h), per.s)
    return (codes.float() - p[:, HEALTH_CELL, None, :].float()).square().mean(-1).argmin(1).cpu().numpy()


def main():
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    test, vi = AUD.encode_sample(model, cfg, "dev", 160, CC.SEED + 1)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    atlas = np.load(CC.ATLAS)[0]
    read, true = [], []
    for i, e in enumerate(vi):
        with TR.autocast(S2.DEV, cfg):
            read.append(codebook(model, atlas, test[i]["p"].to(S2.DEV).float(), test[i]["v"].to(S2.DEV).float()))
        true.append(np.minimum(L["s_body"][so[e]:so[e + 1], 0], 9).astype(int))
    exact = float(np.mean(np.concatenate([r == t for r, t in zip(read, true)])))
    drop_true = np.concatenate([np.diff(t) < 0 for t in true])
    drop_read = np.concatenate([r[1:] < t[:-1] for r, t in zip(read, true)])     # read digit below the true previous
    out = {"frames": int(sum(len(t) for t in true)), "exact": exact,
           "true_drops": int(drop_true.sum()), "drops_read": int((drop_true & drop_read).sum()),
           "false_drops": int((~drop_true & drop_read).sum()), "transitions": int(len(drop_true))}
    (HERE / "coarse_health_codebook.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
