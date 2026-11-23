"""Are P1.5's coarse tokens the same when the same world cell is seen at another screen cell? On the panel's episodes
(simulator positions score only): the L1 change (16 dims) between frames for the SAME world cell,
- after a command that doesn't move the agent: the same screen cell (static);
- after a real move: seen one screen cell over (cross-cell);
raw and after the closed-form per-cell calibration, by day / dark, the agent's own cell and the edge rows/columns
excluded; and the same split for cells identical to their neighbour in the view (uniform terrain).

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v5.equivariance [n_eps]
"""
import importlib, json, sys
from collections import defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")


@torch.no_grad()
def main(n_eps=20):
    loc = LM.Localizer(); LM.calibrate(loc, torch.load(LM.DATA / "tokens_train.pt"))
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    acc = defaultdict(list)
    for r in data:
        p, v, pos, light = r["p"][:, :LM.MAP].float(), r["v"].float(), r["pos"], r["light"]
        d_all, _ = loc.calib(LM.CELLS.expand(len(v), LM.MAP), v[:, None].expand(len(v), LM.MAP, 8))
        q = p - d_all
        for t in range(1, r["T"] + 1):
            a = int(r["prev_action"][t]); d = (pos[t] - pos[t - 1]).tolist()
            dark = "dark" if float(light[t]) < 0.5 else "day"
            if a not in LM.GT.MOVE:
                kind, src = "static_same_cell", LM.CELLS
            elif d != [0, 0]:
                kind = "moved_cross_cell"
                kr, kc = LM.KR + d[0], LM.KC + d[1]                            # where this world cell was on screen at t - 1
                src = (kr + 3) * 9 + (kc + 4)
            else:
                continue
            for k in range(LM.MAP):
                rr, cc = divmod(k, 9)
                if k == LM.SELF or rr in (0, 6) or cc in (0, 8):
                    continue
                kp = int(src[k])
                if kp == LM.SELF:
                    continue
                uni = bool((p[t, k] - p[t, LM.NB[k]]).abs().sum() < 1e-3) and LM.NB[k] != k
                for name, x in (("raw", p), ("calibrated", q)):
                    acc[(kind, dark, name, "uniform" if uni else "other")].append(float((x[t, k] - x[t - 1, kp]).abs().sum()))
    out = {}
    for k, vals in sorted(acc.items()):
        a = np.array(vals)
        out["|".join(k)] = {"n": len(a), "median": round(float(np.median(a)), 3), "mean": round(float(a.mean()), 3),
                            "exact_share": round(float((a < 1e-4).mean()), 3)}
    txt = json.dumps(out, indent=1); print(txt)
    (LM.HERE / "diagnosis_v5" / f"equivariance_{n_eps}eps.json").write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
