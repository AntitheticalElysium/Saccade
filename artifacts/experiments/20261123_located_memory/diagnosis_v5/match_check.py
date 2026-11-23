"""The fifth match before any training (trust at 0.5, nothing learned but the closed-form calibration): on a map built
at the true poses (simulator positions score only), at each movement command, the argmax of the summed scores over the
9 local hypotheses around the true previous pose, on the support cells; by moved / blocked and day / dark. Also the
calibration's per-cell noise scales for a day and a dark view.

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v5.match_check [n_eps] [first_ep]
"""
import importlib, json, sys
from collections import defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")
OFFS = torch.stack([LM.NR, LM.NC], -1)


@torch.no_grad()
def main(n_eps=3, first=0, ckpt=None):
    loc = LM.Localizer()
    if ckpt:
        loc.load_state_dict(torch.load(LM.RUN / ckpt, map_location="cpu"))
    else:
        print("calibration", LM.calibrate(loc, torch.load(LM.DATA / "tokens_train.pt")), flush=True)
    loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[first:first + n_eps]
    acc = defaultdict(list); margin = defaultdict(list)
    for r in data:
        mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
        base = pos[0] - LM.O
        for t in range(r["T"] + 1):
            one = lambda k: r[k][t][None]
            tt = torch.tensor([t]); pose = (pos[t] - base)[None]
            a = int(r["prev_action"][t]) if t > 0 else -1
            if t > 0 and a in LM.GT.MOVE:
                prev_pose = (pos[t - 1] - base)[None]
                ent = mp.view()[0]
                hyp = (prev_pose[:, None] + OFFS[None]).clamp(0, LM.G - 1)
                mask = LM.support(ent, prev_pose, tt)
                L = LM.loglik_at(loc, ent, one("p")[:, :LM.MAP].float(), one("v").float(), tt.float(), hyp, mask=mask)[0]
                d = (pos[t] - pos[t - 1]).tolist(); it = [i for i in range(9) if OFFS[i].tolist() == d][0]
                key = ("moved" if d != [0, 0] else "blocked") + ("_dark" if float(r["light"][t]) < 0.5 else "_day")
                acc[key].append(int(L.argmax()) == it)
                other = torch.cat([L[:it], L[it + 1:]]).max()
                margin[key].append(float(L[it] - other))
                acc[key + "_vs_cmd_or_stay"].append(bool(L[it] > L[[i for i in range(9) if OFFS[i].tolist() in ([0, 0], list(LM.GT.MOVE[a])) and i != it][0]]))
            LM.write(mp, pose, torch.ones(1), one("p").float(), one("v").float(), one("g").float(), one("fix"), tt, tt,
                     torch.tensor([True]), truth=tc[t][None])
            mp.last_p = one("p").float(); mp.last_v = one("v").float()
    out = {"episodes": [first, first + n_eps], "match_alone": {k: [len(v), round(float(np.mean(v)), 3)] for k, v in sorted(acc.items())},
           "true_minus_best_other_median": {k: round(float(np.median(v)), 2) for k, v in sorted(margin.items())}}
    allv = torch.cat([r["v"].float() for r in data]); alll = torch.cat([r["light"].float() for r in data])
    for name, sel in (("day", alll > 0.9), ("dark", alll < 0.2)):
        if sel.any():
            _, ls = loc.calib(LM.CELLS, allv[sel][:1].expand(LM.MAP, 8))
            out[f"{name}_pair_scale_per_dim"] = (2 * ls.exp()).reshape(7, 9).round(decimals=4).tolist()
            d, _ = loc.calib(LM.CELLS, allv[sel][:1].expand(LM.MAP, 8))
            out[f"{name}_offset_L1"] = d.abs().sum(-1).reshape(7, 9).round(decimals=2).tolist()
    txt = json.dumps(out, indent=1); print(txt)
    (LM.HERE / "diagnosis_v5" / f"match_check_{ckpt or 'untrained'}_{first}_{n_eps}eps.json".replace(".pt", "")).write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, int(sys.argv[2]) if len(sys.argv) > 2 else 0,
         sys.argv[3] if len(sys.argv) > 3 else None)
