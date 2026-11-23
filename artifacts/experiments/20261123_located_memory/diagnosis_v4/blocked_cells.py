"""On the fourth localizer's blocked dark steps: is the commanded move's memory closer in mean or kinder in scale than
the sensor-fixed explanation (the same screen cell, previous frame)? Per cell (the 9 x 7 map cells): the L1 residual to
each mean, each log-scale, each log density, and the learned offset's size; medians over blocked dark steps.

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v4.blocked_cells [n_eps]
"""
import importlib, json, sys
from collections import defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v4.memory_v4")


@torch.no_grad()
def main(n_eps=3):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v4.pt", map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    a1 = loc.age(LM.PR.age_features(torch.ones(1), torch.ones(1, dtype=torch.bool)))[0, 0]
    acc = defaultdict(list)
    for r in data:
        mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
        for t in range(r["T"] + 1):
            one = lambda k: r[k][t][None]
            first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
            a = int(r["prev_action"][t]) if t > 0 else -1
            if t > 0 and a in LM.GT.MOVE and (pos[t] == pos[t - 1]).all() and float(r["light"][t]) < 0.5:
                ent = mp.view()[0]; c = one("p")[0, :LM.MAP].float(); v = one("v").float()
                d, s, tau = (x[0] for x in loc.now_terms(v)); dp, sp, tp = (x[0] for x in loc.now_terms(mp.last_v))
                cp = mp.last_p[0, :LM.MAP]
                id_mean = cp - dp + d; id_lb = sp + a1 + s
                pz = mp.pose[0] + torch.tensor(LM.GT.MOVE[a])
                rr = (pz[0] + LM.KR).clamp(0, LM.G - 1); cc = (pz[1] + LM.KC).clamp(0, LM.G - 1)
                e = ent[0, rr, cc]; age = tt - e[..., LM.CTIME]; pres = e[..., LM.CTIME] >= 0
                e0, s_e, t_e = loc.entry_terms(e, age, pres)
                mv_mean = e0 + d; mv_lb = s_e + s
                acc["resid_identity"].append((c - id_mean).abs().sum(-1)); acc["resid_move"].append((c - mv_mean).abs().sum(-1))
                acc["logb_identity"].append(id_lb); acc["logb_move"].append(mv_lb)
                acc["L_identity"].append(LM.laplace(c, id_mean, id_lb)); acc["L_move"].append(LM.laplace(c, mv_mean, mv_lb))
                acc["offset_L1"].append(d.abs().sum(-1)); acc["move_entry_age"].append(age.float())
                acc["move_entry_cpos_is_neighbour"].append((e[..., LM.CPOS] != LM.CELLS.float()).float())
            LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"), one("prev_action"),
                       tt, tt, first, valid, truth=tc[t][None])
    out = {"steps": len(acc["L_move"])}
    for k, v in acc.items():
        m = torch.stack(v).median(0).values
        out[k] = {"median_over_cells": round(float(m.median()), 3), "centre_row": [round(float(x), 2) for x in m.reshape(7, 9)[3]]}
    out["move_beats_identity_share"] = float((torch.stack(acc["L_move"]) > torch.stack(acc["L_identity"])).float().mean())
    txt = json.dumps(out, indent=1); print(txt)
    (LM.HERE / "diagnosis_v4" / f"blocked_cells_localizer_v4_{n_eps}eps.json").write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
