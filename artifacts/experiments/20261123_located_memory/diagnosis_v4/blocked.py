"""Why does the fourth localizer's match never say 'no motion' on blocked dark steps? On the panel's first episodes
(simulator positions score only), at blocked movement commands: the summed memory score under 'no motion' and under
the commanded move, each split into present cells where memory beats the null, present cells where it doesn't, and
the count of present cells; day / dark.

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v4.blocked [n_eps]
"""
import importlib, json, sys
from collections import defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v4.memory_v4")


@torch.no_grad()
def parts(loc, ent, c, v, t, pose, prev):
    rr = pose[0] + LM.KR; cc = pose[1] + LM.KC
    inside = (rr >= 0) & (rr < LM.G) & (cc >= 0) & (cc < LM.G)
    e = ent[0, rr.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)][None]
    age = t[:, None] - e[..., LM.CTIME]; present = inside[None] & (e[..., LM.CTIME] >= 0)
    now = loc.now_terms(v); null = loc.null(c, now, prev)
    sc = loc.score(c, now, loc.entry_terms(e, age, present), present, null)[0]
    pr = present[0]
    return {"total": float(sc.sum()), "gain_cells": float(sc[pr & (sc > 0)].sum()), "cost_cells": float(sc[pr & (sc <= 0)].sum()),
            "present": int(pr.sum())}


@torch.no_grad()
def main(n_eps=3):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v4.pt", map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    acc = defaultdict(lambda: defaultdict(list))
    for r in data:
        mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
        for t in range(r["T"] + 1):
            one = lambda k: r[k][t][None]
            first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
            a = int(r["prev_action"][t]) if t > 0 else -1
            if t > 0 and a in LM.GT.MOVE and (pos[t] == pos[t - 1]).all():
                ent = mp.view()[0]; prev = (mp.last_p[:, :LM.MAP], mp.last_v, torch.tensor([True]))
                c = one("p")[:, :LM.MAP].float(); v = one("v").float()
                stay = parts(loc, ent, c, v, tt, mp.pose[0], prev)
                move = parts(loc, ent, c, v, tt, mp.pose[0] + torch.tensor(LM.GT.MOVE[a]), prev)
                key = "dark" if float(r["light"][t]) < 0.5 else "day"
                for name, x in (("stay", stay), ("commanded_move", move)):
                    for k_, val in x.items():
                        acc[key][f"{name}_{k_}"].append(val)
                acc[key]["move_wins"].append(float(move["total"] > stay["total"]))
            LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"), one("prev_action"),
                       tt, tt, first, valid, truth=tc[t][None])
    out = {k: {n: round(float(np.median(v)), 2) if not n.startswith("move_wins") else round(float(np.mean(v)), 3)
               for n, v in d.items()} | {"steps": len(d["move_wins"])} for k, d in acc.items()}
    txt = json.dumps(out, indent=1); print(txt)
    (LM.HERE / "diagnosis_v4" / f"blocked_localizer_v4_{n_eps}eps.json").write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
