"""The fifth localizer's match between two sightings of the same world cell (simulator positions pair them; scoring
only), on the diagnosis episodes. Each map-cell observation is paired with the previous sighting of the same world
cell; the binary match (change within the calibrated pair scale times the threshold) is tallied by the gap between
the sightings, how far apart on the screen they were seen, and the lighting at each.

    LOC_CKPT=localizer_v5.pt LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261123_located_memory.diagnosis_v5.repeat [n_eps] [first_ep]
"""
import importlib, json, sys
from collections import defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")


def gap_bin(g):
    return "1" if g == 1 else "2-16" if g <= 16 else "17-64" if g <= 64 else "65+"


@torch.no_grad()
def main(n_eps=23, first=0):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / LM.LOC_CKPT, map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[first:first + n_eps]
    tally = defaultdict(lambda: [0, 0])
    ratio = defaultdict(list)
    cells = torch.arange(LM.MAP)
    for r in data:
        T, pos = r["T"], r["pos"]
        p, v = r["p"][:, :LM.MAP].float(), r["v"].float()
        light = r["light"].numpy()
        d, ls = loc.calib(cells.expand(T + 1, LM.MAP), v[:, None].expand(T + 1, LM.MAP, 8))[:2]   # per frame, cell
        last = {}
        for t in range(T + 1):
            for k in range(LM.MAP):
                if k == LM.SELF:
                    continue
                w = (int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k]))
                if w in last:
                    t0, k0 = last[w]
                    x = float((p[t, k] - p[t0, k0]).abs().sum() / torch.logaddexp(ls[t, k], ls[t0, k0]).exp())
                    hit = x < float(loc.thr)
                    sd = max(abs(int(LM.KR[k]) - int(LM.KR[k0])), abs(int(LM.KC[k]) - int(LM.KC[k0])))
                    lt = ("day" if light[t] >= 0.5 else "dark") + "/" + ("day" if light[t0] >= 0.5 else "dark")
                    keys = [("gap", gap_bin(t - t0)), ("screen_shift", str(min(sd, 4))), ("light_now/then", lt),
                            ("gap_x_shift", gap_bin(t - t0) + "|" + str(min(sd, 4))),
                            ("daydaygap_x_shift", (gap_bin(t - t0) + "|" + str(min(sd, 4))) if lt == "day/day" else None),
                            ("daydaylightdiff", ("|dl|<0.02" if abs(light[t] - light[t0]) < 0.02 else "|dl|<0.1" if abs(light[t] - light[t0]) < 0.1 else "|dl|>=0.1") if lt == "day/day" else None)]
                    for kk in keys:
                        if kk[1] is None:
                            continue
                        tally[kk][0] += int(hit); tally[kk][1] += 1
                        ratio[kk].append(x)
                last[w] = (t, k)
    out = {"episodes": [first, first + n_eps], "thr": float(loc.thr)}
    for (a, b), (h, n) in sorted(tally.items()):
        out.setdefault(a, {})[b] = {"n": n, "match": h / n, "change_over_scale_median": float(np.median(ratio[(a, b)]))}
    (LM.HERE / "diagnosis_v5" / f"repeat_{first}_{first + n_eps}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
