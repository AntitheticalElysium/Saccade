"""Does time, viewpoint or lighting break the fifth localizer's match between two sightings of a world cell? Daylight
sightings only (light >= 0.5) on the diagnosis episodes, each paired with the previous sighting of the same world
cell (simulator positions pair them; scoring only), tallied by the gap, the change in daylight between them, and how
far apart on the screen they were seen.

    LOC_CKPT=localizer_v5.pt LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261123_located_memory.diagnosis_v5.light_gap [n_eps] [first_ep]
"""
import importlib, json, sys
from collections import defaultdict

import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")


def gap_bin(g):
    return "1" if g == 1 else "2-16" if g <= 16 else "17-64" if g <= 64 else "65+"


def light_bin(dl):
    return "<0.005" if dl < 0.005 else "0.005-0.02" if dl < 0.02 else "0.02-0.1" if dl < 0.1 else ">=0.1"


@torch.no_grad()
def main(n_eps=23, first=0):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / LM.LOC_CKPT, map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[first:first + n_eps]
    tal = defaultdict(lambda: [0, 0]); cells = torch.arange(LM.MAP)
    for r in data:
        T, pos = r["T"], r["pos"]; p = r["p"][:, :LM.MAP].float(); v = r["v"].float(); light = r["light"].numpy()
        _, ls = loc.calib(cells.expand(T + 1, LM.MAP), v[:, None].expand(T + 1, LM.MAP, 8))[:2]
        last = {}
        for t in range(T + 1):
            for k in range(LM.MAP):
                if k == LM.SELF:
                    continue
                w = (int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k]))
                if w in last:
                    t0, k0 = last[w]
                    if light[t] >= 0.5 and light[t0] >= 0.5:
                        sd = max(abs(int(LM.KR[k]) - int(LM.KR[k0])), abs(int(LM.KC[k]) - int(LM.KC[k0])))
                        x = (p[t, k] - p[t0, k0]).abs().sum() / torch.logaddexp(ls[t, k], ls[t0, k0]).exp()
                        hit = int(float(x) < float(loc.thr))
                        for key in ((gap_bin(t - t0), light_bin(abs(light[t] - light[t0]))),
                                    (gap_bin(t - t0), light_bin(abs(light[t] - light[t0])), "shift<=1" if sd <= 1 else "shift>=2")):
                            tal[key][0] += hit; tal[key][1] += 1
                last[w] = (t, k)
    out = {"episodes": [first, first + n_eps], "thr": float(loc.thr),
           "match": {" | ".join(k): {"n": n, "match": h / n} for k, (h, n) in sorted(tal.items())}}
    (LM.HERE / "diagnosis_v5" / f"light_gap_{first}_{first + n_eps}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
