"""Why each revisit lookup fails, by the light: on the real filter (the fifth localizer; with `scatter`, its match
swapped for the within-place metric as in `metric_chain.py`), diagnosis episodes 0-22. Each lookup (reading 2's
definition) is recorded with: dark now (light < 0.5), the light change since that world cell was last seen, the gap,
whether the address predicted before the frame was right, and whether the right entry sits at the address, within the
read's 3 x 3, within 5 x 5, or anywhere. Simulator positions and light score only.

    LOC_CKPT=localizer_v5.pt LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261123_located_memory.diagnosis_v5.lookup_light [n_eps] [first_ep] [scatter]
"""
import importlib, json, sys

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")


@torch.no_grad()
def main(n_eps=6, first=0, scatter=0):
    tag = ""
    if scatter:
        MC = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v5.metric_chain")
        MC.patch(*MC.scatter()); tag = "_scatter"
    dev = torch.device("cpu")
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / LM.LOC_CKPT, map_location=dev)); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[first:first + n_eps]
    rows = []
    for ei, r in enumerate(data, first):
        mp = LM.Map(1, dev)
        T, pos, light = r["T"], r["pos"], r["light"].numpy()
        tc = LM.truth_cells(pos)
        seen = {}
        for t in range(T + 1):
            one = lambda k: r[k][t][None]
            tt = torch.tensor([t])
            pose_before = mp.pose[0].clone()
            ent = mp.view()[0][0]
            pres = ent[..., LM.CTIME] >= 0
            truth = ent[..., LM.CTRUE].long()
            feats, xhat, chat = LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"),
                                           one("prev_action"), tt, tt, torch.tensor([t == 0]), torch.tensor([True]),
                                           truth=tc[t][None])
            if t > 0:
                dtrue = pos[t] - pos[t - 1]
                pok = bool(((xhat[0] - pose_before) == dtrue).all())
                prev = {(int(pos[t - 1][0]) + int(LM.KR[k]), int(pos[t - 1][1]) + int(LM.KC[k])) for k in range(LM.MAP)}
                for k in range(LM.MAP):
                    w = (int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k]))
                    if w in prev or w not in seen:
                        continue
                    rr_, cc_ = int(xhat[0, 0]) + int(LM.KR[k]), int(xhat[0, 1]) + int(LM.KC[k])
                    where = ((truth == torch.tensor(w)).all(-1) & pres).nonzero()
                    d = int((where - torch.tensor([rr_, cc_])).abs().max(1).values.min()) if len(where) else 99
                    t0 = seen[w]
                    rows.append((ei, t, int(light[t] < 0.5), float(abs(light[t] - light[t0])), t - t0, int(pok), d))
            for k in range(LM.MAP):
                seen[(int(tc[t, k, 0]), int(tc[t, k, 1]))] = t
        print("episode", ei, "lookups", len(rows), flush=True)
    (LM.HERE / "diagnosis_v5" / f"lookup_light{tag}_{first}_{first + n_eps}.json").write_text(
        json.dumps({"columns": ["ep", "t", "dark_now", "light_change_since_seen", "gap", "address_right", "distance_to_entry"],
                    "rows": rows}) + "\n")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
