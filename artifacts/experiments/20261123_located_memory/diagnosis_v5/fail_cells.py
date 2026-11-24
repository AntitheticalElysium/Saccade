"""Where the untrained fifth match loses moved daylight steps (true-pose map, simulator positions score only): per
failing step, the winning hypothesis vs the true one, the support size, and per cell the two scores; summed over
failures by which hypothesis wins each cell, with the L1 residual of each hypothesis's entry to the new token.

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v5.fail_cells [n_eps]
"""
import importlib, json, sys
from collections import Counter, defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")
OFFS = torch.stack([LM.NR, LM.NC], -1)


@torch.no_grad()
def cell_scores(loc, ent, c, v, t, hyp, mask):
    r = hyp[..., 0, None] + LM.KR; cc = hyp[..., 1, None] + LM.KC
    e = ent[torch.zeros_like(r), r.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)]
    age = t[:, None, None] - e[..., LM.CTIME]
    present = (e[..., LM.CTIME] >= 0) & mask[:, None] & (LM.CELLS != LM.SELF)
    now = loc.now_terms(v); null = loc.null(c, now)
    ent_t = loc.entry_terms(e, age, present)
    s = loc.score(c[:, None], tuple(x[:, None] for x in now), ent_t, present, null[:, None])
    resid = (c[:, None] - (ent_t[0] + now[0][:, None])).abs().sum(-1)
    return s[0], present[0], resid[0], e[0]


@torch.no_grad()
def main(n_eps=3):
    loc = LM.Localizer(); print(LM.calibrate(loc, torch.load(LM.DATA / "tokens_train.pt"))); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    win_kind = Counter(); sums = defaultdict(list); n_fail = n_all = 0; examples = []
    for ei, r in enumerate(data):
        mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos); base = pos[0] - LM.O
        for t in range(r["T"] + 1):
            one = lambda k: r[k][t][None]
            tt = torch.tensor([t]); pose = (pos[t] - base)[None]
            a = int(r["prev_action"][t]) if t > 0 else -1
            if t > 0 and a in LM.GT.MOVE and float(r["light"][t]) >= 0.5 and (pos[t] != pos[t - 1]).any():
                n_all += 1
                prev_pose = (pos[t - 1] - base)[None]; ent = mp.view()[0]
                hyp = (prev_pose[:, None] + OFFS[None]).clamp(0, LM.G - 1)
                mask = LM.support(ent, prev_pose, tt)
                s, pr, res, e = cell_scores(loc, ent, one("p")[:, :LM.MAP].float(), one("v").float(), tt.float(), hyp, mask)
                tot = s.sum(1); d = (pos[t] - pos[t - 1]).tolist(); it = [i for i in range(9) if OFFS[i].tolist() == d][0]
                w = int(tot.argmax())
                if w != it:
                    n_fail += 1
                    win_kind["stay" if w == 4 else ("opposite" if (OFFS[w] == -OFFS[it]).all() else "other")] += 1
                    better = s[w] > s[it]
                    sums["support"].append(int(mask.sum()))
                    sums["cells_wrong_wins"].append(int((better & pr[w]).sum())); sums["cells_true_wins"].append(int((~better & pr[it]).sum()))
                    sums["wrong_gain_on_its_cells"].append(float((s[w] - s[it])[better].sum()))
                    sums["true_gain_on_its_cells"].append(float((s[it] - s[w])[~better].sum()))
                    sums["resid_true_median"].append(float(res[it][pr[it]].median())); sums["resid_wrong_median"].append(float(res[w][pr[w]].median()))
                    sums["true_entry_age_median"].append(float((tt - e[it][..., LM.CTIME])[pr[it]].median()))
                    if len(examples) < 4:
                        examples.append({"ep": ei, "t": t, "true": d, "won": OFFS[w].tolist(), "totals": [round(float(tot[it]), 1), round(float(tot[w]), 1)],
                                         "true_cells": s[it].reshape(7, 9).round(decimals=1).tolist(),
                                         "won_cells": s[w].reshape(7, 9).round(decimals=1).tolist(),
                                         "true_resid": res[it].reshape(7, 9).round(decimals=2).tolist()})
            LM.write(mp, pose, torch.ones(1), one("p").float(), one("v").float(), one("g").float(), one("fix"), tt, tt,
                     torch.tensor([True]), truth=tc[t][None])
            mp.last_p = one("p").float(); mp.last_v = one("v").float()
    out = {"moved_day_steps": n_all, "failures": n_fail, "winner_kind": dict(win_kind),
           "medians_over_failures": {k: float(np.median(v)) for k, v in sums.items()}, "examples": examples}
    txt = json.dumps(out, indent=1); print(txt)
    (LM.HERE / "diagnosis_v5" / f"fail_cells_untrained_{n_eps}eps.json").write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
