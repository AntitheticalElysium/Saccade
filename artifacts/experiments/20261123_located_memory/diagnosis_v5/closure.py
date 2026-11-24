"""Could the fifth localizer's own evidence have closed the loop? On the real filter (diagnosis episodes), at each
step where a revisit lookup's world cell is stored under an older drift (6 or more cells from the prior's address),
the summed match at the filter's pose (x_now) is compared with the match at the pose that stored entry implies
(x_old: the true position moved by that entry's drift), three ways:
  - as the filter scores (the local hypotheses' common support, every entry);
  - every present cell, every entry;
  - every present cell, entries older than 16 frames only (the recent trail left out).
The filter relocates only when a far pose beats the local one by about -log(eps / G^2) = 17.6 nats.

    LOC_CKPT=localizer_v5.pt LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261123_located_memory.diagnosis_v5.closure [n_eps] [first_ep]
"""
import importlib, json, sys

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")
TRAIL = 16


@torch.no_grad()
def main(n_eps=6, first=0):
    dev = torch.device("cpu")
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / LM.LOC_CKPT, map_location=dev)); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[first:first + n_eps]
    rows = []
    for ei, r in enumerate(data, first):
        mp = LM.Map(1, dev)
        T, pos = r["T"], r["pos"]
        tc = LM.truth_cells(pos)
        seen = set()
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
                prev = {(int(pos[t - 1][0]) + int(LM.KR[k]), int(pos[t - 1][1]) + int(LM.KC[k])) for k in range(LM.MAP)}
                hit = None
                for k in range(LM.MAP):
                    w = (int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k]))
                    if w in prev or w not in seen:
                        continue
                    rr_, cc_ = int(xhat[0, 0]) + int(LM.KR[k]), int(xhat[0, 1]) + int(LM.KC[k])
                    where = ((truth == torch.tensor(w)).all(-1) & pres).nonzero()
                    if not len(where):
                        continue
                    d = (where - torch.tensor([rr_, cc_])).abs().max(1).values
                    if int(d.min()) >= 6:
                        hit = (where[int(d.argmin())] - torch.tensor(w))          # that entry's drift (address - cell)
                        break
                if hit is not None:
                    x_now = mp.pose[0].clone()
                    x_old = (pos[t] + hit).clamp(0, LM.G - 1)
                    hyp = torch.stack([x_now, x_old])[None]
                    c, v = one("p")[:, :LM.MAP].float(), one("v").float()
                    mask = LM.support(ent[None], pose_before[None], tt)
                    allm = torch.ones(1, LM.MAP, dtype=torch.bool)
                    old = ent.clone(); old[..., LM.CTIME] = torch.where(old[..., LM.CTIME] < t - TRAIL, old[..., LM.CTIME],
                                                                         torch.full_like(old[..., LM.CTIME], -1.0))
                    out = {}
                    for name, e_, m_ in (("filter", ent, mask), ("all_cells", ent, allm), ("all_cells_old_only", old, allm)):
                        L = LM.loglik_at(loc, e_[None], c, v, tt.float(), hyp, mask=m_)[0]
                        # cells each hypothesis is scored on
                        rr = hyp[0, :, 0, None] + LM.KR; cc = hyp[0, :, 1, None] + LM.KC
                        ins = (rr >= 0) & (rr < LM.G) & (cc >= 0) & (cc < LM.G)
                        pr = ins & (e_[rr.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)][..., LM.CTIME] >= 0) & m_ & (LM.CELLS != LM.SELF)
                        tr_ = e_[rr.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)][..., LM.CTRUE].long()        # [2, 63, 2]
                        want = torch.stack([pos[t][0] + LM.KR, pos[t][1] + LM.KC], -1)
                        same = (tr_ == want[None]).all(-1) & pr                          # the stored entry is this cell
                        out[name] = {"L_now": float(L[0]), "L_old": float(L[1]), "cells_now": int(pr[0].sum()), "cells_old": int(pr[1].sum()),
                                     "same_cell_now": int(same[0].sum()), "same_cell_old": int(same[1].sum())}
                    rows.append({"ep": ei, "t": t, "dark": int(float(r["light"][t]) < 0.5),
                                 "drift_gap": int((x_old - x_now).abs().max()), **out})
            for k in range(LM.MAP):
                seen.add((int(tc[t, k, 0]), int(tc[t, k, 1])))
        print("episode", ei, "rows", len(rows), flush=True)
    res = {"episodes": [first, first + n_eps], "steps": len(rows), "trail_frames": TRAIL, "jump_cost_nats": 17.55}
    for name in ("filter", "all_cells", "all_cells_old_only"):
        dL = np.array([x[name]["L_old"] - x[name]["L_now"] for x in rows])
        res[name] = {"old_minus_now_median": float(np.median(dL)), "old_minus_now_q25_q75": [float(np.quantile(dL, .25)), float(np.quantile(dL, .75))],
                     "share_old_wins": float((dL > 0).mean()), "share_old_beats_jump_cost": float((dL > 17.55).mean()),
                     "cells_now_mean": float(np.mean([x[name]["cells_now"] for x in rows])),
                     "cells_old_mean": float(np.mean([x[name]["cells_old"] for x in rows])),
                     "same_cell_now_mean": float(np.mean([x[name]["same_cell_now"] for x in rows])),
                     "same_cell_old_mean": float(np.mean([x[name]["same_cell_old"] for x in rows]))}
        for dk, lab in ((0, "day"), (1, "dark")):
            sel = np.array([x["dark"] == dk for x in rows])
            if sel.any():
                res[name][lab] = {"n": int(sel.sum()), "share_old_beats_jump_cost": float((dL[sel] > 17.55).mean()),
                                  "old_minus_now_median": float(np.median(dL[sel]))}
    (LM.HERE / "diagnosis_v5" / f"closure_{first}_{first + n_eps}.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
