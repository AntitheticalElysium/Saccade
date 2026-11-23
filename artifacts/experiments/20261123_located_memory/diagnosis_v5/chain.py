"""Where the trained fifth localizer's errors come from, on the real filter (its own map, full grid, as reading 2
scores it), on the diagnosis episodes (0-22; 23 on stay unseen). Simulator positions score only.

At each movement command: prior and posterior right or wrong; whether the match alone (the 9 local hypotheses
around the previous pose, on their common support) ranks the true one first, ties it, or loses; the prior's mass on
the true offset; and how many of the support cells were written under the same drift as the current pose (the map
consistent with it). At each revisit lookup (reading 2's definition): whether the prior was right, whether the
right entry sits at the address, within the read's 9 candidates, or elsewhere in the map under another drift (and how
far).

    LOC_CKPT=localizer_v5.pt LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261123_located_memory.diagnosis_v5.chain [n_eps] [first_ep]
"""
import importlib, json, sys
from collections import Counter, defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")
OFFS = torch.stack([LM.NR, LM.NC], -1)


@torch.no_grad()
def main(n_eps=23, first=0):
    dev = torch.device("cpu")
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / LM.LOC_CKPT, map_location=dev)); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[first:first + n_eps]
    steps, looks = [], Counter()
    far = Counter()
    for ei, r in enumerate(data, first):
        mp = LM.Map(1, dev)
        T, pos = r["T"], r["pos"]
        tc = LM.truth_cells(pos)
        seen = set()
        for t in range(T + 1):
            one = lambda k: r[k][t][None]
            tt = torch.tensor([t])
            pose_before = mp.pose[0].clone()
            ent = mp.view()[0][0]                                     # [G, G, NE] before this frame
            pres = ent[..., LM.CTIME] >= 0
            truth = ent[..., LM.CTRUE].long()
            feats, xhat, chat = LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"),
                                           one("prev_action"), tt, tt, torch.tensor([t == 0]), torch.tensor([True]),
                                           truth=tc[t][None])
            if t == 0:
                for k in range(LM.MAP):
                    seen.add((int(tc[t, k, 0]), int(tc[t, k, 1])))
                continue
            a = int(r["prev_action"][t]); dtrue = (pos[t] - pos[t - 1])
            drift_now = pose_before - pos[t - 1]                      # address minus world cell, for the pose before
            if a in LM.GT.MOVE:
                kt = int(((OFFS == dtrue[None]).all(1)).nonzero()[0])
                L = mp.last_L[0]
                hyp = (pose_before[None] + OFFS).clamp(0, LM.G - 1)
                Ll = L[hyp[:, 0], hyp[:, 1]]
                others = torch.cat([Ll[:kt], Ll[kt + 1:]])
                logK = loc.kernel(torch.as_tensor(r["p"][t - 1][None]).float(), torch.as_tensor(r["v"][t - 1][None]).float(),
                                  torch.tensor([a]))[0]
                mask = LM.support(ent[None], pose_before[None], tt)[0]
                # support cells under the true hypothesis: written under the current drift?
                h = pose_before + dtrue
                rr, cc = h[0] + LM.KR, h[1] + LM.KC
                ins = (rr >= 0) & (rr < LM.G) & (cc >= 0) & (cc < LM.G)
                m = mask & ins
                e_tr = truth[rr.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)]
                addr = torch.stack([rr, cc], -1)
                cons = ((addr - e_tr) == drift_now[None]).all(1) & m
                steps.append(dict(ep=ei, t=t, moved=int((dtrue != 0).any()), dark=int(float(r["light"][t]) < 0.5),
                                  prior_ok=int(((xhat[0] - pose_before) == dtrue).all()),
                                  post_ok=int(((mp.pose[0] - pose_before) == dtrue).all()),
                                  jump=int((mp.pose[0] - pose_before).abs().max() > 1),
                                  match=("win" if bool(Ll[kt] > others.max()) else "tie" if bool(Ll[kt] == others.max()) else "lose"),
                                  margin=float(Ll[kt] - others.max()), prior_true=float(logK[kt].exp()),
                                  support=int(m.sum()), consistent=int(cons.sum()), conf=float(mp.conf[0])))
            # lookups (reading 2's definition): cells entering view whose world cell was seen before
            prev = {(int(pos[t - 1][0]) + int(LM.KR[k]), int(pos[t - 1][1]) + int(LM.KC[k])) for k in range(LM.MAP)}
            pok = bool(((xhat[0] - pose_before) == dtrue).all())
            for k in range(LM.MAP):
                w = (int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k]))
                if w in prev or w not in seen:
                    continue
                rr_, cc_ = int(xhat[0, 0]) + int(LM.KR[k]), int(xhat[0, 1]) + int(LM.KC[k])
                at = 0 <= rr_ < LM.G and 0 <= cc_ < LM.G and pres[rr_, cc_] and tuple(truth[rr_, cc_].tolist()) == w
                near = False
                for d in range(9):
                    r2, c2 = rr_ + int(LM.NR[d]), cc_ + int(LM.NC[d])
                    if 0 <= r2 < LM.G and 0 <= c2 < LM.G and pres[r2, c2] and tuple(truth[r2, c2].tolist()) == w:
                        near = True
                where = ((truth == torch.tensor(w)).all(-1) & pres).nonzero()
                key = ("prior_ok" if pok else "prior_wrong")
                looks[key, "n"] += 1
                looks[key, "at"] += int(at); looks[key, "near"] += int(near); looks[key, "stored"] += int(len(where) > 0)
                if len(where) and not at:
                    dd = (where - torch.tensor([rr_, cc_])).abs().max(1).values.min()
                    far[key, int(min(int(dd), 6))] += 1
            for k in range(LM.MAP):
                seen.add((int(tc[t, k, 0]), int(tc[t, k, 1])))
        print("episode", ei, "steps", T, flush=True)
    out = {"episodes": [first, first + n_eps], "ckpt": LM.LOC_CKPT}
    S = {k: np.array([s[k] for s in steps]) for k in steps[0] if k != "match"}
    match = np.array([s["match"] for s in steps])
    for name, sel in (("moved_day", (S["moved"] == 1) & (S["dark"] == 0)), ("moved_dark", (S["moved"] == 1) & (S["dark"] == 1)),
                      ("blocked_day", (S["moved"] == 0) & (S["dark"] == 0)), ("blocked_dark", (S["moved"] == 0) & (S["dark"] == 1))):
        n = int(sel.sum())
        wrong = sel & (S["post_ok"] == 0)
        cons = S["consistent"] / np.maximum(S["support"], 1)
        out[name] = {"n": n, "prior_ok": float(S["prior_ok"][sel].mean()), "post_ok": float(S["post_ok"][sel].mean()),
                     "match_win": float((match[sel] == "win").mean()), "match_tie": float((match[sel] == "tie").mean()),
                     "match_lose": float((match[sel] == "lose").mean()),
                     "support_mean": float(S["support"][sel].mean()), "consistent_share_mean": float(cons[sel].mean()),
                     "post_wrong": {"n": int(wrong.sum()),
                                    "match_win": int((wrong & (match == "win")).sum()), "match_tie": int((wrong & (match == "tie")).sum()),
                                    "match_lose_map_consistent": int((wrong & (match == "lose") & (cons >= 0.9)).sum()),
                                    "match_lose_map_inconsistent": int((wrong & (match == "lose") & (cons < 0.9)).sum()),
                                    "jumps": int((wrong & (S["jump"] == 1)).sum()),
                                    "support_zero": int((wrong & (S["support"] == 0)).sum()),
                                    "prior_true_mass_median": float(np.median(S["prior_true"][wrong])) if wrong.any() else None},
                     "prior_true_mass_median": float(np.median(S["prior_true"][sel]))}
    out["lookups"] = {f"{a}_{b}": v for (a, b), v in looks.items()}
    out["lookups_misplaced_distance"] = {f"{a}_{d}": v for (a, d), v in sorted(far.items())}
    (LM.HERE / "diagnosis_v5" / f"chain_{LM.LOC_CKPT.replace('.pt', '')}_{first}_{first + n_eps}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
