"""Why the label-free localizer misses (reading 2: lookups 0.394, posterior worse than prior on moved commands, better
on blocked). On the panel's movement steps (simulator positions score only): the true displacement against the prior's
and the posterior's, the match alone over the 9 offsets from the last pose, and the same match restricted to the cells
both hypotheses cover (true offset vs the chosen one). If the match prefers hypotheses that overlap more of the store,
the error is a coverage bias in how cells without an entry are scored.

    LOC_DEVICE=cpu MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v1.localizer_diagnosis
"""
import importlib, json, sys
from collections import Counter

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v1.memory_v1")   # the v1 code (54d2c13b)
GT = LM.GT


@torch.no_grad()
def cell_scores(loc, ent, c, v, t, hyp):
    """Per-cell scores and presence at hypotheses hyp [1, H, 2] -> ([H, 63], [H, 63])."""
    r = hyp[..., 0, None] + LM.KR; cc = hyp[..., 1, None] + LM.KC
    inside = (r >= 0) & (r < LM.G) & (cc >= 0) & (cc < LM.G)
    e = ent[torch.zeros_like(r), r.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)]
    age = t[:, None, None] - e[..., LM.CTIME]
    present = inside & (e[..., LM.CTIME] >= 0)
    s = loc.score(c[:, None], e[..., LM.CT], age, present, v[:, None, None].expand(1, hyp.shape[1], LM.MAP, 8))
    return s[0], present[0]


@torch.no_grad()
def main(n_eps=40):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v1.pt", map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    offs = torch.stack([LM.NR, LM.NC], -1)
    rows = []
    for r in data:
        mp = LM.Map(1, torch.device("cpu"))
        pos = r["pos"]
        tc = LM.truth_cells(pos)
        for t in range(r["T"] + 1):
            one = lambda k: r[k][t][None]
            first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
            before = mp.pose.clone()
            ent = mp.view()[0]
            if t > 0 and int(r["prev_action"][t]) in GT.MOVE:
                hyp = (before[:, None] + offs[None]).clamp(0, LM.G - 1)
                s, pr = cell_scores(loc, ent, one("p")[:, :LM.MAP].float(), one("v").float(), tt.float(), hyp)
                dtrue = (pos[t] - pos[t - 1]).tolist()
                it = [i for i in range(9) if offs[i].tolist() == dtrue]
                logK = loc.kernel(mp.last_p, mp.last_v, one("prev_action"))[0]
            feats, xhat, chat = LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"),
                                           one("prev_action"), tt, tt, first, valid, truth=tc[t][None])
            if t > 0 and int(r["prev_action"][t]) in GT.MOVE and it:
                i_t = it[0]
                i_post = int(((mp.pose[0] - before[0]) == offs).all(1).nonzero()[0, 0]) if ((mp.pose[0] - before[0]).abs().max() <= 1) else -1
                i_pri = int(((xhat[0] - before[0]) == offs).all(1).nonzero()[0, 0]) if ((xhat[0] - before[0]).abs().max() <= 1) else -1
                tot = s.sum(1)
                i_lik = int(tot.argmax())
                # common support: cells present under both the true offset and the match's choice
                both = pr[i_t] & pr[i_lik]
                common = (s[i_t][both].sum() - s[i_lik][both].sum()).item()
                p = one("p")[0, :LM.MAP].float()
                grid = p.reshape(7, 9, 16)
                texture = float(((grid[:, 1:] - grid[:, :-1]).abs().sum(-1).mean() + (grid[1:] - grid[:-1]).abs().sum(-1).mean()) / 2)
                rows.append(dict(moved=dtrue != [0, 0], dark=float(r["light"][t]) < 0.5, i_t=i_t, i_pri=i_pri, i_post=i_post,
                                 i_lik=i_lik, cmd=int(r["prev_action"][t]), lik_gap=float(tot[i_t] - tot[i_lik]),
                                 n_true=int(pr[i_t].sum()), n_lik=int(pr[i_lik].sum()), common_gap=common,
                                 prior_true=float(logK[i_t].exp()), texture=texture))
    out = {}
    for name, sel in (("moved", [x for x in rows if x["moved"]]), ("blocked", [x for x in rows if not x["moved"]])):
        lik_ok = np.mean([x["i_lik"] == x["i_t"] for x in sel])
        wrong = [x for x in sel if x["i_lik"] != x["i_t"]]
        out[name] = {
            "n": len(sel),
            "prior_ok": float(np.mean([x["i_pri"] == x["i_t"] for x in sel])),
            "posterior_ok": float(np.mean([x["i_post"] == x["i_t"] for x in sel])),
            "match_alone_ok": float(lik_ok),
            "match_wrong_choice_is_stay": float(np.mean([x["i_lik"] == 4 for x in wrong])) if wrong else None,
            "match_wrong_more_overlap": float(np.mean([x["n_lik"] > x["n_true"] for x in wrong])) if wrong else None,
            "match_wrong_overlap_true_vs_choice": [float(np.mean([x["n_true"] for x in wrong])), float(np.mean([x["n_lik"] for x in wrong]))] if wrong else None,
            "match_wrong_true_wins_on_common_cells": float(np.mean([x["common_gap"] > 0 for x in wrong])) if wrong else None,
            "texture_right_vs_wrong": [float(np.mean([x["texture"] for x in sel if x["i_lik"] == x["i_t"]])),
                                       float(np.mean([x["texture"] for x in wrong]))] if wrong else None,
            "dark_match_alone_ok": float(np.mean([x["i_lik"] == x["i_t"] for x in sel if x["dark"]])) if any(x["dark"] for x in sel) else None,
            "posterior_choice_counts": dict(Counter(x["i_post"] for x in sel)),
            "prior_mass_on_truth": float(np.mean([x["prior_true"] for x in sel])),
        }
    print(json.dumps(out, indent=1))
    (LM.HERE / "diagnosis_v1" / "localizer_diagnosis.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
