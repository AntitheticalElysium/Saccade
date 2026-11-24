"""At the fourth localizer's natural relocalisations: which part of the true pose's memory score is negative, and which
null component explains those cells (simulator positions score only). For each present cell at the true pose: its
score, the null's dominant component (background / spatial prior / sensor-fixed) and memory's log density against
that component's; summed by dominant component.

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v4.jump_cells [n_eps]
"""
import importlib, json, sys
from collections import defaultdict

import torch
import torch.nn.functional as F

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v4.memory_v4")


@torch.no_grad()
def components(loc, c, now, prev):
    """The null's per-component log densities with their log weights, [B, 63, 3] (bg, spatial prior, sensor-fixed)."""
    d, s, tau = now; nb = LM.NB
    cp, vp, hp = prev; dp, sp, tp = loc.now_terms(vp)
    a1 = loc.age(LM.PR.age_features(torch.ones(1), torch.ones(1, dtype=torch.bool)))[0, 0]
    comps = torch.stack([loc.bg(c), LM.laplace(c, c[:, nb] - d[:, nb] + d, loc.nb[0] + s + s[:, nb]),
                         LM.laplace(c, cp.float() - dp + d, sp + a1 + s)], -1)
    has_nb = nb != LM.CELLS
    logits = torch.stack([torch.zeros_like(tau), torch.where(has_nb, loc.nb[1] + tau + tau[:, nb], torch.full_like(tau, -1e4)),
                          loc.nid[0] + tau + tp], -1)
    return comps, logits.log_softmax(-1)


@torch.no_grad()
def main(n_eps=3):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v4.pt", map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    tot = defaultdict(float); cnt = defaultdict(int); mem_minus = defaultdict(list); njump = 0
    for r in data:
        mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
        for t in range(r["T"] + 1):
            one = lambda k: r[k][t][None]
            first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
            before = mp.pose.clone(); ent = mp.view()[0]
            prev = (mp.last_p[:, :LM.MAP].clone(), mp.last_v.clone(), torch.tensor([True]))
            LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"), one("prev_action"),
                       tt, tt, first, valid, truth=tc[t][None])
            if not (t > 0 and (mp.pose[0] - before[0]).abs().max() > 1):
                continue
            njump += 1
            tp_ = before[0] + (pos[t] - pos[t - 1])
            c = one("p")[:, :LM.MAP].float(); v = one("v").float(); now = loc.now_terms(v)
            rr = tp_[0] + LM.KR; cc = tp_[1] + LM.KC
            e = ent[0, rr.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)][None]
            age = tt[:, None] - e[..., LM.CTIME]; present = (e[..., LM.CTIME] >= 0)
            comps, lw = components(loc, c, now, prev)
            null = (comps + lw).logsumexp(-1)
            e0, s_e, t_e = loc.entry_terms(e, age, present)
            Lm = LM.laplace(c, e0 + now[0], s_e + now[1])
            sc = loc.score(c, now, (e0, s_e, t_e), present, null)
            dom = (comps + lw).argmax(-1)
            for k in range(LM.MAP):
                if present[0, k]:
                    name = ("background", "spatial_prior", "sensor_fixed")[int(dom[0, k])]
                    tot[name] += float(sc[0, k]); cnt[name] += 1
                    mem_minus[name].append(float(Lm[0, k] - comps[0, k, dom[0, k]]))
    med = lambda x: sorted(x)[len(x) // 2] if x else None
    out = {"jumps": njump, "true_pose_score_by_dominant_null": {k: {"cells": cnt[k], "score_sum_per_jump": round(tot[k] / max(njump, 1), 2),
                                                                     "memory_minus_component_median": round(med(mem_minus[k]), 1)} for k in cnt},
           "log_scales": {"spatial_prior_offset": float(loc.nb[0]), "sensor_fixed_weight": float(loc.nid[0]),
                          "memory_age_parts_1_4_16_64_256": loc.age(LM.PR.age_features(torch.tensor([1., 4., 16., 64., 256.]), torch.ones(5, dtype=torch.bool)))[:, 0].tolist()}}
    txt = json.dumps(out, indent=1); print(txt)
    (LM.HERE / "diagnosis_v4" / f"jump_cells_localizer_v4_{n_eps}eps.json").write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
