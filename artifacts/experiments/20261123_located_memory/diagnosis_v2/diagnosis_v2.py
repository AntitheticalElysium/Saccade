"""Why the amended localizer (run/localizer.pt, 11-24) still misses: on the panel's movement steps (simulator positions
score only), day and dark: prior, posterior and the match alone; when the match is wrong, how often it says 'stay';
the per-cell advantage of 'stay' over the true move (7 x 9); the learned per-position trust and scale for a day and a
dark view; and the posterior over the first episodes (the mini check's).

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v2.diagnosis_v2 [n_eps] [ckpt]
"""
import importlib, json, sys
from collections import defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v2.memory_v2")   # the v2 code (767fe181)
OFFS = torch.stack([LM.NR, LM.NC], -1)


@torch.no_grad()
def cell_scores(loc, ent, c, v, t, hyp):
    """Per-cell scores [H, 63] and presence at hypotheses hyp [1, H, 2] (loglik_at before its sum)."""
    r = hyp[..., 0, None] + LM.KR; cc = hyp[..., 1, None] + LM.KC
    inside = (r >= 0) & (r < LM.G) & (cc >= 0) & (cc < LM.G)
    e = ent[torch.zeros_like(r), r.clamp(0, LM.G - 1), cc.clamp(0, LM.G - 1)]
    age = t[:, None, None] - e[..., LM.CTIME]
    present = inside & (e[..., LM.CTIME] >= 0)
    now = loc.now_terms(v); null = loc.null(c, now)
    s = loc.score(c[:, None], tuple(x[:, None] for x in now), loc.entry_terms(e, age, present), present, null[:, None])
    return s[0], present[0]


@torch.no_grad()
def main(n_eps=12, ckpt="localizer_v2.pt"):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / ckpt, map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    acc = defaultdict(list); adv = {k: torch.zeros(63) for k in ("day", "dark")}; cnt = {k: torch.zeros(63) for k in ("day", "dark")}
    per_ep = []
    for ei, r in enumerate(data):
        mp = LM.Map(1, torch.device("cpu")); pos = r["pos"]; tc = LM.truth_cells(pos)
        ok_ep = []
        for t in range(r["T"] + 1):
            one = lambda k: r[k][t][None]
            first = torch.tensor([t == 0]); valid = torch.tensor([True]); tt = torch.tensor([t])
            before = mp.pose.clone(); ent = mp.view()[0]
            mv = t > 0 and int(r["prev_action"][t]) in LM.GT.MOVE
            if mv:
                d = (pos[t] - pos[t - 1]).tolist()
                hyp = (before[:, None] + OFFS[None]).clamp(0, LM.G - 1)
                s, pr = cell_scores(loc, ent, one("p")[:, :LM.MAP].float(), one("v").float(), tt.float(), hyp)
                it = [i for i in range(9) if OFFS[i].tolist() == d][0]
                logK = loc.kernel(mp.last_p, mp.last_v, one("prev_action"))[0]
            feats, xhat, chat = LM.advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"),
                                           one("prev_action"), tt, tt, first, valid, truth=tc[t][None])
            if mv:
                dark = float(r["light"][t]) < 0.5
                key = ("moved" if d != [0, 0] else "blocked") + ("_dark" if dark else "_day")
                tot = s.sum(1); ch = int(tot.argmax())
                acc[(key, "match_alone")].append(ch == it)
                acc[(key, "prior")].append(int(logK.argmax()) == it)
                acc[(key, "posterior")].append((mp.pose[0] - before[0]).tolist() == d)
                if ch != it:
                    acc[(key, "wrong_match_is_stay")].append(ch == 4)
                ok_ep.append((mp.pose[0] - before[0]).tolist() == d)
                if d != [0, 0]:
                    both = pr[4] & pr[it]; k_ = "dark" if dark else "day"
                    adv[k_] += torch.where(both, s[4] - s[it], 0.); cnt[k_] += both.float()
        per_ep.append(float(np.mean(ok_ep)) if ok_ep else None)
    out = {"ckpt": ckpt, "episodes": n_eps, "posterior_by_episode": per_ep,
           "rates": {f"{k}|{m}": [len(v), round(float(np.mean(v)), 3)] for (k, m), v in sorted(acc.items())},
           "stay_minus_true_per_cell": {k: (adv[k] / cnt[k].clamp(min=1)).reshape(7, 9).round(decimals=1).tolist() for k in adv}}
    # learned per-position terms for one day and one dark view
    allv = torch.cat([r["v"].float() for r in data]); alll = torch.cat([r["light"].float() for r in data])
    for name, sel in (("day", alll > 0.9), ("dark", alll < 0.2)):
        if sel.any():
            v = allv[sel][:1]
            d, s_, tau = loc.now_terms(v)
            out[f"{name}_log_scale_part"] = s_[0].reshape(7, 9).round(decimals=2).tolist()
            out[f"{name}_trust_part"] = tau[0].reshape(7, 9).round(decimals=2).tolist()
            out[f"{name}_offset_norm"] = d[0].abs().sum(-1).reshape(7, 9).round(decimals=2).tolist()
    a = loc.age(LM.PR.age_features(torch.tensor([1., 4., 16., 64., 256.]), torch.ones(5, dtype=torch.bool)))
    out["age_parts"] = a.round(decimals=2).tolist(); out["nb"] = loc.nb.tolist(); out["b_bg"] = float(loc.log_b_bg.exp())
    out["eps"] = float(torch.sigmoid(loc.logit_eps))
    txt = json.dumps(out, indent=1)
    print(txt)
    (LM.HERE / "diagnosis_v2" / f"diagnosis_v2_{ckpt.replace('.pt', '')}_{n_eps}.json").write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12, sys.argv[2] if len(sys.argv) > 2 else "localizer_v2.pt")
