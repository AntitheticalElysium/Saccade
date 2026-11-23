"""Is the stay bonus a scale mismatch? On the panel's first episodes (simulator positions score only): the L1 residual
(16 dims) between a coarse token and the same screen cell's token in the previous frame, for static frames (no movement
command) and moved frames, day / dark, centre cell vs the rest; and at the centre cell on moved frames, each null
component's log-likelihood vs memory's under 'stay' (log-scales from the trained third localizer).

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.diagnosis_v3.residuals [n_eps]
"""
import importlib, json, sys
from collections import defaultdict

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v3.memory_v3")


@torch.no_grad()
def main(n_eps=3):
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / "localizer_v3.pt", map_location="cpu")); loc.eval()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    res = defaultdict(list); comp = defaultdict(list)
    age1 = loc.age(LM.PR.age_features(torch.tensor([1.]), torch.ones(1, dtype=torch.bool)))[0]      # (scale, trust) parts at age 1
    for r in data:
        p = r["p"][:, :LM.MAP].float(); v = r["v"].float(); pos = r["pos"]
        for t in range(1, r["T"] + 1):
            a = int(r["prev_action"][t]); d = (pos[t] - pos[t - 1]).tolist()
            kind = "static" if a not in LM.GT.MOVE else ("moved" if d != [0, 0] else "blocked")
            dark = "dark" if float(r["light"][t]) < 0.5 else "day"
            R = (p[t] - p[t - 1]).abs().sum(-1)                                   # [63]
            res[(kind, dark, "centre")].append(float(R[31])); res[(kind, dark, "rest")] += R[torch.arange(63) != 31].tolist()
            if kind == "moved":
                c = p[t][None]; now = loc.now_terms(v[t][None]); prv = loc.now_terms(v[t - 1][None])
                dn, sn, tn = now; dp, sp, tp = prv
                k = 31
                mean_id = p[t - 1][None, k] - dp[:, k] + dn[:, k]
                L_id = LM.laplace(c[:, k], mean_id, loc.nid[0] + sn[:, k] + sp[:, k])
                L_mem_stay = LM.laplace(c[:, k], mean_id, sp[:, k] + age1[0] + sn[:, k])     # same mean, the entry's age-1 scale
                L_bg = loc.bg(c)[:, k]
                nb = LM.NB[k]
                L_nb = LM.laplace(c[:, k], c[:, nb] - dn[:, nb] + dn[:, k], loc.nb[0] + sn[:, k] + sn[:, nb])
                for name, x in (("identity", L_id), ("memory_stay_age1", L_mem_stay), ("background", L_bg), ("neighbour", L_nb)):
                    comp[(dark, name)].append(float(x))
                comp[(dark, "log_b_identity")].append(float(loc.nid[0] + sn[0, k] + sp[0, k]))
                comp[(dark, "log_b_memory_age1")].append(float(sp[0, k] + age1[0] + sn[0, k]))
    q = lambda x: {"n": len(x), "median": round(float(np.median(x)), 4), "p10": round(float(np.percentile(x, 10)), 4),
                   "p90": round(float(np.percentile(x, 90)), 4), "zero_share": round(float(np.mean(np.array(x) < 1e-6)), 3)}
    out = {"residual_L1": {"|".join(k): q(v) for k, v in sorted(res.items())},
           "centre_moved_loglik": {"|".join(k): q(v) for k, v in sorted(comp.items())}}
    txt = json.dumps(out, indent=1); print(txt)
    (LM.HERE / "diagnosis_v3" / f"residuals_localizer_v3_{n_eps}eps.json").write_text(txt + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
