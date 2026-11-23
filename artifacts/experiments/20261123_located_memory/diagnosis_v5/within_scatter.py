"""Can the agent's own still periods teach a match that survives lighting changes? Label-free: on TRAIN episodes,
every run of non-movement commands gives pairs of the same screen cell (the first frame of the run against each later
one), so the same world cell under whatever changed meanwhile, lighting included. Their change covariance is the
within-place scatter (Fisher's within-class scatter; Belhumeur, Hespanha & Kriegman 1997 use it to discount
illumination). On the diagnosis episodes' daylight sightings (simulator positions pair them; scoring only), the
current match (L1 over the calibrated pair scale, threshold thr) and a Mahalanobis distance under that scatter
(threshold: its 90th percentile on the still pairs, as thr is for the current one) are compared on the same cell's
previous sighting against a random other cell of that earlier frame, by the change in daylight. Also: how much of a
same-cell change a linear map of the lighting latent's change removes (held-out half).

    LOC_CKPT=localizer_v5.pt LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261123_located_memory.diagnosis_v5.within_scatter [n_eps] [first_ep]
"""
import importlib, json, sys

import numpy as np
import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")


def auc(a, b):
    """P(a same-cell distance is below a different-cell one)."""
    a, b = np.asarray(a), np.asarray(b); rk = np.concatenate([a, b]).argsort().argsort() + 1
    return float((rk[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b)))


@torch.no_grad()
def main(n_eps=23, first=0):
    torch.manual_seed(0)
    keep = torch.arange(LM.MAP) != LM.SELF
    D = []
    for r in torch.load(LM.DATA / "tokens_train.pt"):
        a = r["prev_action"].numpy(); p = r["p"][:, :LM.MAP].float(); T = r["T"]
        t = 1
        while t <= T:
            if int(a[t]) in LM.GT.MOVE:
                t += 1; continue
            s, e = t - 1, t
            while e + 1 <= T and int(a[e + 1]) not in LM.GT.MOVE:
                e += 1
            D += [(p[u] - p[s])[keep] for u in range(s + 1, e + 1)]
            t = e + 1
    D = torch.cat(D)
    Lw = torch.linalg.cholesky(torch.cov(D.T) + 1e-4 * torch.eye(16))
    maha = lambda x: torch.linalg.solve_triangular(Lw, x.T, upper=False).T.norm(dim=1)
    mthr = float(maha(D).quantile(0.9))
    loc = LM.Localizer(); loc.load_state_dict(torch.load(LM.RUN / LM.LOC_CKPT, map_location="cpu")); loc.eval()
    cells = torch.arange(LM.MAP)
    same, diff, ssc, dsc, dls, dvs = [], [], [], [], [], []
    for r in torch.load(LM.DATA / "tokens_panel.pt")[first:first + n_eps]:
        T, pos = r["T"], r["pos"]; p = r["p"][:, :LM.MAP].float(); light = r["light"].numpy(); v = r["v"].float()
        _, ls = loc.calib(cells.expand(T + 1, LM.MAP), v[:, None].expand(T + 1, LM.MAP, 8))[:2]
        last = {}
        for t in range(T + 1):
            for k in range(LM.MAP):
                if k == LM.SELF:
                    continue
                w = (int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k]))
                if w in last and light[t] >= 0.5 and light[last[w][0]] >= 0.5:
                    t0, k0 = last[w]
                    k1 = int(torch.randint(LM.MAP, (1,)))
                    if k1 != LM.SELF and (int(pos[t0][0]) + int(LM.KR[k1]), int(pos[t0][1]) + int(LM.KC[k1])) != w:
                        same.append(p[t, k] - p[t0, k0]); diff.append(p[t, k] - p[t0, k1])
                        ssc.append(float(torch.logaddexp(ls[t, k], ls[t0, k0]).exp())); dsc.append(float(torch.logaddexp(ls[t, k], ls[t0, k1]).exp()))
                        dls.append(abs(light[t] - light[t0])); dvs.append(v[t] - v[t0])
                last[w] = (t, k)
    Ds, Dd, ss, sd, dls, dV = torch.stack(same), torch.stack(diff), torch.tensor(ssc), torch.tensor(dsc), np.array(dls), torch.stack(dvs)
    out = {"episodes": [first, first + n_eps], "still_pairs": int(len(D)), "thr_current": float(loc.thr), "thr_within_scatter": mthr}
    for lab, sel in (("dl<0.02", dls < 0.02), ("0.02<=dl<0.1", (dls >= 0.02) & (dls < 0.1)), ("dl>=0.1", dls >= 0.1)):
        s = torch.as_tensor(sel)
        cs, cd = Ds[s].abs().sum(1) / ss[s], Dd[s].abs().sum(1) / sd[s]
        ms, md = maha(Ds[s]), maha(Dd[s])
        out[lab] = {"n": int(s.sum()),
                    "current": {"auc": auc(cs, cd), "match_same": float((cs < loc.thr).float().mean()), "match_other": float((cd < loc.thr).float().mean())},
                    "within_scatter": {"auc": auc(ms, md), "match_same": float((ms < mthr).float().mean()), "match_other": float((md < mthr).float().mean())}}
    big = torch.as_tensor(dls >= 0.05)                                   # a linear lighting response, held-out half
    X = torch.cat([dV[big], torch.ones(int(big.sum()), 1)], 1); Y = Ds[big]
    idx = torch.randperm(len(X)); a_, b_ = idx[: len(X) // 2], idx[len(X) // 2:]
    W = torch.linalg.lstsq(X[a_], Y[a_]).solution
    out["linear_lighting_response_dl>=0.05"] = {"n": int(big.sum()), "change_l1_median": float(Y.abs().sum(1).median()),
                                                "residual_l1_median_heldout": float((Y[b_] - X[b_] @ W).abs().sum(1).median())}
    (LM.HERE / "diagnosis_v5" / f"within_scatter_{first}_{first + n_eps}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
