"""Development check, not a localizer version: the fifth localizer (its trained motion prior and relocation) with only
the match's metric swapped, in this process, for the within-place scatter's Mahalanobis distance
(`within_scatter.py`: the change covariance of the same screen cells over the agent's own non-movement runs, TRAIN
episodes, label-free; threshold at its 90th percentile on those pairs, as the fifth's is on its own). Hits and the
bank's new-place rate both use it; everything else is the fifth's. Then `chain.py`'s diagnosis on the same episodes,
tagged "_scatter". memory.py itself is not changed.

    LOC_CKPT=localizer_v5.pt LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261123_located_memory.diagnosis_v5.metric_chain [n_eps] [first_ep]
"""
import importlib, json, sys

import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")
CH = importlib.import_module("artifacts.experiments.20261123_located_memory.diagnosis_v5.chain")


@torch.no_grad()
def scatter():
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
    Winv = torch.linalg.inv(torch.linalg.cholesky(torch.cov(D.T) + 1e-4 * torch.eye(16)))
    return Winv, float((D @ Winv.T).norm(dim=1).quantile(0.9))


def patch(Winv, mthr):
    W = Winv.T.contiguous()

    def null(self, c, now, prev=None):
        diff = c[:, :, None].float() - self.bank[:, :16][None, None]
        hit = ((diff @ W.to(diff.device)).norm(dim=-1) < mthr).float().mean(-1)
        return hit.clamp(1 / len(self.bank), 1 - 1 / len(self.bank))

    def score(self, c, now, ent, present, null):
        (d, ls, tau), (e0, ls_e, t_e) = now, ent
        hit = ((c - (e0 + d)) @ W.to(c.device)).norm(dim=-1) < mthr
        q = torch.full_like(null, 0.9)
        ev = torch.where(hit, (q / null).log(), ((1 - q) / (1 - null)).log())
        return torch.where(present, ev, torch.zeros_like(ev))

    LM.Localizer.null, LM.Localizer.score = null, score


if __name__ == "__main__":
    Winv, mthr = scatter()
    print("within-place threshold", mthr, flush=True)
    patch(Winv, mthr)
    a = [int(x) for x in sys.argv[1:]]
    CH.main(*a, tag="_scatter")
