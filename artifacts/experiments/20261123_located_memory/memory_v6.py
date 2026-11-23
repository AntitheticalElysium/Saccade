"""The sixth localizer: memory.py with only the match's metric replaced (PLAN, amended 11-24 before its results).

The fifth's match failed across daylight changes: P1.5's coarse tokens of the same world cell change with the light
by more than the repeatability it was calibrated on (consecutive frames), so a remembered place was never recognised
once the light had moved (RESULTS.md, `diagnosis_v5/`). Here a token and an entry match when their change is small
under the within-place scatter: the change covariance of the same screen cells over the agent's own runs of
non-movement commands (TRAIN episodes, label-free), the scatter Fisher's discriminant discounts and Fisherfaces
project away against illumination (Belhumeur, Hespanha & Kriegman 1997). The distance is Mahalanobis under that
scatter; the threshold is its 90th percentile on those pairs, as the fifth's is on its own pairs, so P(match | same
content) = 0.9 by the same construction. Hits and the bank's new-place rate both use it. Everything else (the motion
prior, relocation, the common support, the agent's own cell, the store, the read, the readings) is memory.py's.

Kept as a separate module that patches memory.py in this process, so the fifth's read fine-tune and its readings,
which import memory.py, are untouched. Same commands as memory.py:

    LOC_CKPT=localizer_v6.pt LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261123_located_memory.memory_v6 localizer
"""
import importlib, sys

import torch

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")
_init, _calibrate = LM.Localizer.__init__, LM.calibrate


def __init__(self, n_bank=512):
    _init(self, n_bank)
    self.register_buffer("w_in", torch.eye(16))                 # inverse Cholesky factor of the within-place scatter
    self.register_buffer("m_thr", torch.zeros(()))              # a match: Mahalanobis distance below this


@torch.no_grad()
def still_pairs(data):
    """The change of each screen cell (the agent's own excluded) from the first frame of every run of non-movement
    commands to each later frame of the run [n, 16]."""
    keep = torch.arange(LM.MAP) != LM.SELF
    D = []
    for r in data:
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
    return torch.cat(D)


@torch.no_grad()
def calibrate(loc, data, seed=20261124):
    """The fifth's calibration (its bank is reused), then the within-place scatter and its threshold."""
    out = _calibrate(loc, data, seed)
    D = still_pairs(data)
    loc.w_in.copy_(torch.linalg.inv(torch.linalg.cholesky(torch.cov(D.T) + 1e-4 * torch.eye(16))))
    loc.m_thr.fill_(float((D @ loc.w_in.T).norm(dim=1).quantile(0.9)))
    out.update(still_pairs=int(len(D)), m_thr=float(loc.m_thr))
    return out


def null(self, c, now, prev=None):
    """The share of the bank's observations the cell's token matches under the within-place metric [B, 63]."""
    diff = c[:, :, None].float() - self.bank[:, :16][None, None]                  # [B, 63, n_bank, 16]
    hit = ((diff @ self.w_in.T).norm(dim=-1) < self.m_thr).float().mean(-1)
    return hit.clamp(1 / len(self.bank), 1 - 1 / len(self.bank))


def score(self, c, now, ent, present, null):
    """As the fifth's: log(0.9 / p_new) if the token matches the entry, log(0.1 / (1 - p_new)) if not, zero without an
    entry; the match is the within-place distance below m_thr."""
    (d, ls, tau), (e0, ls_e, t_e) = now, ent
    hit = ((c - (e0 + d)) @ self.w_in.T).norm(dim=-1) < self.m_thr
    q = torch.full_like(null, 0.9)
    ev = torch.where(hit, (q / null).log(), ((1 - q) / (1 - null)).log())
    return torch.where(present, ev, torch.zeros_like(ev))


LM.Localizer.__init__, LM.Localizer.null, LM.Localizer.score = __init__, null, score
LM.calibrate = calibrate

if __name__ == "__main__":
    a = sys.argv
    {"localizer": lambda: (LM.train_localizer(), LM.score_localizer(True, "full"), LM.score_localizer(False, "odometry")),
     "score": lambda: LM.score_localizer(True, "full"), "train": lambda: LM.train(),
     "readings": lambda: (LM.evaluate(int(a[2])), LM.identity(int(a[2])), LM.inject(int(a[2])), LM.dream(int(a[2])))}[a[1]]()
