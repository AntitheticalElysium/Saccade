"""Where do the chooser's surprise and contingency send the eye, by day and by night? PLAN.md (2026-11-23).

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_eye_drives.eye_drives
"""
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from torch.nn import functional as F

from saccade import chooser as C, model as M
from saccade.eval import gates as G, s1 as S

HERE = Path(__file__).resolve().parent
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
AUD = importlib.import_module("artifacts.experiments.20261119_s2.dream_outcome_audit")
DEV = S2.DEV
SEED = 20261122
FIRST = 8
WIN = C.WIN.numpy()
HUD_WIN = (C.CENTRES[:, 0] >= 6).numpy()


@torch.no_grad()
def drives(head, pol, en, ts):
    """The chooser's contingency c [n, 49], surprise s [n, 49] and P(window) [n, 49] on real frames (s2b.choose)."""
    H, ts = pol["history"], torch.as_tensor(ts)
    idx = ts[:, None] + torch.arange(-H, 0)[None]
    ok = (idx >= 0).float().to(DEV)
    idx = idx.clamp(min=0)
    get = lambda k: en[k][idx].to(DEV)
    ctx = torch.cat([get("v").float() * ok[..., None],
                     F.one_hot((get("prev_action") + 1).clamp(0, 17), 18).float() * ok[..., None],
                     (M.SYMLOG(get("reward").float()) * ok)[..., None], (get("terminated").float() * ok)[..., None],
                     ok[..., None]], -1).flatten(1)
    ph = get("p").float() * ok[..., None, None]
    pa = en["prev_action"][ts].to(DEV)
    win = torch.as_tensor(WIN, device=DEV)
    taken, noop = head((ph, ctx, pa)), head((ph, ctx, torch.zeros_like(pa)))
    c_ = (taken - noop).square().sum(-1)[:, win].sum(-1)
    s_ = (en["p"][ts].to(DEV).float() - taken).square().sum(-1)[:, win].sum(-1)
    return c_.cpu().numpy(), s_.cpu().numpy(), C.policy(c_, s_, pol).cpu().numpy()


def main():
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    pol = dict(cfg["fixation"]["chooser"])
    head = C.Head(pol["history"]).to(DEV)
    head.load_state_dict(torch.load(S2.P15, weights_only=False, map_location="cpu")["fixation_controller"]["head"])
    head.eval().requires_grad_(False)
    enc, ids = AUD.encode_sample(model, cfg, "dev", 160, SEED + 1)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    sub = S2.sublabels(L, np.asarray(ids))
    grids = G.episode_grids(sub)
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    out = {k: [] for k in ("episode", "c", "s", "p", "night", "ent", "creature")}
    for i, e in enumerate(ids):
        T = len(enc[i]["reward"]) - 1
        ts = np.arange(FIRST, T + 1)
        if not len(ts):
            continue
        c_, s_, p_ = drives(head, pol, enc[i], ts)
        gr = grids[i]["grid"][ts].reshape(len(ts), 63) > 0
        occ = np.concatenate([gr, np.zeros((len(ts), 18), bool)], 1)
        for k, v in (("episode", np.full(len(ts), i)), ("c", c_), ("s", s_), ("p", p_),
                     ("night", L["s_light"][so[e] + ts] < 0.5), ("ent", occ[:, WIN].any(-1)), ("creature", occ.any(1))):
            out[k].append(v)
    z = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(HERE / "drives.npz", **z)
    cats = {"entity": z["ent"], "hud": np.broadcast_to(HUD_WIN, z["ent"].shape),
            "static_map": ~z["ent"] & ~np.broadcast_to(HUD_WIN, z["ent"].shape)}
    rng = np.random.default_rng(0)
    eps = np.unique(z["episode"])
    idx = {k: np.flatnonzero(z["episode"] == k) for k in eps}

    def stat(fn, mask):
        point = float(fn(np.flatnonzero(mask)))
        bs = []
        for _ in range(500):
            ii = np.concatenate([idx[k] for k in rng.choice(eps, len(eps))])
            ii = ii[mask[ii]]
            if len(ii):
                bs.append(fn(ii))
        return {"mean": point, "ci95": np.quantile(bs, [0.025, 0.975]).tolist(), "frames": int(mask.sum())}

    res = {"frames": int(len(z["episode"])), "episodes": int(len(eps))}
    for period, m in (("day", ~z["night"]), ("night", z["night"])):
        res[period] = {}
        for drive in ("s", "c", "p"):
            x = z[drive] / np.clip(z[drive].sum(1, keepdims=True), 1e-12, None)
            res[period][drive] = {}
            for name, cm in cats.items():
                share = lambda ii, x=x, cm=cm: (x[ii] * cm[ii]).sum(1).mean()
                lift = lambda ii, x=x, cm=cm: ((x[ii] * cm[ii]).sum(1) / np.clip(cm[ii].mean(1), 1e-12, None)).mean()
                ok = m & (cm.any(1))
                res[period][drive][name] = {"share": stat(share, ok), "lift": stat(lift, ok)}
        pe = lambda ii: (z["p"][ii] * z["ent"][ii]).sum(1).mean()
        ue = lambda ii: z["ent"][ii].mean(1).mean()
        res[period]["p_on_entity_when_present"] = {"chooser": stat(pe, m & z["creature"]), "uniform": stat(ue, m & z["creature"])}
    (HERE / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
