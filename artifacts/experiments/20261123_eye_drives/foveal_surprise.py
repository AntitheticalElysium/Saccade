"""Second reading of PLAN.md: the world's own precision-weighted foveal surprise (TSP NLL) per window, day and night.

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_eye_drives.foveal_surprise
"""
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import gates as G, s1 as S

HERE = Path(__file__).resolve().parent
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
AUD = importlib.import_module("artifacts.experiments.20261119_s2.dream_outcome_audit")
ED = importlib.import_module("artifacts.experiments.20261123_eye_drives.eye_drives")
DEV = S2.DEV
CENT = C.CENTRES


@torch.no_grad()
def window_scores(model, cfg, frames, en, ts):
    """NLL and MSE of the TSP prediction against the real crop at all 49 windows: [n, 49] each."""
    n = len(ts); t = torch.as_tensor(ts)
    bp, p, v = (en[k][t].to(DEV).float() for k in ("bp", "p", "v"))
    fr = frames[t].to(DEV)
    nll, mse = [], []
    for w in range(49):
        fix = CENT[w].to(DEV).expand(n, 2)
        cells = M.window(fix)
        pix = M.cell_pixels(M.crop(fr, fix))
        with TR.autocast(DEV, cfg):
            mean, log_s = model.pixels(model.tsp.predict(bp, p, v, cells), v, cells)
        nll.append(M.pixel_nll(mean.float(), log_s.float(), pix).mean((-1, -2)))
        mse.append((mean.float() - (pix.float() / 127.5 - 1)).square().mean((-1, -2)))
    return torch.stack(nll, 1).cpu().numpy(), torch.stack(mse, 1).cpu().numpy()


def main():
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    enc, ids = AUD.encode_sample(model, cfg, "dev", 160, ED.SEED + 1)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    sub = S2.sublabels(L, np.asarray(ids))
    eps_, grids = S.episodes(sub), G.episode_grids(sub)
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    out = {k: [] for k in ("episode", "nll", "mse", "night", "ent", "creature")}
    for i, e in enumerate(ids):
        T = len(enc[i]["reward"]) - 1
        ts = np.arange(ED.FIRST, T + 1)
        if not len(ts):
            continue
        frames = torch.as_tensor(D.gather([eps_[i]], [[(0, 1, t, 4, 4) for t in range(T + 1)]])["frames"][0])
        nll, mse = window_scores(model, cfg, frames, enc[i], ts)
        gr = grids[i]["grid"][ts].reshape(len(ts), 63) > 0
        occ = np.concatenate([gr, np.zeros((len(ts), 18), bool)], 1)
        for k, x in (("episode", np.full(len(ts), i)), ("nll", nll), ("mse", mse),
                     ("night", L["s_light"][so[e] + ts] < 0.5), ("ent", occ[:, ED.WIN].any(-1)), ("creature", occ.any(1))):
            out[k].append(x)
        if i % 40 == 0:
            print("episode", i, flush=True)
    z = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(HERE / "foveal_surprise.npz", **z)
    hud = np.broadcast_to(ED.HUD_WIN, z["ent"].shape)
    cats = {"entity": z["ent"], "hud": hud, "static_map": ~z["ent"] & ~hud}
    rng = np.random.default_rng(0)
    ue = np.unique(z["episode"]); idx = {k: np.flatnonzero(z["episode"] == k) for k in ue}

    def stat(fn, mask):
        point = float(fn(np.flatnonzero(mask))); bs = []
        for _ in range(500):
            ii = np.concatenate([idx[k] for k in rng.choice(ue, len(ue))]); ii = ii[mask[ii]]
            if len(ii):
                bs.append(fn(ii))
        return {"mean": point, "ci95": np.quantile(bs, [0.025, 0.975]).tolist(), "frames": int(mask.sum())}

    res = {"frames": int(len(z["episode"]))}
    for period, m in (("day", ~z["night"]), ("night", z["night"])):
        res[period] = {}
        for sig in ("nll", "mse"):
            x = z[sig] - z[sig].min(1, keepdims=True)                       # non-negative per frame, so shares exist
            x = x / np.clip(x.sum(1, keepdims=True), 1e-12, None)
            res[period][sig] = {}
            for name, cm in cats.items():
                lift = lambda ii, x=x, cm=cm: ((x[ii] * cm[ii]).sum(1) / np.clip(cm[ii].mean(1), 1e-12, None)).mean()
                res[period][sig][name] = stat(lift, m & cm.any(1))
            arg = z[sig].argmax(1)
            res[period][sig]["argmax_on_entity_when_present"] = stat(
                lambda ii, arg=arg: z["ent"][ii, arg[ii]].mean(), m & z["creature"])
        res[period]["uniform_entity_share_when_present"] = stat(lambda ii: z["ent"][ii].mean(1).mean(), m & z["creature"])
    (HERE / "foveal_surprise.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
