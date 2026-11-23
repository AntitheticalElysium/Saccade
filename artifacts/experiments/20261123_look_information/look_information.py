"""What does a look add to the agent's knowledge? PLAN.md (2026-11-23).

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_look_information.look_information
"""
import argparse
import dataclasses
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
DV = importlib.import_module("artifacts.experiments.20261123_gaze_decision_value.decision_value")
DEV = S2.DEV
SEED = 20261122
FIRST = 8
WIN = C.WIN.numpy()
CENT = C.CENTRES.numpy()
HUD_WIN = CENT[:, 0] >= 6


def rep_carry(carry, n):
    return tuple(type(c)(*(getattr(c, f.name).repeat_interleave(n, 0) for f in dataclasses.fields(c))) for c in carry)


@torch.no_grad()
def one_start(model, cfg, en, frames, frames_b, t, windows, shifted):
    """Belief change and next-frame TSP NLL for real vs periphery-acuity crops of frame t at each window."""
    acu = model.perception.acuity
    k = len(windows)
    with TR.autocast(DEV, cfg):
        pp = M.Percept(*(en[key][:t].to(DEV).float()[None] for key in ("p", "g", "v")), en["fix"][:t].to(DEV)[None])
        x = model.belief.tokens(S.embed(model, pp, en["reads"][:t].to(DEV).float()[None]),
                                en["reward"][:t].to(DEV)[None], en["terminated"][:t].to(DEV)[None],
                                en["action"][:t].to(DEV)[None])
        _, carry = model.belief(x, model.belief.initial(1, DEV))
        fix = torch.as_tensor(np.repeat(CENT[windows], 2, 0), device=DEV)               # [2k, 2]: real, blur per window
        src = torch.stack([frames[t], frames_b[t]] * k)                                 # matches the repeat order
        g = DV.glimpses(model, cfg, src, fix.cpu()).float()
        P = M.Percept(en["p"][t].to(DEV).float().expand(2 * k, -1, -1)[:, None], g[:, None],
                      en["v"][t].to(DEV).float().expand(2 * k, -1)[:, None], fix[:, None])
        emb = S.embed(model, P, en["reads"][t].to(DEV).float().expand(2 * k, -1)[:, None])
        tok = model.belief.tokens(emb, en["reward"][t:t + 1].to(DEV).expand(2 * k)[:, None],
                                  en["terminated"][t:t + 1].to(DEV).expand(2 * k)[:, None],
                                  en["action"][t:t + 1].to(DEV).expand(2 * k)[:, None])
        y, _ = model.belief(tok, rep_carry(carry, 2 * k))
        b, bp1 = y[:, 0].float(), y[:, 1].float()
        nfix = torch.as_tensor(np.repeat(CENT[shifted], 2, 0), device=DEV)
        cells = M.window(nfix)
        pix = M.cell_pixels(M.crop(frames[t + 1].to(DEV).expand(2 * k, -1, -1, -1), nfix))
        mean, log_s = model.pixels(model.tsp.predict(bp1, en["p"][t + 1].to(DEV).float().expand(2 * k, -1, -1),
                                                     en["v"][t + 1].to(DEV).float().expand(2 * k, -1), cells),
                                   en["v"][t + 1].to(DEV).float().expand(2 * k, -1), cells)
        nll_next = M.pixel_nll(mean.float(), log_s.float(), pix).mean((-1, -2)).view(k, 2)
        cells0 = M.window(fix[0::2])
        pix0 = M.cell_pixels(M.crop(frames[t].to(DEV).expand(k, -1, -1, -1), fix[0::2]))
        m0, s0 = model.pixels(model.tsp.predict(en["bp"][t].to(DEV).float().expand(k, -1),
                                                en["p"][t].to(DEV).float().expand(k, -1, -1),
                                                en["v"][t].to(DEV).float().expand(k, -1), cells0),
                              en["v"][t].to(DEV).float().expand(k, -1), cells0)
        nll_now = M.pixel_nll(m0.float(), s0.float(), pix0).mean((-1, -2))
    bv = b.view(k, 2, -1)
    change = (bv[:, 0] - bv[:, 1]).norm(dim=-1)
    rel = change / bv[:, 0].norm(dim=-1).clamp_min(1e-6)
    gain = nll_next[:, 1] - nll_next[:, 0]                                              # blur minus real: > 0 helps
    return gain.cpu().numpy(), change.cpu().numpy(), rel.cpu().numpy(), nll_now.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    enc, ids = AUD.encode_sample(model, cfg, "dev", 6 if args.smoke else 160, SEED + 1)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    sub = S2.sublabels(L, np.asarray(ids))
    eps_, grids = S.episodes(sub), G.episode_grids(sub)
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rng = np.random.default_rng(SEED + 31)
    acu = model.perception.acuity
    rows = {k: [] for k in ("episode", "night", "cat", "gain", "change", "rel", "nll_now")}
    for i, e in enumerate(ids):
        T = len(enc[i]["reward"]) - 1
        ts = np.arange(FIRST, T)
        if not len(ts):
            continue
        night = L["s_light"][so[e] + ts] < 0.5
        pick = []
        for m in (night, ~night):
            c = ts[m]
            pick += list(rng.choice(c, min(len(c), 2 if args.smoke else 6), replace=False)) if len(c) else []
        if not pick:
            continue
        frames = DV.frames_of(eps_[i])
        frames_b = DV.blurred(frames, acu)
        pos = L["s_pos"][so[e]:so[e + 1]].astype(int)
        for t in sorted(pick):
            occ = np.concatenate([grids[i]["grid"][t].reshape(63) > 0, np.zeros(18, bool)])
            cover = occ[WIN].any(-1)
            cand = {"creature": np.flatnonzero(cover & ~HUD_WIN), "static": np.flatnonzero(~cover & ~HUD_WIN),
                    "hud": np.flatnonzero(HUD_WIN)}
            wins, cats = [], []
            for cat, n in (("creature", 1), ("static", 2), ("hud", 1)):
                if len(cand[cat]):
                    for w in rng.choice(cand[cat], min(n, len(cand[cat])), replace=False):
                        wins.append(int(w)); cats.append(cat)
            d = pos[t + 1] - pos[t]
            shifted, keep = [], []
            for w, cat in zip(wins, cats):
                r, c = CENT[w]
                if cat != "hud":
                    r, c = r - d[0], c - d[1]
                ok = 1 <= r <= 7 and 1 <= c <= 7 and (cat == "hud" or r <= 5)
                keep.append(ok)
                shifted.append(int(np.flatnonzero((CENT[:, 0] == r) & (CENT[:, 1] == c))[0]) if ok else 0)
            wins = [w for w, k_ in zip(wins, keep) if k_]; cats = [c for c, k_ in zip(cats, keep) if k_]
            shifted = [s for s, k_ in zip(shifted, keep) if k_]
            if not wins:
                continue
            gain, change, rel, nll_now = one_start(model, cfg, enc[i], frames, frames_b, int(t), wins, shifted)
            for j, cat in enumerate(cats):
                for key, val in (("episode", i), ("night", bool(L["s_light"][so[e] + t] < 0.5)), ("cat", cat),
                                 ("gain", gain[j]), ("change", change[j]), ("rel", rel[j]), ("nll_now", nll_now[j])):
                    rows[key].append(val)
        if i % 20 == 0:
            print("episode", i, "rows", len(rows["cat"]), flush=True)
    z = {k: np.asarray(v) for k, v in rows.items()}
    np.savez_compressed(HERE / ("rows_smoke.npz" if args.smoke else "rows.npz"), **z)
    res = summarise(z)
    (HERE / ("summary_smoke.json" if args.smoke else "summary.json")).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


def summarise(z, reps=500):
    rng = np.random.default_rng(0)
    eps = np.unique(z["episode"]); idx = {k: np.flatnonzero(z["episode"] == k) for k in eps}

    def boot(v, m):
        point = float(v[m].mean()) if m.any() else None; bs = []
        for _ in range(reps):
            ii = np.concatenate([idx[k] for k in rng.choice(eps, len(eps))]); ii = ii[m[ii]]
            if len(ii):
                bs.append(v[ii].mean())
        return {"mean": point, "ci95": np.quantile(bs, [0.025, 0.975]).tolist() if bs else None, "n": int(m.sum())}

    res = {}
    for period, pm in (("day", ~z["night"]), ("night", z["night"])):
        res[period] = {cat: {k: boot(z[k], pm & (z["cat"] == cat)) for k in ("gain", "change", "rel", "nll_now")}
                       for cat in ("creature", "static", "hud")}
    return res


if __name__ == "__main__":
    main()
