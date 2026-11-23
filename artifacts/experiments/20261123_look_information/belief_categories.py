"""Third reading of PLAN.md: belief change (fine minus periphery-acuity crop) on C8's failure classes.

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_look_information.belief_categories
"""
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import gates as G, s1 as S

HERE = Path(__file__).resolve().parent
LI = importlib.import_module("artifacts.experiments.20261123_look_information.look_information")
S2, AUD, DV, DEV = LI.S2, LI.AUD, LI.DV, LI.DEV
WIN, CENT, HUD_WIN = LI.WIN, LI.CENT, LI.HUD_WIN
MAP_ONLY = CENT[:, 0] <= 5
HUD_FIELDS = [("s_body", 0, 63), ("s_body", 1, 64), ("s_body", 2, 65), ("s_body", 3, 66), ("s_inv", 5, 67), ("s_inv", 0, 68),
              ("s_inv", 1, 69), ("s_inv", 2, 70), ("s_inv", 3, 71), ("s_inv", 4, 72), ("s_inv", 6, 73), ("s_inv", 7, 74),
              ("s_inv", 8, 75), ("s_inv", 9, 76), ("s_inv", 10, 77), ("s_inv", 11, 78)]   # 20261104 hud_codebook FIELDS


@torch.no_grad()
def belief_change(model, cfg, en, frames, frames_b, t, windows):
    k = len(windows)
    with TR.autocast(DEV, cfg):
        pp = M.Percept(*(en[key][:t].to(DEV).float()[None] for key in ("p", "g", "v")), en["fix"][:t].to(DEV)[None])
        x = model.belief.tokens(S.embed(model, pp, en["reads"][:t].to(DEV).float()[None]), en["reward"][:t].to(DEV)[None],
                                en["terminated"][:t].to(DEV)[None], en["action"][:t].to(DEV)[None])
        _, carry = model.belief(x, model.belief.initial(1, DEV))
        fix = torch.as_tensor(np.repeat(CENT[windows], 2, 0), device=DEV)
        g = DV.glimpses(model, cfg, torch.stack([frames[t], frames_b[t]] * k), fix.cpu()).float()
        P = M.Percept(en["p"][t].to(DEV).float().expand(2 * k, -1, -1)[:, None], g[:, None],
                      en["v"][t].to(DEV).float().expand(2 * k, -1)[:, None], fix[:, None])
        emb = S.embed(model, P, en["reads"][t].to(DEV).float().expand(2 * k, -1)[:, None])
        tok = model.belief.tokens(emb, en["reward"][t:t + 1].to(DEV).expand(2 * k)[:, None],
                                  en["terminated"][t:t + 1].to(DEV).expand(2 * k)[:, None],
                                  en["action"][t:t + 1].to(DEV).expand(2 * k)[:, None])
        y, _ = model.belief(tok, LI.rep_carry(carry, 2 * k))
    b = y[:, 0].float().view(k, 2, -1)
    ch = (b[:, 0] - b[:, 1]).norm(dim=-1)
    return ch.cpu().numpy(), (ch / b[:, 0].norm(dim=-1).clamp_min(1e-6)).cpu().numpy()


def main():
    model, cfg = S2.AI.load(S2.P15)
    model.requires_grad_(False).eval()
    enc, ids = AUD.encode_sample(model, cfg, "dev", 160, LI.SEED + 1)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    sub = S2.sublabels(L, np.asarray(ids))
    eps_, grids = S.episodes(sub), G.episode_grids(sub)
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rng = np.random.default_rng(20261163)
    acu = model.perception.acuity
    rows = {k: [] for k in ("episode", "night", "cat", "change", "rel")}
    for i, e in enumerate(ids):
        T = len(enc[i]["reward"]) - 1
        s0 = so[e]
        pos = L["s_pos"][s0:s0 + T + 1].astype(int)
        hud = {cell: L[arr][s0:s0 + T + 1, col].astype(int) for arr, col, cell in HUD_FIELDS}
        ts = np.arange(LI.FIRST, T + 1)
        moved = np.abs(pos[ts] - pos[ts - 1]).sum(1) > 0
        hchg = np.array([any(hud[c][t] != hud[c][t - 1] for c in hud) for t in ts])
        cand_t = ts[moved | hchg]
        if not len(cand_t):
            continue
        frames = DV.frames_of(eps_[i]); frames_b = DV.blurred(frames, acu)
        for t in sorted(rng.choice(cand_t, min(8, len(cand_t)), replace=False)):
            occ = np.concatenate([grids[i]["grid"][t].reshape(63) > 0, np.zeros(18, bool)])
            cover = occ[WIN].any(-1)
            d = pos[t] - pos[t - 1]
            new = np.zeros(81, bool)
            if d[0] == 1: new[6 * 9:7 * 9] = True
            if d[0] == -1: new[0:9] = True
            if d[1] == 1: new[[r * 9 + 8 for r in range(7)]] = True
            if d[1] == -1: new[[r * 9 for r in range(7)]] = True
            entering = new[WIN].any(-1) & MAP_ONLY & ~cover
            changed = np.zeros(81, bool)
            for c in hud:
                changed[c] = hud[c][t] != hud[c][t - 1]
            hud_changed = changed[WIN].any(-1) & HUD_WIN
            cats = {"creature": np.flatnonzero(cover & MAP_ONLY), "entering": np.flatnonzero(entering),
                    "static": np.flatnonzero(MAP_ONLY & ~cover & ~new[WIN].any(-1)),
                    "hud_changed": np.flatnonzero(hud_changed), "hud_unchanged": np.flatnonzero(HUD_WIN & ~hud_changed)}
            wins, names = [], []
            for name, cand in cats.items():
                if len(cand):
                    wins.append(int(rng.choice(cand))); names.append(name)
            ch, rel = belief_change(model, cfg, enc[i], frames, frames_b, int(t), wins)
            for j, name in enumerate(names):
                for key, val in (("episode", i), ("night", bool(L["s_light"][s0 + t] < 0.5)), ("cat", name),
                                 ("change", ch[j]), ("rel", rel[j])):
                    rows[key].append(val)
        if i % 40 == 0:
            print("episode", i, "rows", len(rows["cat"]), flush=True)
    z = {k: np.asarray(v) for k, v in rows.items()}
    np.savez_compressed(HERE / "belief_categories.npz", **z)
    rng2 = np.random.default_rng(0); eps = np.unique(z["episode"]); idx = {k: np.flatnonzero(z["episode"] == k) for k in eps}

    def boot(v, m):
        bs = []
        for _ in range(500):
            ii = np.concatenate([idx[k] for k in rng2.choice(eps, len(eps))]); ii = ii[m[ii]]
            if len(ii):
                bs.append(v[ii].mean())
        return {"mean": float(v[m].mean()) if m.any() else None, "ci95": np.quantile(bs, [0.025, 0.975]).tolist() if bs else None,
                "n": int(m.sum())}
    res = {p: {c: {k: boot(z[k], pm & (z["cat"] == c)) for k in ("change", "rel")}
               for c in ("creature", "entering", "static", "hud_changed", "hud_unchanged")}
           for p, pm in (("day", ~z["night"]), ("night", z["night"]))}
    (HERE / "belief_categories.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
