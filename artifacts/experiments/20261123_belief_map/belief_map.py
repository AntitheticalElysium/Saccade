"""What does P1.5's belief keep about the map beyond the view? PLAN.md (2026-11-23).

    JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_belief_map.belief_map
"""
import argparse
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
import torch.nn as nn

from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
DEV = S2.DEV
CLASSES = {"water": 3, "tree": 5, "lava": 14, "table": 11, "stone": 4, "sand": 13, "path": 7}
MOBS = ("zombies", "cows")
AGES = ((1, 16), (17, 64), (65, 10 ** 9))


def sector(dr, dc):
    return ((np.arctan2(dr, dc) + np.pi) / (2 * np.pi) * 8).astype(int) % 8


def targets(L, e, so, stride, first=16):
    """Per sampled frame of episode e: map, mob, self-position and thirst targets."""
    T = int(L["ep_T"][e]); a, b = so[e], so[e] + T + 1
    view, pos = L["s_view"][a:b], L["s_pos"][a:b].astype(int)
    grid, seen = np.full((64, 64), -1, np.int16), np.full((64, 64), -1, np.int32)
    rr, cc = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
    gr, gc = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    mob_seen = {m: np.full(L[f"s_{m}_m"].shape[1], -10 ** 6) for m in MOBS}
    rows = []
    for t in range(T + 1):
        inview = (np.abs(gr - pos[t, 0]) <= 3) & (np.abs(gc - pos[t, 1]) <= 4)
        if t >= first and (t - first) % stride == 0:
            row = {"t": t, "drink": int(L["s_body"][a + t, 2]), "self": (pos[t] - pos[0]).tolist()}
            zp, zm = L["s_zombies_p"][a + t].astype(int), L["s_zombies_m"][a + t]   # positive control (added after reading 1)
            row["zombie_adjacent"] = int(any(zm[s] and np.abs(zp[s] - pos[t]).max() <= 1 for s in range(len(zm))))
            for name, c in CLASSES.items():
                m = (grid == c) & ~inview
                if m.any():
                    dr, dc = gr[m] - pos[t, 0], gc[m] - pos[t, 1]
                    d = np.hypot(dr, dc); k = int(np.argmin(d))
                    if d[k] <= 24:
                        row[name] = (int(sector(dr[k], dc[k])), float(np.log(d[k])), int(t - seen[m][k]))
            for mname in MOBS:                                              # the nearest of each kind
                best = None
                mp, mm = L[f"s_{mname}_p"][a + t].astype(int), L[f"s_{mname}_m"][a + t]
                for s in range(len(mm)):
                    out = abs(mp[s, 0] - pos[t, 0]) > 3 or abs(mp[s, 1] - pos[t, 1]) > 4
                    if mm[s] and out and t - mob_seen[mname][s] <= 32:
                        dr, dc = mp[s, 0] - pos[t, 0], mp[s, 1] - pos[t, 1]
                        d = float(np.hypot(dr, dc))
                        if best is None or d < best[1]:
                            best = (int(sector(dr, dc)), d, int(t - mob_seen[mname][s]))
                if best is not None:
                    row[mname] = (best[0], float(np.log(best[1])), best[2])
            rows.append(row)
        ok = (pos[t, 0] + rr >= 0) & (pos[t, 0] + rr < 64) & (pos[t, 1] + cc >= 0) & (pos[t, 1] + cc < 64)
        grid[pos[t, 0] + rr[ok], pos[t, 1] + cc[ok]] = view[t][ok]
        seen[pos[t, 0] + rr[ok], pos[t, 1] + cc[ok]] = t
        for mname in MOBS:
            mp, mm = L[f"s_{mname}_p"][a + t].astype(int), L[f"s_{mname}_m"][a + t]
            for s in range(len(mm)):
                if not mm[s]:
                    mob_seen[mname][s] = -10 ** 6
                elif abs(mp[s, 0] - pos[t, 0]) <= 3 and abs(mp[s, 1] - pos[t, 1]) <= 4:
                    mob_seen[mname][s] = t
    return rows


class Probe(nn.Module):
    def __init__(self, n, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 256), nn.GELU(), nn.Linear(256, out))

    def forward(self, x):
        return self.net(x)


def fit(X, y, split, kind, out, seed=0):
    """Train on split 0, early-stop on split 1, predict split 2."""
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    mu, sd = Xt[split == 0].mean(0), Xt[split == 0].std(0) + 1e-5
    Xt = (Xt - mu) / sd
    yt = torch.as_tensor(y, device=DEV, dtype=torch.long if kind == "cls" else torch.float32)
    if kind == "reg":
        ym, ys = yt[split == 0].mean(0), yt[split == 0].std(0) + 1e-6
        yt = (yt - ym) / ys
    net = Probe(X.shape[1], out).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss() if kind == "cls" else nn.HuberLoss()
    tr, va, te = (torch.as_tensor(np.flatnonzero(split == k), device=DEV) for k in range(3))
    best, state, bad = 1e9, None, 0
    for ep in range(80):
        net.train()
        for ix in tr[torch.randperm(len(tr), device=DEV)].split(512):
            opt.zero_grad(); o = net(Xt[ix]); l = lossf(o, yt[ix] if kind == "cls" else yt[ix].view(o.shape)); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            o = net(Xt[va]); v = float(lossf(o, yt[va] if kind == "cls" else yt[va].view(o.shape)))
        if v < best - 1e-4:
            best, state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        o = net(Xt[te])
    if kind == "cls":
        return o.argmax(-1).cpu().numpy()
    return (o * ys + ym).cpu().numpy()


def boot(fn, ep, rng, reps=1000):
    u = np.unique(ep); idx = {k: np.flatnonzero(ep == k) for k in u}
    vals = [fn(np.concatenate([idx[k] for k in rng.choice(u, len(u))])) for _ in range(reps)]
    return np.quantile(vals, [.025, .975]).tolist()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    model, cfg = S2.AI.load(S2.P15); model.requires_grad_(False).eval()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rng = np.random.default_rng(20261124)
    pool = np.flatnonzero(L["ep_eps"] < 0.99)
    n = 24 if args.smoke else 360
    eps = np.sort(rng.choice(pool, n, replace=False))
    role = np.zeros(n, int); perm = rng.permutation(n)
    role[perm[int(n * 220 / 360):int(n * 260 / 360)]] = 1; role[perm[int(n * 260 / 360):]] = 2
    enc = S.encode(model, S2.sublabels(L, eps), cfg)
    feats = {"belief": [], "view": [], "view_record": []}
    meta = {"episode": [], "split": [], "drink": [], "self": [], "zombie_adjacent": []}
    tg = {k: [] for k in (*CLASSES, *MOBS)}
    for i, e in enumerate(eps):
        rows = targets(L, int(e), so, 3)
        en = enc[i]
        for r in rows:
            t = r["t"]
            b = en["b"][t].float().numpy(); p = en["p"][t].float().flatten().numpy(); v = en["v"][t].float().numpy()
            rd = en["reads"][t].float().numpy()
            feats["belief"].append(b); feats["view"].append(np.r_[p, v]); feats["view_record"].append(np.r_[p, v, rd])
            meta["episode"].append(i); meta["split"].append(role[i]); meta["drink"].append(r["drink"]); meta["self"].append(r["self"])
            meta["zombie_adjacent"].append(r["zombie_adjacent"])
            for k in tg:
                tg[k].append(r.get(k, (-1, np.nan, -1)))
    X = {k: np.stack(v) for k, v in feats.items()}
    M = {k: np.asarray(v) for k, v in meta.items()}
    Tg = {k: np.asarray(v, dtype=float) for k, v in tg.items()}
    print("frames", len(M["split"]), {k: int((Tg[k][:, 0] >= 0).sum()) for k in Tg}, flush=True)
    res = {"frames": int(len(M["split"])), "episodes": {"fit": int((role == 0).sum()), "val": int((role == 1).sum()), "test": int((role == 2).sum())},
           "direction": {}, "log_distance_r2": {}, "self_r2": {}}
    rng2 = np.random.default_rng(1)
    preds = {}
    for k in Tg:
        have = Tg[k][:, 0] >= 0
        if have.sum() < 300:
            continue
        sp = M["split"][have]; y = Tg[k][have, 0].astype(int); ep = M["episode"][have]; age = Tg[k][have, 2]
        te = sp == 2
        maj = np.bincount(y[sp == 0], minlength=8).argmax()
        out = {"n_test": int(te.sum()), "majority": float((y[te] == maj).mean())}
        hit = {}
        for name, Xf in X.items():
            p = fit(Xf[have], y, sp, "cls", 8)
            hit[name] = (p == y[te]).astype(float)
            out[name] = {"acc": float(hit[name].mean()), "ci95": boot(lambda ii: hit[name][ii].mean(), ep[te], rng2)}
        for ctrl in ("view", "view_record"):
            d = hit["belief"] - hit[ctrl]
            out[f"belief_minus_{ctrl}"] = {"mean": float(d.mean()), "ci95": boot(lambda ii: d[ii].mean(), ep[te], rng2)}
        if k in CLASSES:
            out["by_age"] = {}
            for lo, hi in AGES:
                m = (age[te] >= lo) & (age[te] <= hi)
                if m.sum() >= 30:
                    out["by_age"][f"{lo}-{hi}"] = {"n": int(m.sum()), **{nm: float(hit[nm][m].mean()) for nm in hit},
                                                   "majority": float((y[te][m] == maj).mean())}
        if k == "water":
            dr = M["drink"][have][te]
            out["thirst"] = {lab: {"n": int(m.sum()), **{nm: float(hit[nm][m].mean()) for nm in hit}}
                             for lab, m in (("thirsty_le4", dr <= 4), ("not_ge7", dr >= 7))}
        res["direction"][k] = out
        yd = Tg[k][have, 1]
        res["log_distance_r2"][k] = {}
        for name, Xf in X.items():
            p = fit(Xf[have], yd[:, None], sp, "reg", 1)[:, 0]
            res["log_distance_r2"][k][name] = float(1 - ((p - yd[te]) ** 2).sum() / ((yd[te] - yd[te].mean()) ** 2).sum())
        print(k, json.dumps(out)[:400], flush=True)
    res["positive_controls"] = {}                                           # added after reading 1: does the probe work?
    for name_t, ncls in (("drink", 10), ("zombie_adjacent", 2)):
        y = M[name_t].astype(int); sp = M["split"]; te = sp == 2
        maj = np.bincount(y[sp == 0], minlength=ncls).argmax()
        res["positive_controls"][name_t] = {"majority": float((y[te] == maj).mean()), "positives_test": int((y[te] > 0).sum()) if ncls == 2 else None,
                                            **{nm: float((fit(Xf, y, sp, "cls", ncls) == y[te]).mean()) for nm, Xf in X.items()}}
        if ncls == 2:                                                       # balanced accuracy for the rare positive
            for nm, Xf in X.items():
                pr = fit(Xf, y, sp, "cls", 2)
                res["positive_controls"][name_t][nm + "_balanced"] = float(((pr == 1) & (y[te] == 1)).sum() / max((y[te] == 1).sum(), 1) / 2
                                                                          + ((pr == 0) & (y[te] == 0)).sum() / max((y[te] == 0).sum(), 1) / 2)
    print(json.dumps(res["positive_controls"]), flush=True)
    ys = M["self"].astype(float); sp = M["split"]; te = sp == 2
    for name, Xf in X.items():
        p = fit(Xf, ys, sp, "reg", 2)
        res["self_r2"][name] = {ax: float(1 - ((p[:, j] - ys[te, j]) ** 2).sum() / ((ys[te, j] - ys[te, j].mean()) ** 2).sum())
                                for j, ax in enumerate(("rows", "cols"))}
    res["self_r2"]["note"] = "displacement from the episode start"
    (HERE / ("belief_map_smoke.json" if args.smoke else "belief_map.json")).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["self_r2"]), flush=True)


if __name__ == "__main__":
    main()
