"""C1 of the eye's checks (PLAN.md): why P1.5's chooser predicts the next frame better than F6's and F6.1's.

The same DEV frames go through each world's own coarse encoder (`chooser.coarse`) and its final chooser head
(checkpoint 10's `fixation_controller`), with inputs built exactly as `chooser.inputs` does. Per region, read with the
F2 review's semantics on the DEV labels (`20261104_p1f2_review/trace.training_semantics`):
- the token change from the previous frame;
- the head's surprise and contingency;
- the head's next-token error.

Also each policy's expected coverage of entities, changed HUD and entering map, by day and night.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c1 [N_EPISODES]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, train as TR
from saccade.eval import gates as G, s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
TRC = importlib.import_module("artifacts.experiments.20261104_p1f2_review.trace")
A = TRC.A
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
WORLDS = {"F6": "20261113_p1f6", "F6.1": "20261114_p1f61", "P1.5": "20261116_p15"}
GROUPS = ("mob", "changed_hud", "hud_static", "entering_map", "terrain_edit", "map_static")
DEV = torch.device("cuda")


def semantics(L):
    """trace.training_semantics on every DEV frame t >= 1 (as trace.main sets it up) -> masks, light, row lookup."""
    A.L = L; A.E = G.episode_grids(L); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]; A.OFF = loff; A.AOFF = np.r_[0, np.cumsum(L["ep_T"])]
    eall = np.concatenate([np.full(int(T), e) for e, T in enumerate(L["ep_T"])])
    tall = np.concatenate([np.arange(1, int(T) + 1) for T in L["ep_T"]]); si = loff[eall] + tall
    sem = TRC.training_semantics(L, eall, tall, si, loff)
    lookup = np.full(int(loff[-1]), -1, int); lookup[si] = np.arange(len(si))
    m = sem["masks"]
    hud = np.zeros_like(m["changed_hud"]); hud[:, 63:] = True
    groups = {"mob": m["any_entity"], "changed_hud": m["changed_hud"], "hud_static": hud & ~m["changed_hud"],
              "entering_map": m["entering_map"], "terrain_edit": m["terrain_edit"]}
    used = np.zeros_like(hud)
    for k in groups:
        groups[k] = groups[k] & ~used; used |= groups[k]
    groups["map_static"] = ~used & ~hud
    return groups, sem["light"], lookup, loff, {k: m[k] for k in ("any_entity", "cows", "zombies", "changed_hud", "entering_map")}


@torch.no_grad()
def world(name, L, eps, groups, light, lookup, loff, cover):
    ck = EXP / WORLDS[name] / "run/ckpt_10.pt"
    model, cfg = AI.load(ck)
    model.requires_grad_(False).eval()
    pol, Hh = dict(cfg["fixation"]["chooser"]), cfg["fixation"]["chooser"]["history"]
    head = C.Head(Hh).to(DEV); head.load_state_dict(torch.load(ck, weights_only=False, map_location="cpu")["fixation_controller"]["head"]); head.eval()
    win = C.WIN.to(DEV)
    dl = S.episodes(L)
    acc = {g: np.zeros(4) for g in GROUPS}                                 # sums: change, surprise, contingency, cells
    cov = {f"{k}|{lt}": [0., 0., 0.] for k in cover for lt in ("day", "night", "all")}   # learned, uniform, frames
    hl = [0., 0.]
    for e in eps:
        T = int(L["ep_T"][e])
        if T < Hh + 1:
            continue
        chunk = D.gather([dl[e]], [[(0, 1, t, 4, 4) for t in range(T + 1)]])
        fr = torch.as_tensor(chunk["frames"], device=DEV)
        with TR.autocast(DEV, cfg):
            p, v = C.coarse(model.perception, fr.flatten(0, 1))
        p, v = p.reshape(1, T + 1, 81, 16).float(), v.reshape(1, T + 1, 8).float()
        xx, y = C.inputs(p, v, chunk, Hh, Hh)                              # targets t = Hh .. T
        with torch.autocast("cuda", enabled=False):
            taken = head(xx); noop = head((xx[0], xx[1], torch.zeros_like(xx[2])))
        s = (y - taken).square().sum(-1); c = (taken - noop).square().sum(-1); ch = (y - xx[0][:, -1]).square().sum(-1)
        prob = C.policy(c[:, win].sum(-1), s[:, win].sum(-1), pol).cpu().numpy()
        rows = lookup[loff[e] + np.arange(Hh, T + 1)]
        assert (rows >= 0).all()
        s, c, ch = s.cpu().numpy(), c.cpu().numpy(), ch.cpu().numpy()
        hl[0] += float((y - taken).square().mean(-1).sum()); hl[1] += len(rows) * 81
        for gname in GROUPS:
            mk = groups[gname][rows]
            acc[gname] += [float((ch * mk).sum()), float((s * mk).sum()), float((c * mk).sum()), float(mk.sum())]
        lt = np.where(light[rows] < 0.5, "night", "day")
        for k, mask in cover.items():
            present = mask[rows].any(-1)
            hit = mask[rows][:, C.WIN.numpy()].any(-1)                      # [n, 49]
            lw, uw = (prob * hit).sum(-1), hit.mean(-1)
            for tag in ("all", "day", "night"):
                sel = present & ((lt == tag) if tag != "all" else True)
                cov[f"{k}|{tag}"][0] += float(lw[sel].sum()); cov[f"{k}|{tag}"][1] += float(uw[sel].sum()); cov[f"{k}|{tag}"][2] += float(sel.sum())
    tot = {i: sum(acc[g][i] for g in GROUPS) for i in range(3)}
    out = {"head_mse": hl[0] / hl[1],
           "groups": {g: {"cells": acc[g][3], "change_share": acc[g][0] / tot[0], "surprise_share": acc[g][1] / tot[1],
                          "contingency_share": acc[g][2] / tot[2], "change_per_cell": acc[g][0] / max(acc[g][3], 1),
                          "surprise_per_cell": acc[g][1] / max(acc[g][3], 1)} for g in GROUPS},
           "coverage": {k: {"learned": v[0] / max(v[2], 1), "uniform": v[1] / max(v[2], 1), "frames": v[2]} for k, v in cov.items()}}
    del model, head; torch.cuda.empty_cache()
    return out


def main(n=150):
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, light, lookup, loff, cover = semantics(L)
    eps = np.sort(np.random.default_rng(20261120).choice(len(L["ep_T"]), n, replace=False))
    res = {"episodes": eps.tolist()}
    for name in WORLDS:
        res[name] = world(name, L, eps, groups, light, lookup, loff, cover)
        print("==", name, "head next-token MSE", round(res[name]["head_mse"], 6), flush=True)
    print("share of token change / chooser surprise / contingency, and per-cell surprise, by region")
    for g in GROUPS:
        print(f"  {g:13s}", " | ".join(f"{w}: {r['groups'][g]['change_share']:.3f} / {r['groups'][g]['surprise_share']:.3f} / "
                                       f"{r['groups'][g]['contingency_share']:.3f}  per-cell {r['groups'][g]['surprise_per_cell']:.4f}"
                                       for w, r in ((w, res[w]) for w in WORLDS)))
    print("expected coverage when present: learned (uniform)")
    for k in res["P1.5"]["coverage"]:
        print(f"  {k:20s}", " | ".join(f"{w}: {res[w]['coverage'][k]['learned']:.3f} ({res[w]['coverage'][k]['uniform']:.3f})" for w in WORLDS))
    (HERE / ("c1.json" if n == 150 else f"c1_{n}.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
