"""Within midsignal's reading (3): what makes a drop-prone cow's own-cell evidence weak. For the staying day cows of
midsignal (same filter and order): the terrain under the cow, the token separation |x_cow - x_terrain|, and the raw
pixel difference between the cow's cell and its same-terrain neighbour at full resolution (7 x 7) and after the
periphery's own pooling (3 x 3, Perception.forward), against the unit's drop count at tau 2/8.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.midsignal_cells
"""
import importlib, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

MS = importlib.import_module("artifacts.experiments.20261120_switching_dissection.midsignal")
C5, DL, S2, AI = MS.C5, MS.DL, MS.S2, MS.AI
HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda")


@torch.no_grad()
def main():
    model, cfg = AI.load(C5.CK); model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg); acu = cfg["perception"]["periphery"]["acuity"]
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L); grids = G.episode_grids(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    zz = np.load(DL.EV.S2 / "model.npz"); mg, mp = zz["m_grid"], zz["m_pos"].astype(int)
    units = [(w, int(rows[w, 0]), int(rows[w, 3]) + 1, int(kind), int(i), int(j), bool(u[7])) for u in U
             for w, kind, i, j in [u[:4]] if 9 <= int(rows[w, 3]) + 1 <= L["ep_T"][int(rows[w, 0])] and DL.deleted(mg, mp, u)]
    cows = []
    for e in sorted({x[1] for x in units}):
        for x in [u for u in units if u[1] == e]:
            t, kind, i, j, night = x[2:7]
            dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            r0, c0 = i - dpos[0], j - dpos[1]
            cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
            if cells and kind == 2 and not night:
                cows.append((x, (r0 * 9 + c0) if 0 <= r0 < 7 and 0 <= c0 < 9 else None))
    kept = np.load(HERE / "midsignal_units.npz")["kept|2|deployed"]        # [cows, 8], midsignal's order
    assert len(kept) == len(cows)
    out = []
    for n, ((w, e, t, kind, i, j, _), own) in enumerate(cows):
        if own is None:
            continue
        g = grids[e]; kinds, view = g["grid"].reshape(-1, 63)[t], g["view"].reshape(-1, 63)[t]
        c = TR.tensors(D.gather([dl[e]], [[(0, 1, t, 4, 4)]]), DEV)
        with TR.autocast(DEV, cfg):
            Pt = model.perception(c["frames"], c["fixation"])
        real = C5.kind_probs(R, Pt.p[0].float(), Pt.v[0].float()).argmax(-1)[0]
        if int(real[own]) != kind:
            continue
        cand = [cc for cc in range(63) if cc != own and kinds[cc] == 0 and view[cc] == view[own]]
        if not cand:
            continue
        near = min(cand, key=lambda cc: max(abs(cc // 9 - own // 9), abs(cc % 9 - own % 9)))
        fr = c["frames"][0, 0].float() / 255                               # [63, 63, 3]
        cell = lambda q: fr[(q // 9) * 7:(q // 9) * 7 + 7, (q % 9) * 7:(q % 9) * 7 + 7].permute(2, 0, 1)   # [3, 7, 7]
        pool = lambda x: torch.round(F.adaptive_avg_pool2d(x[None] * 255, acu))[0] / 255
        a, z = cell(own), cell(near)
        out.append({"terrain": int(view[own]), "drops": int((~kept[n]).sum()), "token_sep": float((Pt.p[0, 0, own].float() - Pt.p[0, 0, near].float()).norm()),
                    "pix_full": float((a - z).square().mean().sqrt()), "pix_pooled": float((pool(a) - pool(z)).square().mean().sqrt())})
    d = {k: np.array([x[k] for x in out]) for k in out[0]}
    res = {"units": len(out), "drop_rate": float(d["drops"].sum() / (8 * len(out))),
           "corr_drop": {k: float(np.corrcoef(d[k], d["drops"])[0, 1]) for k in ("token_sep", "pix_full", "pix_pooled")},
           "corr_token_sep": {k: float(np.corrcoef(d[k], d["token_sep"])[0, 1]) for k in ("pix_full", "pix_pooled")},
           "by_terrain": {}}
    for tt in np.unique(d["terrain"]):
        m = d["terrain"] == tt
        res["by_terrain"][int(tt)] = {"units": int(m.sum()), "drop_rate": float(d["drops"][m].sum() / (8 * m.sum())),
                                      "token_sep": float(d["token_sep"][m].mean()), "pix_full": float(d["pix_full"][m].mean()),
                                      "pix_pooled": float(d["pix_pooled"][m].mean())}
    lo = d["token_sep"] < np.quantile(d["token_sep"], 1 / 3)
    res["low_separation_third"] = {g_: {"pix_full": float(d["pix_full"][m].mean()), "pix_pooled": float(d["pix_pooled"][m].mean()),
                                        "pooled_over_full": float((d["pix_pooled"][m] / d["pix_full"][m]).mean()), "drop_rate": float(d["drops"][m].sum() / (8 * m.sum()))}
                                   for g_, m in (("low", lo), ("rest", ~lo))}
    (HERE / "midsignal_cells.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
