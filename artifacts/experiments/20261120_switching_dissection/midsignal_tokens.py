"""Removal or reader-thresholded hedge, and is the coin already in the model's own posterior (PLAN.md, the user 11-22).
The estimate read in token space: at the cow's own cell its cow weight a on the terrain -> cow line, at each neighbour
the cow weight moved there, their sum M. Native K = 4 chain and the chains from the real frame noised to 2/8 and 4/8
(coin_form.paths), plus the finest code's estimates (posterior means) at tau 0, 2/8, 4/8 (midsignal.star_estimates).
P1.5 checkpoint 10, C5's day cows, 8 draws, S2's readers.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.midsignal_tokens [smoke] [CKPT]

Another checkpoint (CKPT) is read with its own S1 kind reader, and its outputs are tagged with its run.
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade.eval import s1 as S, gates as G

MS = importlib.import_module("artifacts.experiments.20261120_switching_dissection.midsignal")
CF, C2, C5, DL, S2, AI = MS.CF, MS.C2, MS.C5, MS.DL, MS.S2, MS.AI
HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda")
GRID_A = np.round(np.arange(0, 1.0001, 0.05), 2)


def main(smoke=False, ck=None, out=HERE):
    ck = Path(ck).resolve() if ck else C5.CK
    model, cfg = AI.load(ck); model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg) if ck == C5.CK else {"kind": C5.reader(model, cfg, False)}
    tag = ("_smoke" if smoke else "") + ("" if ck == C5.CK else "_" + ck.parent.parent.name + "_" + ck.stem)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L); grids = G.episode_grids(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    zz = np.load(DL.EV.S2 / "model.npz"); mg, mp = zz["m_grid"], zz["m_pos"].astype(int)
    units = [(w, int(rows[w, 0]), int(rows[w, 3]) + 1, int(kind), int(i), int(j), bool(u[7])) for u in U
             for w, kind, i, j in [u[:4]] if 9 <= int(rows[w, 3]) + 1 <= L["ep_T"][int(rows[w, 0])] and DL.deleted(mg, mp, u)]
    cows = []                                                                 # midsignal's filter and order
    for e in sorted({x[1] for x in units}):
        for x in [u for u in units if u[1] == e]:
            t, kind, i, j, night = x[2:7]
            dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            r0, c0 = i - dpos[0], j - dpos[1]
            cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
            if cells and kind == 2 and not night:
                cows.append((x, cells, (r0 * 9 + c0) if 0 <= r0 < 7 and 0 <= c0 < 9 else None))
    ref = np.load(HERE / "midsignal_units.npz") if ck == C5.CK else None   # midsignal's drops: P1.5 only
    if smoke:
        e3 = sorted({u[0][1] for u in cows})[:4]; cows = [u for u in cows if u[0][1] in e3]
    MS.LEVELS = (0, 2, 4)
    rec, checks = [], {"star_vs_paths_max_abs": 0.0, "kept2_vs_midsignal": [0, 0]}
    n_all = 0
    for e in sorted({u[0][1] for u in cows}):
        st = C2.episode(model, cfg, dl, e)
        us = [u for u in cows if u[0][1] == e]; ts = sorted({u[0][2] for u in us})
        real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1)
        paths = [CF.paths(model, cfg, st, e, ts, dr) for dr in range(C5.DRAWS)]
        stars = [MS.star_estimates(model, cfg, st, e, ts, dr)[0] for dr in range(C5.DRAWS)]
        for q, s in zip(paths, stars):                                       # the two instruments agree exactly
            for lv, i_ in ((2, 1), (4, 2)):
                checks["star_vs_paths_max_abs"] = max(checks["star_vs_paths_max_abs"],
                                                      float((s[(lv, "deployed")][0] - q["starts"][i_][0][0]).abs().max()))
        kind_of = lambda pv: C5.kind_probs(R, pv[0], pv[1]).argmax(-1)         # [W, 63]
        g = grids[e]; n63 = lambda a: a.reshape(len(a), 63)
        for (w, _, t, kind, i, j, _), cells, own in us:
            qi = ts.index(t); n_unit = n_all; n_all += 1
            nb = lambda cls: (cls[qi][cells] == kind).any().item()
            r = {"unit": n_unit,
                 "kept_native": np.array([nb(kind_of(q["native"])) for q in paths]),
                 "kept_from2": np.array([nb(kind_of(q["starts"][1][1])) for q in paths]),
                 "drop2_dep": np.array([not nb(kind_of(q["starts"][1][0])) for q in paths])}
            if ref is not None:
                k2 = ref["kept|2|deployed"][n_unit]
                checks["kept2_vs_midsignal"][0] += int(((~r["drop2_dep"]) == k2).sum()); checks["kept2_vs_midsignal"][1] += C5.DRAWS
            if own is None or int(real[qi, own]) != kind:
                rec.append(r); continue
            kinds, view = n63(g["grid"])[t], n63(g["view"])[t]
            cand = [cc for cc in range(63) if cc != own and kinds[cc] == 0 and view[cc] == view[own]]
            if not cand:
                rec.append(r); continue
            near = min(cand, key=lambda cc: max(abs(cc // 9 - own // 9), abs(cc % 9 - own % 9)))
            P0 = st[1].p[0, t].float(); V0 = st[1].v[0, t].float()
            xc, xt = P0[own], P0[near]; D = xc - xt; DD = float(D @ D)
            nbr = [c for c in cells if c != own and kinds[c] == 0]
            same = [c for c in nbr if view[c] == view[own]]

            def read(p_est):                                                  # p_est [81, 16] -> a, off-line, M_same, M_all
                a = float((p_est[own] - xt) @ D) / DD
                off = float((p_est[own] - xt - a * D).norm()) / DD ** 0.5
                ac = {c: float((p_est[c] - P0[c]) @ D) / DD for c in nbr}
                return a, off, a + sum(ac[c] for c in same), a + sum(ac.values())
            tok = {}
            for name, get in (("native", lambda q, s, i_: q["estimates"][i_][0][qi]),
                              ("from2", lambda q, s, i_: q["starts"][1][2][i_][0][qi]),
                              ("from4", lambda q, s, i_: q["starts"][2][2][i_][0][qi])):
                for i_ in range({"native": 4, "from2": 3, "from4": 2}[name]):
                    tok[f"{name}{i_}"] = np.array([read(get(q, s, i_).float()) for q, s in zip(paths, stars)])   # [D, 4]
            for lv in (0, 2, 4):
                for code in ("deployed", "finest"):
                    tok[f"star{lv}_{code}"] = np.array([read(s[(lv, code)][0][qi].float()) for s in stars])
                    tok[f"keepown_star{lv}_{code}"] = np.array([int(kind_of((s[(lv, code)][0], s[(lv, code)][1]))[qi, own] == kind)
                                                                for s in stars])
                    tok[f"keepnb_star{lv}_{code}"] = np.array([nb(kind_of((s[(lv, code)][0], s[(lv, code)][1]))) for s in stars])
            zc = torch.stack([q["star"][1][0][qi, own] for q in paths])       # the 2/8 input at the own cell [D, 16]
            tau = 2 / 8
            tok["lambda2"] = (((zc - tau * xt).square().sum(-1) - (zc - tau * xc).square().sum(-1)) / (2 * (1 - tau) ** 2)).cpu().numpy()
            blends = P0[None].repeat(len(GRID_A), 1, 1)
            blends[:, own] = xt + torch.as_tensor(GRID_A, device=DEV, dtype=torch.float32)[:, None] * D
            cow = (C5.kind_probs(R, blends, V0[None].expand(len(GRID_A), -1)).argmax(-1)[:, own] == kind).cpu().numpy()
            r.update(tok); r["flip_a"] = float(GRID_A[cow.argmax()]) if cow.any() else np.nan
            r["sep"] = DD ** 0.5; r["n_same"] = len(same); r["n_nbr"] = len(nbr)
            rec.append(r)
        print(e, len(rec), flush=True)
    res = {"units": len(rec), "draws": C5.DRAWS, "checks": checks}
    kn = np.stack([r["kept_native"] for r in rec])
    res["native_keep_all"] = float(kn.mean())
    tr = [r for r in rec if "flip_a" in r]
    res["token_units"] = len(tr)
    fl = np.array([r["flip_a"] for r in tr])
    res["0_reader_flip_a"] = {"median": float(np.nanmedian(fl)), "q25": float(np.nanquantile(fl, .25)),
                              "q75": float(np.nanquantile(fl, .75)), "never": int(np.isnan(fl).sum())}
    T = lambda key, col: np.stack([r[key][:, col] for r in tr])               # [units, D]
    res["0_off_line_median"] = {k: float(np.median(T(k, 1))) for k in ("star0_finest", "star2_finest", "star2_deployed", "star4_finest")}
    keep_t = np.stack([r["kept_native"] for r in tr])
    M0 = T("star0_finest", 2); M0a = T("star0_finest", 3)
    res["1_posterior_vs_chain"] = {
        "M0_same_mean": float(M0.mean()), "M0_all_mean": float(M0a.mean()), "a0_own_mean": float(T("star0_finest", 0).mean()),
        "native_keep": float(keep_t.mean()), "gap_same": float(M0.mean() - keep_t.mean()), "gap_all": float(M0a.mean() - keep_t.mean()),
        "unit_corr_M0_keep": float(np.corrcoef(M0.mean(1), keep_t.mean(1))[0, 1]),
        "native_chain_M_same": [float(T(f"native{i}", 2).mean()) for i in range(4)],
        "native_chain_a_own": [float(T(f"native{i}", 0).mean()) for i in range(4)],
        "staying_share_data": 0.603}
    for lv in (2, 4):
        for code in ("deployed", "finest"):
            k = f"star{lv}_{code}"; dr = ~np.stack([r[f"keepnb_{k}"] for r in tr]).astype(bool)
            Mv, av = T(k, 2), T(k, 0)
            res[f"2_{k}"] = {"reader_drop": float(dr.mean()),
                             "drops_M_median": float(np.median(Mv[dr])) if dr.any() else None,
                             "drops_a_median": float(np.median(av[dr])) if dr.any() else None,
                             "drops_M_lt_0.3": float((Mv[dr] < 0.3).mean()) if dr.any() else None,
                             "drops_M_ge_0.6": float((Mv[dr] >= 0.6).mean()) if dr.any() else None,
                             "kept_M_median": float(np.median(Mv[~dr])), "kept_a_median": float(np.median(av[~dr]))}
    d2 = np.stack([r["drop2_dep"] for r in tr]); fin2 = np.stack([r["kept_from2"] for r in tr]); a2 = T("from20", 0); a4 = T("from21", 0)
    out3 = {"drops_at_2": int(d2.sum())}
    for outcome, sel in (("ending_deleted", d2 & ~fin2), ("repaired", d2 & fin2)):
        out3[outcome] = {"n": int(sel.sum()), "share_a2_gt_quarter": float((a2[sel] > 0.25).mean()) if sel.any() else None,
                     "a2_median": float(np.median(a2[sel])) if sel.any() else None, "a4_median": float(np.median(a4[sel])) if sel.any() else None}
    res["3_non_repair"] = out3
    sep = np.array([r["sep"] for r in tr]); qs = np.quantile(sep, [0, 1 / 3, 2 / 3, 1])
    terc = np.digitize(sep, qs[1:-1])                                         # 0 low, 1 mid, 2 high
    lam = np.stack([r["lambda2"] for r in tr]); a2f = T("star2_finest", 0)
    lb = np.quantile(lam, [0, .2, .4, .6, .8, 1])
    res["4_separation"] = {"tercile_edges": [float(x) for x in qs],
                           "M0_by_tercile": [float(M0[terc == k].mean()) for k in range(3)],
                           "a0_by_tercile": [float(T("star0_finest", 0)[terc == k].mean()) for k in range(3)],
                           "native_keep_by_tercile": [float(keep_t[terc == k].mean()) for k in range(3)],
                           "a2_finest_by_lambda_bin_and_tercile": [
                               [float(lo), float(hi)] + [float(a2f[(terc == k)[:, None] & (lam >= lo) & (lam <= hi)].mean())
                                                         if ((terc == k)[:, None] & (lam >= lo) & (lam <= hi)).any() else None for k in range(3)]
                               for lo, hi in zip(lb[:-1], lb[1:])]}
    np.savez_compressed(Path(out) / f"midsignal_tokens_units{tag}.npz",
                        **{k: np.stack([r[k] for r in tr]) for k in tr[0] if k not in ("unit",)},
                        units=np.array([r["unit"] for r in tr]))
    res["checkpoint"] = str(ck)
    (Path(out) / f"midsignal_tokens{tag}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main("smoke" in sys.argv[1:], next((a for a in sys.argv[1:] if a.endswith(".pt")), None))
