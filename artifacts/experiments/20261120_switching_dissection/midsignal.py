"""Why the mid-signal estimate drops the cow (PLAN.md, the user 11-21). From the real frame noised to tau 2/8 and 4/8
with the draw's own noise, the estimate under the deployed step code (d = 1/4) and the finest (d = 1/8): unit
concentration, hedge against absence, the cell's noise evidence, shortcut output against flow output. P1.5 checkpoint
10, C5's day cows, 8 draws, S2's readers.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.midsignal [smoke]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

CF = importlib.import_module("artifacts.experiments.20261120_switching_dissection.coin_form")
C2, C5, DL, S2, AI = CF.C2, CF.C5, CF.DL, CF.S2, CF.AI
HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda")
LEVELS = (2, 4)


@torch.no_grad()
def star_estimates(model, cfg, st, e, ts, draw):
    """C4's seeds; z*_tau = (1 - tau) eps0 + tau x for tau in LEVELS / k_max, read under the deployed and the finest
    step code -> {(level, code): (p, v)}, eps0, x."""
    c, Pe, rin, bp = st
    m, sc = cfg["p1"]["monitor"], cfg["edm"]["shortcut"]
    P, k = cfg["data"]["context_prefix"], sc["k_max"]
    _, e_step, _ = M.shortcut_ladder(cfg); e_max = k.bit_length() - 1
    stack = lambda a: torch.stack([a[0, t - P:t + 1] for t in ts])
    x0 = [stack(a) for a in Pe.parts()]
    W = len(ts); ctx_noise, z = [], []
    for t in ts:
        g = torch.Generator(device=DEV).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(e) * 4_099 + int(t))
        ctx_noise.append(tuple(torch.randn(a.shape[1:], device=DEV, generator=g) for a in x0))
        g2 = torch.Generator(device=DEV).manual_seed(77_000_003 * (draw + 1) + int(e) * 4_099 + int(t))
        z.append((torch.randn(x0[0].shape[2:], device=DEV, generator=g2), torch.randn(x0[2].shape[2:], device=DEV, generator=g2)))
    ctx_noise = [torch.stack(q) for q in zip(*ctx_noise)]
    eps = (torch.stack([q[0] for q in z]), torch.stack([q[1] for q in z])); xr = (x0[0][:, -1].float(), x0[2][:, -1].float())
    tc = sc["context_signal"] / k
    xc = [(1 - tc) * n + tc * x for x, n in zip(x0, ctx_noise)]
    mem, valid = (model.belief.memory(stack(bp)), stack(rin)), stack(c["valid"])
    pa, fix, epi = stack(c["prev_action"]), stack(Pe.fix), stack(c["episode"])
    out = {}
    for s_ in LEVELS:
        tau = s_ / k
        p_, v_ = (1 - tau) * eps[0] + tau * xr[0], (1 - tau) * eps[1] + tau * xr[1]
        for name, ec in (("deployed", e_step), ("finest", e_max)):
            sg = torch.full((W, P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = s_
            em = torch.full((W, P + 1), e_max, device=DEV, dtype=torch.long); em[:, -1] = ec
            xin = [x.clone() for x in xc]
            xin[0][:, -1], xin[1][:, -1], xin[2][:, -1] = p_, torch.zeros_like(xin[1][:, -1]), v_
            code = M.shortcut_code(sg, em, k)
            with TR.autocast(DEV, cfg):
                o = model.denoiser(tuple(xin), (code, code), pa, mem, fix, epi, valid)
            out[(s_, name)] = (o[0][:, -1].float(), o[2][:, -1].float(), p_)
    return out, eps, xr


def logistic(x, y, iters=50):
    """IRLS for P(y) = sigmoid(a + b x) -> (a, b)."""
    X = np.stack([np.ones_like(x), x], 1); w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - p))
    return w


def main(smoke=False):
    model, cfg = AI.load(C5.CK); model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg)
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
                cows.append((x, cells, (r0 * 9 + c0) if 0 <= r0 < 7 and 0 <= c0 < 9 else None))
    if smoke:
        e3 = sorted({u[0][1] for u in cows})[:4]; cows = [u for u in cows if u[0][1] in e3]
    rec = []
    for e in sorted({u[0][1] for u in cows}):
        st = C2.episode(model, cfg, dl, e)
        us = [u for u in cows if u[0][1] == e]; ts = sorted({u[0][2] for u in us})
        real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1)
        draws = [star_estimates(model, cfg, st, e, ts, dr) for dr in range(C5.DRAWS)]
        probs = {key: torch.stack([C5.kind_probs(R, q[0][key][0], q[0][key][1]) for q in draws]) for key in draws[0][0]}   # [D, W, 63, C]
        g = grids[e]; n63 = lambda a: a.reshape(len(a), 63)
        for (w, _, t, kind, i, j, _), cells, own in us:
            qi = ts.index(t)
            r = {"unit": len(rec)}
            for key, pr in probs.items():
                pk = pr[:, qi, :, kind]                                   # [D, 63]
                r[f"kept|{key[0]}|{key[1]}"] = (pr[:, qi][:, cells].argmax(-1) == kind).any(-1).cpu().numpy()
                r[f"mass|{key[0]}|{key[1]}"] = pk[:, cells].sum(-1).cpu().numpy(); r[f"maxp|{key[0]}|{key[1]}"] = pk[:, cells].amax(-1).cpu().numpy()
                if own is not None:
                    r[f"own|{key[0]}|{key[1]}"] = (pr[:, qi, own].argmax(-1) == kind).cpu().numpy()
            if own is not None and int(real[qi, own]) == kind:              # (3): a cow on its own cell in the real frame
                kinds, view = n63(g["grid"])[t], n63(g["view"])[t]
                cand = [cc for cc in range(63) if cc != own and kinds[cc] == 0 and view[cc] == view[own]]
                if cand:
                    near = min(cand, key=lambda cc: max(abs(cc // 9 - own // 9), abs(cc % 9 - own % 9)))
                    xc, xt = st[1].p[0, t, own].float(), st[1].p[0, t, near].float()
                    for s_ in LEVELS:
                        tau = s_ / 8
                        zc = torch.stack([q[0][(s_, "deployed")][2][qi, own] for q in draws])   # [D, 16]
                        lam = ((zc - tau * xt).square().sum(-1) - (zc - tau * xc).square().sum(-1)) / (2 * (1 - tau) ** 2)
                        r[f"lambda|{s_}"] = lam.cpu().numpy()
                    r["sep"] = float((xc - xt).norm())
            rec.append(r)
        print(e, len(rec), flush=True)
    res = {"units": len(rec), "draws": C5.DRAWS}
    for s_ in LEVELS:
        for name in ("deployed", "finest"):
            key = f"{s_}|{name}"
            kept = np.stack([r[f"kept|{key}"] for r in rec]); drop = ~kept
            p = drop.mean(); k_ = drop.sum(1); disp = float(k_.var() / max(8 * p * (1 - p), 1e-9))
            mass = np.stack([r[f"mass|{key}"] for r in rec]); maxp = np.stack([r[f"maxp|{key}"] for r in rec])
            out = {"drop": float(p), "dispersion": disp, "share_of_drops_in_units_dropped_4plus": float(drop[k_ >= 4].sum() / max(drop.sum(), 1)),
                   "units_dropped_all_8": int((k_ == 8).sum()), "units_never_dropped": int((k_ == 0).sum())}
            if drop.any():
                out["drops_mass_ge_0.5"] = float((mass[drop] >= 0.5).mean()); out["drops_mass_lt_0.2"] = float((mass[drop] < 0.2).mean())
                out["drops_mean_mass"] = float(mass[drop].mean()); out["drops_mean_maxp"] = float(maxp[drop].mean())
                out["kept_mean_mass"] = float(mass[kept].mean())
            res[key] = out
        lam = [r for r in rec if f"lambda|{s_}" in r]
        if lam:
            x = np.concatenate([r[f"lambda|{s_}"] for r in lam]); y = np.concatenate([r[f"own|{s_}|deployed"] for r in lam]).astype(float)
            a, b_ = logistic(x, y)
            qs = np.quantile(x, [0, .1, .25, .5, .75, .9, 1])
            bins = [(float(lo), float(hi), float(y[(x >= lo) & (x <= hi)].mean()), int(((x >= lo) & (x <= hi)).sum())) for lo, hi in zip(qs[:-1], qs[1:])]
            res[f"noise_evidence|{s_}"] = {"units": len(lam), "draws": int(len(x)), "own_keep": float(y.mean()), "logistic_slope": float(b_),
                                           "indifference_lambda": float(-a / b_) if b_ != 0 else None, "bins_lo_hi_keep_n": bins,
                                           "lambda_mean": float(x.mean()), "lambda_share_negative": float((x < 0).mean()),
                                           "token_separation_mean": float(np.mean([r["sep"] for r in lam]))}
    # Within reading (3): when the own cell loses the cow, is it relocated to a neighbour (a cow that moved) or deleted;
    # and the drop-prone units against the cow token's separation from its terrain (how weak the cell's evidence is).
    for s_ in LEVELS:
        lam = [r for r in rec if f"lambda|{s_}" in r]
        if lam:
            own = np.concatenate([r[f"own|{s_}|deployed"] for r in lam]); kept = np.concatenate([r[f"kept|{s_}|deployed"] for r in lam])
            sep = np.array([r["sep"] for r in lam]); kd = np.array([(~r[f"kept|{s_}|deployed"]).sum() for r in lam])
            qs = np.quantile(sep, [0, 1 / 3, 2 / 3, 1])
            res[f"own_loss|{s_}"] = {"own_cell_losses": int((~own).sum()), "relocated_share": float((kept & ~own).sum() / max((~own).sum(), 1)),
                                     "deleted_share": float((~kept & ~own).sum() / max((~own).sum(), 1)),
                                     "drop_by_separation_tercile": [(float(lo), float(hi), float(kd[(sep >= lo) & (sep <= hi)].mean() / 8))
                                                                    for lo, hi in zip(qs[:-1], qs[1:])],
                                     "separation_drop_correlation": float(np.corrcoef(sep, kd)[0, 1])}
    np.savez_compressed(HERE / f"midsignal_units{'_smoke' if smoke else ''}.npz",
                        **{k: np.stack([r[k] for r in rec]) for k in rec[0] if k.startswith(("kept|", "mass|", "maxp|"))})
    (HERE / f"midsignal{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main("smoke" in sys.argv[1:])
