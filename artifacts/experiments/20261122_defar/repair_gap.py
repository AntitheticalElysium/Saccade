"""The repair gap, a held-out gate before the drift-row readings (PLAN.md, user 11-22: "is 625 updates enough ... Is there
a way to know this before committing to running the evals on it?"). For each checkpoint on its own tokens: from the real
frame noised to 2/8 (4/8), one deployed quarter-step gives the drifted state at 4/8 (6/8). The finest output there is
compared with the finest output on the real frame noised to the same level, on the day cows' 3 x 3 neighbourhoods and
on the whole frame. It also gives the cow's token-space mass in both. The day-coin panel's day cows, 4 draws, the
monitor's context seeds.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261122_defar.repair_gap [CKPT ...]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

MT = importlib.import_module("artifacts.experiments.20261120_switching_dissection.midsignal_tokens")
MS, C2, C5, DL, AI = MT.MS, MT.C2, MT.C5, MT.DL, MT.AI
HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda")
DRAWS = 4
CKS = [HERE.parent / "20261116_p15/run/ckpt_10.pt", HERE.parent / "20261120_switching_dissection/c14_run/ckpt_11.pt",
       HERE / "defar_run/ckpt_11.pt"]


@torch.no_grad()
def transitions(model, cfg, st, e, ts, draw):
    """midsignal.star_estimates' seeds and contexts. For (s0, s1) in ((2, 4), (4, 6)): the finest output at s1 on the real
    frame noised there (clean), on one deployed quarter-step from the real frame noised to s0 (drift), and at s0 on the
    real frame noised there (source) -> {s1: (clean, drift, source)} periphery estimates [W, 81, 16], and the real target
    frame's periphery [W, 81, 16]. The drifted state is a function of the s0 state and the history, so it holds no more
    information about the frame than the s0 state: the source estimate's error is the floor a perfect repair reaches."""
    c, Pe, rin, bp = st
    m, sc = cfg["p1"]["monitor"], cfg["edm"]["shortcut"]
    P, k, clamp = cfg["data"]["context_prefix"], sc["k_max"], cfg["perception"]["periphery"]["tanh_clamp"]
    _, e_step, d = M.shortcut_ladder(cfg); e_max = k.bit_length() - 1
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

    def est(s_, ec, p_, v_):
        sg = torch.full((W, P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = s_
        em = torch.full((W, P + 1), e_max, device=DEV, dtype=torch.long); em[:, -1] = ec
        xin = [x.clone() for x in xc]
        xin[0][:, -1], xin[1][:, -1], xin[2][:, -1] = p_, torch.zeros_like(xin[1][:, -1]), v_
        code = M.shortcut_code(sg, em, k)
        with TR.autocast(DEV, cfg):
            o = model.denoiser(tuple(xin), (code, code), pa, mem, fix, epi, valid)
        return o[0][:, -1].float(), o[2][:, -1].float()
    star = lambda s_: tuple((1 - s_ / k) * n + s_ / k * x for n, x in zip(eps, xr))
    out = {}
    for s0, s1 in ((2, 4), (4, 6)):
        p0, v0 = star(s0); xp, xv = est(s0, e_step, p0, v0)
        move = lambda zz, xx: zz + (xx.clamp(-clamp, clamp) - zz) / (1 - s0 / k) * d
        out[s1] = (est(s1, e_max, *star(s1))[0], est(s1, e_max, move(p0, xp), move(v0, xv))[0], est(s0, e_max, p0, v0)[0])
    return out, xr[0]


def gap(ck):
    ck = Path(ck)
    model, cfg = AI.load(ck); model.requires_grad_(False).eval()
    R = {"kind": C5.reader(model, cfg, False)}
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L); grids = G.episode_grids(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    zz = np.load(DL.EV.S2 / "model.npz"); mg, mp = zz["m_grid"], zz["m_pos"].astype(int)
    units = [(w, int(rows[w, 0]), int(rows[w, 3]) + 1, int(kind), int(i), int(j), bool(u[7])) for u in U
             for w, kind, i, j in [u[:4]] if 9 <= int(rows[w, 3]) + 1 <= L["ep_T"][int(rows[w, 0])] and DL.deleted(mg, mp, u)]
    cows = []                                                             # midsignal's day cows, its filter and order
    for e in sorted({x[1] for x in units}):
        for x in [u for u in units if u[1] == e]:
            t, kind, i, j, night = x[2:7]
            dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            r0, c0 = i - dpos[0], j - dpos[1]
            cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
            if cells and kind == 2 and not night:
                cows.append((x, cells, (r0 * 9 + c0) if 0 <= r0 < 7 and 0 <= c0 < 9 else None))
    acc = {s1: {"nb": [0., 0., 0.], "frame": [0., 0., 0.], "mass": [[], [], []]} for s1 in (4, 6)}   # clean, drift, source
    for e in sorted({u[0][1] for u in cows}):
        st = C2.episode(model, cfg, dl, e)
        us = [u for u in cows if u[0][1] == e]; ts = sorted({u[0][2] for u in us})
        real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1)
        g = grids[e]; n63 = lambda a: a.reshape(len(a), 63)
        for dr in range(DRAWS):
            out, xr = transitions(model, cfg, st, e, ts, dr)
            for s1, pair in out.items():
                err = [(q - xr).square().sum(-1) for q in pair]               # [W, 81] clean, drift, source
                for side in (0, 1, 2):
                    acc[s1]["frame"][side] += float(err[side].mean())
                for (w, _, t, kind, i, j, _), cells, own in us:
                    qi = ts.index(t)
                    for side in (0, 1, 2):
                        acc[s1]["nb"][side] += float(err[side][qi, cells].sum())
                    if own is None or int(real[qi, own]) != kind:
                        continue
                    kinds, view = n63(g["grid"])[t], n63(g["view"])[t]
                    cand = [cc for cc in range(63) if cc != own and kinds[cc] == 0 and view[cc] == view[own]]
                    if not cand:
                        continue
                    near = min(cand, key=lambda cc: max(abs(cc // 9 - own // 9), abs(cc % 9 - own % 9)))
                    P0 = st[1].p[0, t].float(); D = P0[own] - P0[near]; DD = float(D @ D)
                    same = [cc for cc in cells if cc != own and kinds[cc] == 0 and view[cc] == view[own]]
                    for side in (0, 1, 2):
                        x_ = pair[side][qi]
                        acc[s1]["mass"][side].append(float((x_[own] - P0[near]) @ D) / DD + sum(float((x_[cc] - P0[cc]) @ D) / DD for cc in same))
        print(ck.parent.name, e, flush=True)
    res = {"checkpoint": str(ck), "draws": DRAWS, "day_cows": len(cows)}
    for s1, a in acc.items():
        res[f"to_{s1}/8"] = {"neighbourhood_error": {"clean": a["nb"][0], "drift": a["nb"][1], "source_floor": a["nb"][2]},
                             "neighbourhood_repair_gap": a["nb"][1] / a["nb"][2] - 1, "frame_repair_gap": a["frame"][1] / a["frame"][2] - 1,
                             "mass_clean": float(np.mean(a["mass"][0])), "mass_drift": float(np.mean(a["mass"][1])),
                             "mass_source": float(np.mean(a["mass"][2])),
                             "drift_mass_below_half": float(np.mean(np.array(a["mass"][1]) < 0.5)), "mass_units_draws": len(a["mass"][1])}
    print(json.dumps(res), flush=True)
    return res


def main(cks):
    res = [gap(c) for c in cks]
    (HERE / "repair_gap.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if a.endswith(".pt")] or CKS)
