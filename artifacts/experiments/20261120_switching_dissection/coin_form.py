"""The day coin's error form (PLAN.md; the user 11-21): (1) whether later steps repair a single step's loss of the
cow (starts from the real frame noised to each ladder level, with the draw's own noise), (2) what the native state's
deviation from that same-noise real path is made of at the cow's cell. P1.5 checkpoint 10, C5's day units, the native
K = 4 sampler, C5's seeds, S2's readers.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.coin_form [smoke]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C2, C4, C5, DL = (importlib.import_module("artifacts.experiments.20261120_switching_dissection." + n) for n in ("c2_eye", "c4_sample", "c5_calibration", "deletions"))
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda")


@torch.no_grad()
def paths(model, cfg, st, e, ts, draw):
    """C4.samples' arithmetic and seeds (C15's remedy form). Returns the native path's states [4] and estimates [4], its
    final, and for each start level i = 1..3 the estimate at step i from z*_i and the final continued from z*_i."""
    c, Pe, rin, bp = st
    m, sc = cfg["p1"]["monitor"], cfg["edm"]["shortcut"]
    P, k, clamp = cfg["data"]["context_prefix"], sc["k_max"], cfg["perception"]["periphery"]["tanh_clamp"]
    ladder, e_step, d = M.shortcut_ladder(cfg)
    stack = lambda a: torch.stack([a[0, t - P:t + 1] for t in ts])
    x0 = [stack(a) for a in Pe.parts()]
    W = len(ts)
    ctx_noise, z = [], []
    for t in ts:
        g = torch.Generator(device=DEV).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(e) * 4_099 + int(t))
        ctx_noise.append(tuple(torch.randn(a.shape[1:], device=DEV, generator=g) for a in x0))
        g2 = torch.Generator(device=DEV).manual_seed(77_000_003 * (draw + 1) + int(e) * 4_099 + int(t))
        z.append((torch.randn(x0[0].shape[2:], device=DEV, generator=g2), torch.randn(x0[2].shape[2:], device=DEV, generator=g2)))
    ctx_noise = [torch.stack(q) for q in zip(*ctx_noise)]
    eps = (torch.stack([q[0] for q in z]), torch.stack([q[1] for q in z]))
    xr = (x0[0][:, -1].float(), x0[2][:, -1].float())                     # the real target frame (periphery, lighting)
    tc = sc["context_signal"] / k
    xc = [(1 - tc) * n + tc * x for x, n in zip(x0, ctx_noise)]
    mem, valid = (model.belief.memory(stack(bp)), stack(rin)), stack(c["valid"])
    em = torch.full((W, P + 1), k.bit_length() - 1, device=DEV, dtype=torch.long); em[:, -1] = e_step
    pa, fix, epi = stack(c["prev_action"]), stack(Pe.fix), stack(c["episode"])

    def estimate(i, p_, v_):
        sg = torch.full((W, P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = ladder[i][0]
        xin = [x.clone() for x in xc]
        xin[0][:, -1], xin[1][:, -1], xin[2][:, -1] = p_, torch.zeros_like(xin[1][:, -1]), v_
        code = M.shortcut_code(sg, em, k)
        with TR.autocast(DEV, cfg):
            o = model.denoiser(tuple(xin), (code, code), pa, mem, fix, epi, valid)
        return o[0][:, -1], o[2][:, -1]

    def step(i, p_, v_):
        tau = ladder[i][1]; pc, vc = estimate(i, p_, v_)
        move = lambda zz, xx: zz + (xx.float().clamp(-clamp, clamp) - zz) / (1 - tau) * d
        return move(p_, pc), move(v_, vc), (pc.float(), vc.float())

    star = lambda i: tuple((1 - ladder[i][1]) * n + ladder[i][1] * x for n, x in zip(eps, xr))
    states, ests = [], []
    p_t, v_t = eps
    for i in range(len(ladder)):
        states.append((p_t, v_t)); p_t, v_t, est = step(i, p_t, v_t); ests.append(est)
    native = (p_t, v_t)
    starts = {}
    for i in range(1, len(ladder)):
        p_s, v_s = star(i); ests_i = []
        for j in range(i, len(ladder)):
            p_s, v_s, est = step(j, p_s, v_s); ests_i.append(est)
        starts[i] = (ests_i[0], (p_s, v_s), ests_i)                          # (first estimate, final, every estimate)
    return {"states": states, "estimates": ests, "native": native, "starts": starts, "star": [star(i) for i in range(len(ladder))], "eps": eps}


def main(smoke=False):
    model, cfg = AI.load(C5.CK); model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    zz = np.load(DL.EV.S2 / "model.npz"); mg, mp = zz["m_grid"], zz["m_pos"].astype(int)
    units = [(w, int(rows[w, 0]), int(rows[w, 3]) + 1, int(kind), int(i), int(j), bool(u[7])) for u in U
             for w, kind, i, j in [u[:4]] if 9 <= int(rows[w, 3]) + 1 <= L["ep_T"][int(rows[w, 0])] and DL.deleted(mg, mp, u)]
    c5 = np.load(HERE / "c5_units.npz")["kept"]                           # C5's order = these units with cells
    order, n = [], 0
    for e in sorted({x[1] for x in units}):
        for x in [u for u in units if u[1] == e]:
            t, kind, i, j = x[2:6]
            dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            r0, c0 = i - dpos[0], j - dpos[1]
            cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
            if cells:
                order.append((n, x, cells, (r0 * 9 + c0) if 0 <= r0 < 7 and 0 <= c0 < 9 else None)); n += 1
    day = [u for u in order if not u[1][6]]
    if smoke:
        e2 = sorted({u[1][1] for u in day})[:3]; day = [u for u in day if u[1][1] in e2]
    rec, checks = [], {"native_vs_sampler": 0.0, "A0_vs_c5": [0, 0]}
    for e in sorted({u[1][1] for u in day}):
        st = C2.episode(model, cfg, dl, e)
        us = [u for u in day if u[1][1] == e]; ts = sorted({u[1][2] for u in us})
        real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1)
        out = [paths(model, cfg, st, e, ts, dr) for dr in range(C5.DRAWS)]
        if smoke and checks["native_vs_sampler"] == 0.0:
            a = C4.samples(model, cfg, st, e, ts, 0)
            checks["native_vs_sampler"] = max(float((x - y).abs().max()) for x, y in zip(a, out[0]["native"]))
        kind_of = lambda pv: C5.kind_probs(R, pv[0], pv[1]).argmax(-1)       # [W, 63]
        nat = torch.stack([kind_of(q["native"]) for q in out])                # [D, W, 63]
        nat_est = torch.stack([torch.stack([kind_of(est) for est in q["estimates"]]) for q in out])   # [D, 4, W, 63]
        st_fin = {i: torch.stack([kind_of(q["starts"][i][1]) for q in out]) for i in (1, 2, 3)}
        st_est = {i: torch.stack([kind_of(q["starts"][i][0]) for q in out]) for i in (1, 2, 3)}
        for n_, (w, _, t, kind, i, j, _), cells, own in us:
            qi = ts.index(t)
            kept = lambda cls: (cls[..., qi, :][..., cells] == kind).any(-1).cpu().numpy()
            r = {"n": n_, "kind": kind, "A0": kept(nat), "drop0": ~kept(nat_est[:, 0]),
                 **{f"A{i_}": kept(st_fin[i_]) for i_ in (1, 2, 3)}, **{f"drop{i_}": ~kept(st_est[i_]) for i_ in (1, 2, 3)}}
            checks["A0_vs_c5"][0] += int((r["A0"] == c5[n_]).sum()); checks["A0_vs_c5"][1] += C5.DRAWS
            if own is not None and int(real[qi, own]) == kind:                 # (2): the cow on its own cell in the real frame
                x = st[1].p[0, t, own].float()
                dele = ~r["A0"]
                absent = torch.stack([q["native"][0][qi, own] for q, dd in zip(out, dele) if dd]).mean(0) if dele.any() else None
                if absent is not None:
                    absent = absent - (absent @ x) / (x @ x) * x
                comp = []
                for dr, q in enumerate(out):
                    row = []
                    for lv in (1, 2, 3):
                        zt, zs = q["states"][lv][0][qi], q["star"][lv][0][qi]
                        D_ = (zt - zs)[own]; e0 = q["eps"][0][qi, own]
                        basis = [x, e0] + ([absent] if absent is not None and absent.norm() > 1e-6 else [])
                        B = torch.stack(basis, 1)                               # [16, 2 or 3]
                        coef = torch.linalg.lstsq(B, D_[:, None]).solution[:, 0]
                        parts = [float((coef[k_] * basis[k_]).square().sum()) for k_ in range(len(basis))]
                        tot = float(D_.square().sum()) + 1e-12
                        row.append([float(coef[0]), float(coef[1]), parts[0] / tot, parts[1] / tot,
                                    parts[2] / tot if len(parts) > 2 else np.nan, tot / ((1 - [0, .25, .5, .75][lv]) ** 2 * 16),
                                    float(zt.square().sum() / zs.square().sum())])
                    comp.append(row)
                r["decomp"] = np.array(comp)                                    # [D, 3 levels, 7]
            rec.append(r)
        print(e, len(rec), flush=True)
    res = {"units": len(rec), "draws": C5.DRAWS, "smoke_checks": checks, "by_kind": {}}
    for kn in (2, 1, 3):
        rr = [r for r in rec if r["kind"] == kn]
        if not rr:
            continue
        A = {i: 1 - np.mean([r[f"A{i}"].mean() for r in rr]) for i in range(4)}; A[4] = 0.0
        drop = {i: float(np.mean([r[f"drop{i}"].mean() for r in rr])) for i in range(4)}
        out = {"units": len(rr), "deletion_from_start": A, "drop": drop,
               "delta": {i: A[i - 1] - A[i] for i in range(1, 5)},
               "repair": {i: (1 - (A[i] - A[i + 1]) / drop[i]) if drop[i] > 0 else None for i in range(4)},
               "delta_se": {i: float(np.std([(1 - r[f"A{i-1}"].mean()) - ((1 - r[f"A{i}"].mean()) if i < 4 else 0) for r in rr], ddof=1) / np.sqrt(len(rr))) for i in range(1, 5)}}
        dd = [r for r in rr if "decomp" in r]
        if dd:
            Dm = np.concatenate([r["decomp"] for r in dd]); fin = np.concatenate([~r["A0"] for r in dd])   # [N draws, 3, 7]
            names = ["coef_x", "coef_eps0", "share_x", "share_eps0", "share_absent", "deviation_over_noise_power", "frame_energy_ratio"]
            for tag, sel in (("deleted", fin), ("kept", ~fin)):
                if sel.sum():
                    out[f"decomposition_{tag}"] = {f"tau_{lv}/8": {nm: float(np.nanmean(Dm[sel, li, ni])) for ni, nm in enumerate(names)}
                                                   for li, lv in enumerate((2, 4, 6))}
            out["decomposition_draws"] = {"deleted": int(fin.sum()), "kept": int((~fin).sum())}
        res["by_kind"][C5.KINDS[kn]] = out
        print(C5.KINDS[kn], json.dumps(out)[:1500], flush=True)
    (HERE / f"coin_form{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("checks", checks, flush=True)


if __name__ == "__main__":
    main("smoke" in sys.argv[1:])
