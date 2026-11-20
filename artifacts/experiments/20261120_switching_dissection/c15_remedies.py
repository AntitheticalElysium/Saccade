"""C15 (PLAN.md): sampling-time remedies for the day coin on C5's units and draws (P1.5 checkpoint 10, the default K = 4
sampler, physical stage). A0: C4.samples itself. A1: whole-history guidance (C10's "all" unconditional) limited to the
mid-chain interval, the third and fourth ladder steps (Kynkaanniemi et al. 2024; w = 2.0). A2: restart sampling (Xu et
al. 2023): after the third step (tau 6/8) re-noise to tau 2/8 and re-solve the second and third steps, K = 2 times.
Readings as C10's.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c15_remedies [smoke] [CKPT]
"""
import importlib, json, math, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c1")
C2 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_eye")
C4 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c4_sample")
C5 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c5_calibration")
DL = importlib.import_module("artifacts.experiments.20261120_switching_dissection.deletions")
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda")
DRAWS = 8
ARMS = (("A0_unguided", {}),
        ("A1_interval_guidance_w2", {"guide_steps": (2, 3), "w": 2.0}),   # steps starting at tau 4/8, 6/8: sigma 1.0, 0.33
        ("A2_restart_K2", {"restarts": 2}))                                # re-noise 6/8 -> 2/8, re-solve steps 2-3, twice


@torch.no_grad()
def remedy(model, cfg, st, e, ts, draw, guide_steps=(), w=1.0, restarts=0, lo=1, hi=3):
    """C4.samples (its seeds and arithmetic) with, at the ladder steps `guide_steps`, x_hat = x_u + w (x_c - x_u) for the
    whole-history unconditional (no context frames, record and belief at their episode-start zeros, action kept);
    weight 1 elsewhere (the conditional alone). With `restarts`: after the steps before ladder[hi], re-noise from tau_hi
    to tau_lo = ladder[lo] (EDM sigma = (1 - tau) / tau: x = z / tau_hi + sqrt(sigma_lo^2 - sigma_hi^2) eps', z = tau_lo x)
    and re-solve steps lo..hi-1, `restarts` times, then the rest."""
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
    p_t, v_t = torch.stack([q[0] for q in z]), torch.stack([q[1] for q in z])
    tc = sc["context_signal"] / k
    xc = [(1 - tc) * n + tc * x for x, n in zip(x0, ctx_noise)]
    mem, valid = (model.belief.memory(stack(bp)), stack(rin)), stack(c["valid"])
    mem_u = (model.belief.memory(torch.zeros_like(stack(bp))), torch.zeros_like(mem[1]))
    valid_u = valid.clone(); valid_u[:, :-1] = False
    em = torch.full((W, P + 1), k.bit_length() - 1, device=DEV, dtype=torch.long); em[:, -1] = e_step
    pa, fix, epi = stack(c["prev_action"]), stack(Pe.fix), stack(c["episode"])

    def evaluate(s_, p_, v_, mm, vv):
        sg = torch.full((W, P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = s_
        xin = [x.clone() for x in xc]
        xin[0][:, -1], xin[1][:, -1], xin[2][:, -1] = p_, torch.zeros_like(xin[1][:, -1]), v_
        code = M.shortcut_code(sg, em, k)
        with TR.autocast(DEV, cfg):
            o = model.denoiser(tuple(xin), (code, code), pa, mm, fix, epi, vv)
        return o[0][:, -1], o[2][:, -1]

    def step(i, p_, v_):
        s_, tau = ladder[i]
        pc, vc = evaluate(s_, p_, v_, mem, valid)
        if i in guide_steps:
            pu, vu = evaluate(s_, p_, v_, mem_u, valid_u)
            pc, vc = pu.float() + w * (pc.float() - pu.float()), vu.float() + w * (vc.float() - vu.float())
        move = lambda zz, xx: zz + (xx.float().clamp(-clamp, clamp) - zz) / (1 - tau) * d
        return move(p_, pc), move(v_, vc)

    order = list(range(hi)) if restarts else list(range(len(ladder)))
    for i in order:
        p_t, v_t = step(i, p_t, v_t)
    if restarts:
        t_lo, t_hi = ladder[lo][1], ladder[hi][1]
        s_lo, s_hi = (1 - t_lo) / t_lo, (1 - t_hi) / t_hi
        for r in range(restarts):
            gr = torch.Generator(device=DEV).manual_seed(33_000_003 * (draw + 1) + 7_919 * r + int(e) * 4_099 + int(ts[0]))
            add = math.sqrt(s_lo ** 2 - s_hi ** 2)
            p_t = t_lo * (p_t / t_hi + add * torch.randn(p_t.shape, device=DEV, generator=gr))
            v_t = t_lo * (v_t / t_hi + add * torch.randn(v_t.shape, device=DEV, generator=gr))
            for i in range(lo, hi):
                p_t, v_t = step(i, p_t, v_t)
        for i in range(hi, len(ladder)):
            p_t, v_t = step(i, p_t, v_t)
    return p_t, v_t


def main(smoke=False, ck=None):
    ck = Path(ck) if ck else C5.CK
    model, cfg = AI.load(ck)
    model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg) if ck == C5.CK else {"kind": C5.reader(model, cfg, False)}
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, _, lookup, loff, _ = C1.semantics(L)
    dl = S.episodes(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)        # C5's units, exactly
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    units = []
    for u in U:
        w, kind, i, j = u[:4]
        e, t0 = int(rows[w, 0]), int(rows[w, 3])
        if 9 <= t0 + 1 <= L["ep_T"][e] and DL.deleted(mg, mp, u):
            units.append((w, e, t0 + 1, int(kind), int(i), int(j), bool(u[7])))
    units = units[:60] if smoke else units
    rec = {a[0]: [] for a in ARMS}
    fr = {a[0]: {"spurious": [], "err": {g: [0., 0] for g in C1.GROUPS}, "spread": {g: [0., 0] for g in C1.GROUPS}} for a in ARMS}
    checks = {}
    for e in sorted({x[1] for x in units}):
        st = C2.episode(model, cfg, dl, e)
        us = [x for x in units if x[1] == e]
        ts = sorted({x[2] for x in us})
        real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1)
        tok = st[1].p[0, ts].float()
        reg = {g: torch.as_tensor(groups[g][lookup[loff[e] + np.array(ts)]], device=DEV) for g in C1.GROUPS}
        if smoke and not checks:                                            # A0 is the sampler; w = 1 and K = 0 change nothing
            a = C4.samples(model, cfg, st, e, ts, 0)
            for name, kw in (("A0_vs_sampler", {}), ("guidance_w1", {"guide_steps": (2, 3), "w": 1.0})):
                bb = remedy(model, cfg, st, e, ts, 0, **kw)
                checks[name] = max(float((x - y).abs().max()) for x, y in zip(a, bb))
        for name, kw in ARMS:
            ps, cls = [], []
            for dr in range(DRAWS):
                pd, vd = remedy(model, cfg, st, e, ts, dr, **kw)
                ps.append(pd); cls.append(C5.kind_probs(R, pd, vd).argmax(-1))
            ps, cls = torch.stack(ps), torch.stack(cls)
            for _, _, t, kind, i, j, night in us:
                qi = ts.index(t)
                dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
                r0, c0 = i - dpos[0], j - dpos[1]
                cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
                if not cells:
                    continue
                inside = 0 <= r0 < 7 and 0 <= c0 < 9
                own = r0 * 9 + c0 if inside else None
                rec[name].append({"kind": kind, "night": night, "edge": i in (1, 5) or j in (1, 7), "inside": inside,
                                  "kept": (cls[:, qi][:, cells] == kind).any(-1).cpu().numpy(),
                                  "stay": (cls[:, qi, own] == kind).cpu().numpy() if inside else np.zeros(DRAWS, bool),
                                  "real_kept": bool((real[qi, cells] == kind).any()), "real_stay": bool(inside and real[qi, own] == kind)})
            f = fr[name]
            f["spurious"].append(((cls > 0) & (real[None] == 0)).sum(-1).float().mean(0).cpu().numpy())
            err, spread = ((ps - tok[None]) ** 2).sum(-1), ps.var(0).sum(-1)
            for g, mk in reg.items():
                f["err"][g][0] += float((err * mk[None]).sum()); f["err"][g][1] += int(mk.sum()) * DRAWS
                f["spread"][g][0] += float((spread * mk).sum()); f["spread"][g][1] += int(mk.sum())
        print(e, len(rec["A0_unguided"]), flush=True)
    KN = C5.KINDS
    res = {"units": len(rec["A0_unguided"]), "draws": DRAWS, "checkpoint": str(ck), "smoke_checks": checks, "arms": {}}
    if not smoke and ck == C5.CK:
        a = np.stack([x["kept"] for x in rec["A0_unguided"]]); z5 = np.load(HERE / "c5_units.npz")
        res["A0_reproduces_c5"] = float((a == z5["kept"]).mean()) if a.shape == z5["kept"].shape else f"shape {a.shape} vs {z5['kept'].shape}"
    for name, rr in rec.items():
        kinds = np.array([x["kind"] for x in rr]); night = np.array([x["night"] for x in rr]); edge = np.array([x["edge"] for x in rr])
        kept = np.stack([x["kept"] for x in rr]); realk = np.array([x["real_kept"] for x in rr])
        out = {"keep": {}, "stay": {}}
        for kn in (1, 2, 3):
            for tag, sel in (("day", ~night), ("night", night), ("night_edge", night & edge), ("night_inner", night & ~edge)):
                mk = (kinds == kn) & sel
                if mk.any():
                    out["keep"][f"{KN[kn]}|{tag}"] = {"world": float(kept[mk].mean()), "real_frame": float(realk[mk].mean()), "units": int(mk.sum())}
        ins = np.array([x["inside"] for x in rr]); stay = np.stack([x["stay"] for x in rr]); reals = np.array([x["real_stay"] for x in rr])
        for kn in (1, 2):
            mk = (kinds == kn) & ins & realk
            out["stay"][KN[kn]] = {"world_share_of_kept_draws": float(stay[mk].sum() / max(kept[mk].sum(), 1)), "real_frame": float(reals[mk].mean()),
                                   "units": int(mk.sum())}
        f = fr[name]
        out["spurious_mobs_per_frame"] = float(np.concatenate(f["spurious"]).mean())
        out["token_error_by_region"] = {g: v[0] / max(v[1], 1) for g, v in f["err"].items()}
        out["spread_by_region"] = {g: v[0] / max(v[1], 1) for g, v in f["spread"].items()}
        res["arms"][name] = out
        print(name, json.dumps(out), flush=True)
    tag = ("_smoke" if smoke else "") + ("" if ck == C5.CK else "_" + ck.parent.parent.name + "_" + ck.stem)
    (HERE / f"c15_remedies{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez_compressed(HERE / f"c15_units{tag}.npz", **{f"{n}_kept": np.stack([x["kept"] for x in rr]) for n, rr in rec.items()},
                        kinds=np.array([x["kind"] for x in rec["A0_unguided"]]), night=np.array([x["night"] for x in rec["A0_unguided"]]))
    print("checks", checks, res.get("A0_reproduces_c5"), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main("smoke" in a, next((x for x in a if x.endswith(".pt")), None))
