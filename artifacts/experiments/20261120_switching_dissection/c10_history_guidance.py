"""C10 (PLAN.md): history guidance (DFoT, Song et al. 2025, arXiv 2502.06764) at the persistence coin, on C5's units and
draws. x_hat = x_u + w (x_c - x_u) at each of the sampler's four physical-stage steps; the unconditional pass masks the
8 context frames by noise ("frames", DFoT's form) or removes the whole history ("all": no context frames, the record
and belief at their episode-start zeros), the action kept in both. The unguided arm is C4.samples itself (C5's draws).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c10_history_guidance [smoke | night]
`night` (added after the full run): the night units only, with C5's edge ring (unit row 1 / 5 or column 1 / 7; the
full run's ring, the map's outer ring after the camera move, held no units).
"""
import importlib, json, sys
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
ARMS = (("unguided", None, 1.0), ("frames_1.5", "frames", 1.5), ("frames_3", "frames", 3.0), ("all_1.5", "all", 1.5))


@torch.no_grad()
def guided(model, cfg, st, e, ts, draw, mode, w):
    """C4.samples (its seeds) with history guidance; mode None is C4.samples itself."""
    if mode is None:
        return C4.samples(model, cfg, st, e, ts, draw)
    c, Pe, rin, bp = st
    m, sc = cfg["p1"]["monitor"], cfg["edm"]["shortcut"]
    P, k, clamp = cfg["data"]["context_prefix"], sc["k_max"], cfg["perception"]["periphery"]["tanh_clamp"]
    ladder, e_step, d = M.shortcut_ladder(cfg)
    stack = lambda a: torch.stack([a[0, t - P:t + 1] for t in ts])
    x0 = [stack(a) for a in Pe.parts()]
    W = len(ts)
    ctx_noise, z, un = [], [], []
    for t in ts:
        g = torch.Generator(device=DEV).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(e) * 4_099 + int(t))
        ctx_noise.append(tuple(torch.randn(a.shape[1:], device=DEV, generator=g) for a in x0))
        g2 = torch.Generator(device=DEV).manual_seed(77_000_003 * (draw + 1) + int(e) * 4_099 + int(t))
        z.append((torch.randn(x0[0].shape[2:], device=DEV, generator=g2), torch.randn(x0[2].shape[2:], device=DEV, generator=g2)))
        g3 = torch.Generator(device=DEV).manual_seed(91_000_003 * (draw + 1) + int(e) * 4_099 + int(t))
        un.append(tuple(torch.randn(a.shape[1:], device=DEV, generator=g3) for a in x0))     # the masked history's noise
    ctx_noise, un = [torch.stack(q) for q in zip(*ctx_noise)], [torch.stack(q) for q in zip(*un)]
    p_t, v_t = torch.stack([q[0] for q in z]), torch.stack([q[1] for q in z])
    tc = sc["context_signal"] / k
    xc = [(1 - tc) * n + tc * x for x, n in zip(x0, ctx_noise)]            # context frames at the context signal
    mem = (model.belief.memory(stack(bp)), stack(rin))
    valid = stack(c["valid"])
    if mode == "all":                                                       # an episode's first frame: no context,
        mem_u, valid_u, xu = (model.belief.memory(torch.zeros_like(stack(bp))), torch.zeros_like(mem[1])), valid.clone(), xc
        valid_u[:, :-1] = False                                             # record and belief zero
    else:                                                                   # DFoT: the history masked by full noise
        mem_u, valid_u, xu = mem, valid, un
    em = torch.full((W, P + 1), k.bit_length() - 1, device=DEV, dtype=torch.long); em[:, -1] = e_step
    args = (stack(c["prev_action"]),)
    tail = (stack(Pe.fix), stack(c["episode"]))
    for s_, tau in ladder:
        sg = torch.full((W, P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = s_
        sgu = sg.clone()
        if mode == "frames":
            sgu[:, :-1] = 0
        outs = []
        for base, sgx, mm, vv in ((xc, sg, mem, valid), (xu, sgu, mem_u, valid_u)):
            xin = [x.clone() for x in base]
            xin[0][:, -1], xin[1][:, -1], xin[2][:, -1] = p_t, torch.zeros_like(xin[1][:, -1]), v_t
            code = M.shortcut_code(sgx, em, k)
            with TR.autocast(DEV, cfg):
                o = model.denoiser(tuple(xin), (code, code), *args, mm, *tail, vv)
            outs.append((o[0][:, -1].float(), o[2][:, -1].float()))
        (pc, vc), (pu, vu) = outs
        move = lambda zz, xx: zz + (xx.clamp(-clamp, clamp) - zz) / (1 - tau) * d
        p_t, v_t = move(p_t, pu + w * (pc - pu)), move(v_t, vu + w * (vc - vu))
    return p_t, v_t


def main(smoke=False, night_only=False):
    model, cfg = AI.load(C5.CK)
    model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, _, lookup, loff, _ = C1.semantics(L)
    dl = S.episodes(L)
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)                # C5's units, exactly
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    units = []
    for u in U:
        w, kind, i, j = u[:4]
        e, t0 = int(rows[w, 0]), int(rows[w, 3])
        if 9 <= t0 + 1 <= L["ep_T"][e]:
            r = DL.deleted(mg, mp, u)
            if r:
                units.append((w, e, t0 + 1, int(kind), int(i), int(j), np.mean([x[0] for x in r]), bool(u[7])))
    units = units[:60] if smoke else [x for x in units if x[7]] if night_only else units
    rec = {a[0]: [] for a in ARMS}
    fr = {a[0]: {"spurious": [], "err": {g: [0., 0] for g in C1.GROUPS}, "spread": {g: [0., 0] for g in C1.GROUPS}} for a in ARMS}
    check = None
    for e in sorted({x[1] for x in units}):
        st = C2.episode(model, cfg, dl, e)
        us = [x for x in units if x[1] == e]
        ts = sorted({x[2] for x in us})
        real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1)        # [W, 63]
        tok = st[1].p[0, ts].float()
        reg = {g: torch.as_tensor(groups[g][lookup[loff[e] + np.array(ts)]], device=DEV) for g in C1.GROUPS}   # [W, 81]
        if smoke and check is None:                                         # guided at w = 1 equals the sampler
            a, bb = guided(model, cfg, st, e, ts, 0, None, 1.0), guided(model, cfg, st, e, ts, 0, "frames", 1.0)
            check = max(float((x - y).abs().max()) for x, y in zip(a, bb))
        for name, mode, w in ARMS:
            ps, cls = [], []
            for dr in range(DRAWS):
                pd, vd = guided(model, cfg, st, e, ts, dr, mode, w)
                ps.append(pd); cls.append(C5.kind_probs(R, pd, vd).argmax(-1))
            ps, cls = torch.stack(ps), torch.stack(cls)                     # [D, W, 81, 16], [D, W, 63]
            for _, _, t, kind, i, j, _, night in us:
                qi = ts.index(t)
                dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
                r0, c0 = i - dpos[0], j - dpos[1]
                cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
                if not cells:
                    continue
                inside = 0 <= r0 < 7 and 0 <= c0 < 9
                own = r0 * 9 + c0 if inside else None
                rec[name].append({"kind": kind, "night": night, "edge": i in (1, 5) or j in (1, 7), "inside": inside,   # C5's ring (unit grid)
                                  "kept": (cls[:, qi][:, cells] == kind).any(-1).cpu().numpy(),
                                  "stay": (cls[:, qi, own] == kind).cpu().numpy() if inside else np.zeros(DRAWS, bool),
                                  "real_kept": bool((real[qi, cells] == kind).any()), "real_stay": bool(inside and real[qi, own] == kind)})
            f = fr[name]
            f["spurious"].append(((cls > 0) & (real[None] == 0)).sum(-1).float().mean(0).cpu().numpy())    # per frame, mean over draws
            err, spread = ((ps - tok[None]) ** 2).sum(-1), ps.var(0).sum(-1)
            for g, mk in reg.items():
                f["err"][g][0] += float((err * mk[None]).sum()); f["err"][g][1] += int(mk.sum()) * DRAWS
                f["spread"][g][0] += float((spread * mk).sum()); f["spread"][g][1] += int(mk.sum())
        print(e, len(rec["unguided"]), flush=True)
    KN = C5.KINDS
    res = {"units": len(rec["unguided"]), "draws": DRAWS, "smoke_check_w1_max_abs": check, "arms": {}}
    if not smoke and not night_only:
        z5 = np.load(HERE / "c5_units.npz")
        a = np.stack([x["kept"] for x in rec["unguided"]])
        res["unguided_reproduces_c5"] = float((a == z5["kept"]).mean()) if a.shape == z5["kept"].shape else f"shape {a.shape} vs {z5['kept'].shape}"
    for name, rr in rec.items():
        kinds = np.array([x["kind"] for x in rr]); night = np.array([x["night"] for x in rr]); edge = np.array([x["edge"] for x in rr])
        kept = np.stack([x["kept"] for x in rr]); realk = np.array([x["real_kept"] for x in rr])
        out = {"keep": {}, "stay": {}}
        for kn in (1, 2):
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
    tag = "_smoke" if smoke else "_night" if night_only else ""
    (HERE / f"c10_history_guidance{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez_compressed(HERE / f"c10_units{tag}.npz", **{f"{n}_kept": np.stack([x["kept"] for x in rr]) for n, rr in rec.items()},
                        kinds=np.array([x["kind"] for x in rec["unguided"]]), edge=np.array([x["edge"] for x in rec["unguided"]]))
    print("check", check, res.get("unguided_reproduces_c5"), flush=True)


if __name__ == "__main__":
    main("smoke" in sys.argv[1:], "night" in sys.argv[1:])
