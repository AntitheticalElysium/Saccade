"""C13 (PLAN.md): where along the signal levels the world loses a visible mob. (a) the real target noised to each
finest-grid level, one evaluation; (b) the K = 8 sampler's own path (C4.samples, flow term only); (c) the zero-signal
estimate's spread over draws at the mob's cell; (d) the share of the target's periphery error on entity cells, by
level. C5's units, real past, context at the context signal.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c13_levels [smoke] [CKPT own]
"""
import copy, importlib, json, sys
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
KINDS = C5.KINDS


@torch.no_grad()
def at_level(model, cfg, st, e, ts, j, draw):
    """(a) The real target frame noised to level j / k_max, one evaluation at the finest step (the flow term's own
    training form); context as C4.samples (its seeds). Returns the x-estimate (p [W, 81, 16], v [W, 8])."""
    c, Pe, rin, bp = st
    m, sc = cfg["p1"]["monitor"], cfg["edm"]["shortcut"]
    P, k = cfg["data"]["context_prefix"], sc["k_max"]
    stack = lambda a: torch.stack([a[0, t - P:t + 1] for t in ts])
    x0 = [stack(a) for a in Pe.parts()]
    ctx_noise, z = [], []
    for t in ts:
        g = torch.Generator(device=DEV).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(e) * 4_099 + int(t))
        ctx_noise.append(tuple(torch.randn(a.shape[1:], device=DEV, generator=g) for a in x0))
        g2 = torch.Generator(device=DEV).manual_seed(55_000_003 * (draw + 1) + 911 * j + int(e) * 4_099 + int(t))
        z.append((torch.randn(x0[0].shape[2:], device=DEV, generator=g2), torch.randn(x0[2].shape[2:], device=DEV, generator=g2)))
    ctx_noise = [torch.stack(q) for q in zip(*ctx_noise)]
    tc, tau = sc["context_signal"] / k, j / k
    xin = [(1 - tc) * n + tc * x.float() for x, n in zip(x0, ctx_noise)]
    xin[0][:, -1] = (1 - tau) * torch.stack([q[0] for q in z]) + tau * x0[0][:, -1].float()
    xin[1][:, -1] = 0
    xin[2][:, -1] = (1 - tau) * torch.stack([q[1] for q in z]) + tau * x0[2][:, -1].float()
    sg = torch.full((len(ts), P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = j
    code = M.shortcut_code(sg, torch.full_like(sg, k.bit_length() - 1), k)
    with TR.autocast(DEV, cfg):
        o = model.denoiser(tuple(xin), (code, code), stack(c["prev_action"]), (model.belief.memory(stack(bp)), stack(rin)),
                           stack(Pe.fix), stack(c["episode"]), stack(c["valid"]))
    return o[0][:, -1].float(), o[2][:, -1].float()


def main(smoke=False, ck=None, own=False):
    ck = Path(ck) if ck else C5.CK
    model, cfg = AI.load(ck)
    model.requires_grad_(False).eval()
    R = {"kind": C5.reader(model, cfg, smoke)} if own else S2.load_readers(model, cfg)
    cfg8 = copy.deepcopy(cfg); cfg8["edm"]["sampler"] = dict(cfg8["edm"]["sampler"], steps=8)
    k = cfg["edm"]["shortcut"]["k_max"]
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, _, lookup, loff, cover = C1.semantics(L)
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
    rec = []
    err = np.zeros((k, 3))                                              # per level: entity error, all error, frames
    for e in sorted({x[1] for x in units}):
        st = C2.episode(model, cfg, dl, e)
        us = [x for x in units if x[1] == e]
        ts = sorted({x[2] for x in us})
        real = C5.kind_probs(R, st[1].p[0, ts].float(), st[1].v[0, ts].float()).argmax(-1).cpu()   # [W, 63]
        tok = st[1].p[0, ts].float()                                    # [W, 81, 16]
        ent = torch.as_tensor(cover["any_entity"][lookup[loff[e] + np.array(ts)]], device=DEV)     # [W, 81]
        lev = []                                                        # [levels, draws, W, 63] classes; level-0 tokens
        own0 = []
        for j in range(k):
            per = []
            for d in range(DRAWS):
                p_, v_ = at_level(model, cfg, st, e, ts, j, d)
                per.append(C5.kind_probs(R, p_, v_).cpu())
                se = (p_ - tok).square().sum(-1)                        # [W, 81]
                err[j] += [float((se * ent).sum()), float(se.sum()), len(ts)]
                if j == 0:
                    own0.append(p_.cpu())
            lev.append(torch.stack(per))
        lev = torch.stack(lev)                                          # [k, D, W, 63, classes]
        own0 = torch.stack(own0)                                        # [D, W, 81, 16]
        smp = []
        for d in range(DRAWS):
            tr = []
            pd, vd = C4.samples(model, cfg8, st, e, ts, d, tr)
            smp.append(torch.stack([C5.kind_probs(R, a, v).cpu() for a, v in tr + [(pd, vd)]]))
        smp = torch.stack(smp)                                          # [D, steps + final, W, 63, classes]
        for _, _, t, kind, i, j, night in us:
            qi = ts.index(t)
            dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            r0, c0 = i - dpos[0], j - dpos[1]
            cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
            if not cells:
                continue
            inside = 0 <= r0 < 7 and 0 <= c0 < 9
            own = r0 * 9 + c0 if inside else None
            x = own0[:, qi, own] if inside else None                    # [D, 16] level-0 estimates at the mob's cell
            rec.append({"kind": kind, "night": night, "inside": inside,
                        "stayed": bool(inside and real[qi, own] == kind), "real_kept": bool((real[qi, cells] == kind).any()),
                        "level_kept": (lev[:, :, qi][:, :, cells].argmax(-1) == kind).any(-1).numpy(),      # [k, D]
                        "sampler_kept": (smp[:, :, qi][:, :, cells].argmax(-1) == kind).any(-1).numpy(),   # [D, k + 1]
                        "level_P": lev[:, :, qi][:, :, cells, kind].amax(-1).numpy(), "level_Psum": lev[:, :, qi][:, :, cells, kind].sum(-1).numpy(),
                        "sampler_P": smp[:, :, qi][:, :, cells, kind].amax(-1).numpy(), "sampler_Psum": smp[:, :, qi][:, :, cells, kind].sum(-1).numpy(),
                        "spread0": float((x - x.mean(0)).square().sum(-1).mean()) if inside else np.nan,
                        "dist0": float((x.mean(0) - tok[qi, own].cpu()).square().sum()) if inside else np.nan})
        print(e, len(rec), flush=True)
    kinds = np.array([x["kind"] for x in rec]); night = np.array([x["night"] for x in rec])
    lk = np.stack([x["level_kept"] for x in rec]); sk = np.stack([x["sampler_kept"] for x in rec])   # [U, k, D], [U, D, k+1]
    w = np.array([0.9 * j / k + 0.1 for j in range(k)])
    res = {"units": len(rec), "draws": DRAWS, "checkpoint": str(ck), "reader": "own" if own else "s2", "by_group": {},
           "objective": {"entity_share_of_error_by_level": (err[:, 0] / err[:, 1]).tolist(),
                         "ramp_weight": w.tolist(),
                         "level_share_of_weighted_objective": (w * err[:, 1] / (w * err[:, 1]).sum()).tolist(),
                         "entity_share_of_weighted_objective": float((w * err[:, 0]).sum() / (w * err[:, 1]).sum())}}
    for kn in (2, 1, 3):
        for tag, sel in (("day", ~night), ("night", night)):
            m = (kinds == kn) & sel
            if not m.any():
                continue
            g = {"units": int(m.sum()), "real_frame_kept": float(np.mean([rec[q]["real_kept"] for q in np.nonzero(m)[0]])),
                 "real_path_kept_by_level": lk[m].mean((0, 2)).round(4).tolist(),
                 "sampler_kept_by_step": sk[m].mean((0, 1)).round(4).tolist()}
            st_ = m & np.array([x["stayed"] for x in rec])
            if st_.any():
                sp, di = np.array([rec[q]["spread0"] for q in np.nonzero(st_)[0]]), np.array([rec[q]["dist0"] for q in np.nonzero(st_)[0]])
                g["zero_signal_at_own_cell_stayed"] = {"units": int(st_.sum()), "spread_over_draws": float(sp.mean()), "mean_to_real": float(di.mean()),
                                                       "level0_lost": float(1 - lk[st_][:, 0].mean())}
            arr = lambda key: np.stack([rec[q][key] for q in np.nonzero(m)[0]])
            g["real_path_P_by_level"] = arr("level_P").mean((0, 2)).round(4).tolist()          # the reader's P(kind), best cell
            g["real_path_Psum_by_level"] = arr("level_Psum").mean((0, 2)).round(4).tolist()    # summed over the neighbourhood
            g["sampler_P_by_step"] = arr("sampler_P").mean((0, 1)).round(4).tolist()
            g["sampler_Psum_by_step"] = arr("sampler_Psum").mean((0, 1)).round(4).tolist()
            stayed = np.array([rec[q]["stayed"] for q in np.nonzero(m)[0]])
            for nm, sel in (("stayed", stayed), ("moved", ~stayed)):
                if sel.any():
                    g[f"split_{nm}"] = {"units": int(sel.sum()), "level0_lost": float(1 - lk[m][sel][:, 0].mean()),
                                        "final_deleted": float(1 - sk[m][sel][:, :, -1].mean())}
            fin = 1 - sk[m][:, :, -1].mean(1)                           # per-unit final deletion (sampler)
            l0 = 1 - lk[m][:, 0].mean(1)                                # per-unit level-0 loss (real path = sampler at step 1)
            g["corr_unit_level0_loss_vs_final_deletion"] = float(np.corrcoef(l0, fin)[0, 1]) if fin.std() > 0 and l0.std() > 0 else None
            res["by_group"][f"{KINDS[kn]}|{tag}"] = g
    np.savez_compressed(HERE / f"c13_units{'_smoke' if smoke else ''}{'_' + ck.stem if ck != C5.CK else ''}.npz", level_kept=lk, sampler_kept=sk,
                        kinds=kinds, night=night, stayed=np.array([x["stayed"] for x in rec]))
    (HERE / f"c13_levels{'_smoke' if smoke else ''}{'_' + ck.stem if ck != C5.CK else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main("smoke" in a, next((x for x in a if x.endswith(".pt")), None), "own" in a)
