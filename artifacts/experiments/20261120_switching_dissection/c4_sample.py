"""C4 (PLAN.md): the sample-eye. Surprise is the squared distance between the real frame's tokens and the world's own
K-step sample of that frame given its real past. The sample is the sampler's periphery stage run as monitor-style
window forwards: context frames at the context signal, the target from noise over the shortcut ladder.

Readings, on C2's DEV frames and S2 start mobs (two draws each):
- coverage under the chooser's policy with contingency off;
- surprise shares;
- the deletion link: is the eye on the mob when its own draw lost it?

Plus its cost at the trainer's shapes (`c2_cost.inputs`; K evaluations and a context pass per frame).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c4_sample [N_EPISODES]
"""
import importlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c1")
C2 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_eye")
CC = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_cost")
DL = importlib.import_module("artifacts.experiments.20261120_switching_dissection.deletions")
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
HERE = Path(__file__).resolve().parent
CK = HERE.parent / "20261116_p15/run/ckpt_10.pt"
DEV = torch.device("cuda")
WIN = C.WIN.numpy()
DRAWS = 2


@torch.no_grad()
def samples(model, cfg, st, e, ts, draw, trace=None):
    """The world's K-step periphery / lighting sample of target frames ts given their real past -> p [W, 81, 16].
    A list `trace` collects each step's clean estimate (p, v) (C5)."""
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
    xc = [(1 - tc) * n + tc * x for x, n in zip(x0, ctx_noise)]            # context frames at the context signal
    mem = (model.belief.memory(stack(bp)), stack(rin))
    em = torch.full((W, P + 1), k.bit_length() - 1, device=DEV, dtype=torch.long); em[:, -1] = e_step
    for s_, tau in ladder:
        sg = torch.full((W, P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = s_
        xin = [x.clone() for x in xc]
        xin[0][:, -1], xin[1][:, -1], xin[2][:, -1] = p_t, torch.zeros_like(xin[1][:, -1]), v_t
        code = M.shortcut_code(sg, em, k)
        with TR.autocast(DEV, cfg):
            o = model.denoiser(tuple(xin), (code, code), stack(c["prev_action"]), mem, stack(Pe.fix), stack(c["episode"]), stack(c["valid"]))
        move = lambda zz, xx: zz + (xx.float().clamp(-clamp, clamp) - zz) / (1 - tau) * d
        if trace is not None:
            trace.append((o[0][:, -1].float(), o[2][:, -1].float()))
        p_t, v_t = move(p_t, o[0][:, -1]), move(v_t, o[2][:, -1])
    return p_t, v_t


@torch.no_grad()
def eye_pass_sample(model, cfg, c, P, rin, bp, prefix, dev):
    """c2_cost.eye_pass with the sampler's K evaluations per target frame (timing only)."""
    den, k = model.denoiser, cfg["edm"]["shortcut"]["k_max"]
    ladder, e, d = M.shortcut_ladder(cfg)
    B, L = P.p.shape[:2]
    g = torch.Generator(device=dev).manual_seed(7)
    cache = den.new_cache()
    with TR.autocast(dev, cfg):
        for t in range(L):
            mem = S.cond(model.belief.memory(bp[:, t:t + 1]), rin[:, t:t + 1])
            if t >= prefix:
                p = torch.randn(P.p[:, t:t + 1].shape, device=dev, generator=g); v = torch.randn(P.v[:, t:t + 1].shape, device=dev, generator=g)
                g0 = torch.zeros_like(P.g[:, t:t + 1])
                for s_, tau in ladder:
                    o, _ = den.step((p, g0, v), torch.full((B, 1), M.shortcut_code(s_, e, k), device=dev), c["prev_action"][:, t:t + 1], mem, P.fix[:, t:t + 1], cache)
                    p = p + (o[0].float() - p) / (1 - tau) * d; v = v + (o[2].float() - v) / (1 - tau) * d
            cache = M.context_pass(den, tuple(x[:, t:t + 1].float() for x in P.parts()), c["prev_action"][:, t:t + 1], mem, P.fix[:, t:t + 1], cache, cfg, g)


def cost(model, cfg):
    d, prefix = cfg["data"], cfg["data"]["context_prefix"]
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], prefix, seed=CC.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    out = {}
    for Sx in (64, 256):
        times = []
        for rep in range(3):
            c, P, rin, bp = CC.inputs(model, cfg, streams.next(Sx), DEV)
            t0 = CC.sync(); eye_pass_sample(model, cfg, c, P, rin, bp, prefix, DEV); t1 = CC.sync()
            if rep:
                times.append(t1 - t0)
        out[f"sample_eye_{Sx}"] = {"sec_per_update": float(np.mean(times)), "fraction_of_update": float(np.mean(times)) / CC.BASE[Sx][0]}
        print(Sx, out[f"sample_eye_{Sx}"], flush=True)
    fresh = (out["sample_eye_64"]["sec_per_update"] * CC.BASE[64][1] + out["sample_eye_256"]["sec_per_update"] * CC.BASE[256][1]) / 3600
    out["projection_hours"] = {"fresh_sample_eye_extra": fresh}
    return out


def main(n=150):
    model, cfg = AI.load(CK)
    model.requires_grad_(False).eval()
    pol = dict(cfg["fixation"]["chooser"])
    R = S2.load_readers(model, cfg)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, light, lookup, loff, cover = C1.semantics(L)
    dl = S.episodes(L)
    eps = np.sort(np.random.default_rng(20261120).choice(len(L["ep_T"]), n, replace=False))
    rng = np.random.default_rng(7)                                          # C2's picks, exactly
    picks = {int(e): np.sort(rng.choice(np.arange(9, int(L["ep_T"][e]) + 1), min(20, int(L["ep_T"][e]) - 8), replace=False))
             for e in eps if L["ep_T"][e] >= 10}
    o = np.load(DL.EV.S2 / "oracle.npz"); U = DL.units(o, 4000)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    units = []
    for u in U:
        w, kind, i, j = u[:4]
        e, t0 = int(rows[w, 0]), int(rows[w, 3])
        if 9 <= t0 + 1 <= L["ep_T"][e]:
            r = DL.deleted(mg, mp, u)
            if r:
                units.append((w, e, t0 + 1, kind, i, j, np.mean([x[0] for x in r])))
    units = units if n == 150 else units[:60]
    S_by, tok, drawn_kind = {}, {}, {}
    for e in sorted(set(picks) | {x[1] for x in units}):
        st = C2.episode(model, cfg, dl, e)
        ts = sorted({int(t) for t in picks.get(e, [])} | {x[2] for x in units if x[1] == e})
        draws = [samples(model, cfg, st, e, ts, dr) for dr in range(DRAWS)]
        real = st[1].p[0, ts].float()
        for qi, t in enumerate(ts):
            S_by[(e, t)] = np.stack([((real[qi] - dr[0][qi]) ** 2).sum(-1).cpu().numpy() for dr in draws])      # [draws, 81]
        if any(x[1] == e for x in units):                                   # each draw's outcome through the reader
            for dr_i, (pd, vd) in enumerate(draws):
                pin = torch.stack([st[1].p[0, [t - 1 for t in ts]].float(), pd], 1)
                vin = torch.stack([st[1].v[0, [t - 1 for t in ts]].float(), vd], 1)
                kind = S2.read(R, pin, vin, np.zeros((len(ts), 2), int))[0][:, 1]
                for qi, t in enumerate(ts):
                    drawn_kind[(e, t, dr_i)] = kind[qi]
    keys = [(e, int(t)) for e in picks for t in picks[e]]
    rws = np.array([lookup[loff[e] + t] for e, t in keys])
    night = light[rws] < 0.5
    Sm = np.stack([S_by[q].mean(0) for q in keys])
    prob = C2.policy_s(Sm, pol)
    res = {"windows": len(keys), "draws": DRAWS,
           "draw_spread_over_mean": float(np.mean([np.abs(S_by[q][0] - S_by[q][1]).sum() / max(S_by[q].mean(0).sum(), 1e-9) for q in keys]))}
    cov = {}
    for name, mask in cover.items():
        mk = mask[rws]; present = mk.any(-1); hit = mk[:, WIN].any(-1)
        lw, uw = (prob * hit).sum(-1), hit.mean(-1)
        for tag, sel in (("all", present), ("day", present & ~night), ("night", present & night)):
            cov[f"{name}|{tag}"] = [float(lw[sel].mean()), float(uw[sel].mean()), int(sel.sum())]
    res["coverage_sample_eye"] = cov
    prev = json.loads((HERE / "c2_eye.json").read_text())["coverage"] if n == 150 else {}
    res["surprise_share"] = {g: float((Sm * groups[g][rws]).sum() / Sm.sum()) for g in C1.GROUPS}
    res["surprise_share_night"] = {g: float((Sm[night] * groups[g][rws][night]).sum() / Sm[night].sum()) for g in C1.GROUPS}
    # deletion link
    rowsU, covd = [], {"deleted": [], "kept": []}
    for w, e, t, kind, i, j, rate in units:
        dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
        r0, c0 = i - dpos[0], j - dpos[1]
        cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
        if not cells:
            continue
        for dr_i in range(DRAWS):
            gk = drawn_kind[(e, t, dr_i)]
            rr, cc_ = np.array(cells) // 9, np.array(cells) % 9
            kept = bool((gk[rr, cc_] == kind).any())
            p1 = C2.policy_s(S_by[(e, t)][dr_i][None], pol)[0]
            cv = float(sum(p1[wi] for wi in range(49) if set(WIN[wi]) & set(cells)))
            covd["kept" if kept else "deleted"].append(cv)
        rowsU.append((float(S_by[(e, t)].mean(0)[cells].max()), rate))
    a = np.array(rowsU); qs = np.quantile(a[:, 0], [0.25, 0.5, 0.75]); bins = np.digitize(a[:, 0], qs)
    res["deletion_link"] = {"units": len(a), "draw_outcomes": {k_: len(v) for k_, v in covd.items()},
                            "eye_coverage_of_mob_when_its_draw_deleted_it": float(np.mean(covd["deleted"])) if covd["deleted"] else None,
                            "eye_coverage_of_mob_when_its_draw_kept_it": float(np.mean(covd["kept"])) if covd["kept"] else None,
                            "s2_deletion_by_surprise_quartile": [float(a[bins == q, 1].mean()) for q in range(4)],
                            "corr_surprise_s2_deletion": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])}
    print(json.dumps({k_: v for k_, v in res.items()}, indent=1), flush=True)
    if prev:
        print("coverage, sample-eye | world-eye (C2) | chooser | uniform")
        for k_ in ("any_entity|all", "any_entity|night", "cows|all", "zombies|all", "changed_hud|all", "entering_map|all"):
            print(f"  {k_:18s} {cov[k_][0]:.3f} | {prev['world_eye_fresh'][k_][0]:.3f} | {prev['chooser'][k_][0]:.3f} | {cov[k_][1]:.3f}", flush=True)
    if n == 150:
        res["cost"] = cost(model, cfg)
    (HERE / ("c4_sample.json" if n == 150 else f"c4_sample_{n}.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
