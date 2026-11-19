"""C2's location part (PLAN.md): where a world-eye would look, how a stale cache changes that, and whether the world's
surprise marks the cows it deletes.
- **World-eye.** Per DEV target frame, P1.5's zero-signal estimate given the real past (the monitor's window: 8
  context frames at the context signal, the real record and belief, the monitor's noise seeding). Surprise per cell is
  the squared distance from checkpoint 10's tokens of the real frame. Windows go through the chooser's own policy with
  contingency off (`chooser.policy(0, S)`).
- **Same frames, current chooser.** P1.5's chooser on those frames (`chooser.inputs`, final head).
- **Staleness.** Estimates from checkpoints 9, 8 and 5 (1, 2 and 5 passes stale) against checkpoint 10's tokens, as a
  cache would hold them. Reading: total variation and argmax agreement with the fresh eye, and coverage.
- **Deletion link.** For S2's isolated start mobs: the world's zero-signal surprise around the mob in the recorded next
  frame, against P1.5's step-1 deletion rate in its S2 futures.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c2_eye [N_EPISODES]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c1")
DL = importlib.import_module("artifacts.experiments.20261120_switching_dissection.deletions")
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "20261116_p15/run"
CKPTS = (10, 9, 8, 5)
DEV = torch.device("cuda")
WIN = C.WIN.numpy()


@torch.no_grad()
def episode(model, cfg, dl, e):
    """Real frames, tokens, record (denoiser read, bf16) and predicted beliefs of one DEV episode (b2's real pass)."""
    m = cfg["p1"]["monitor"]
    T = int(dl[e].T)
    recs = [[(0, 1, t, *D.fixation_prior(np.random.default_rng([m["fixed_fixation_seed"], int(e), t]), ())) for t in range(T + 1)]]
    c = TR.tensors(D.gather([dl[e]], recs), DEV)
    with TR.autocast(DEV, cfg):
        Pe = M.Percept(*(torch.cat(z, 1) for z in zip(*[model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                                                        for a in range(0, T + 1, 512)])), c["fixation"])
        state, carry, rins, ys = model.record.zero(1, DEV), model.belief.initial(1, DEV), [], []
        for a in range(0, T + 1, 512):
            sl = slice(a, min(T + 1, a + 512))
            first = torch.zeros(1, sl.stop - a, dtype=torch.bool, device=DEV); first[:, 0] = a == 0
            ri, si = model.record.scan(Pe.p[:, sl], first, state); state = si[:, -1]
            y, carry = model.belief(model.belief.tokens(model.belief.percept_embed(M.Percept(Pe.p[:, sl], Pe.g[:, sl], Pe.v[:, sl], Pe.fix[:, sl]), si),
                                                        c["reward"][:, sl], c["terminated"][:, sl], c["action"][:, sl]), carry)
            rins.append(ri.to(torch.bfloat16)); ys.append(y[:, 1::2])
        y = torch.cat(ys, 1)
    return c, Pe, torch.cat(rins, 1), torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], 1)


@torch.no_grad()
def estimates(model, cfg, st, e, ts):
    """Zero-signal periphery estimates of target frames ts (all >= the prefix) -> [len(ts), 81, 16] float."""
    c, Pe, rin, bp = st
    m, sc = cfg["p1"]["monitor"], cfg["edm"]["shortcut"]
    P, k = cfg["data"]["context_prefix"], sc["k_max"]
    sl = lambda a, t: a[0, t - P:t + 1]
    stack = lambda a: torch.stack([sl(a, t) for t in ts])
    x0 = tuple(stack(a) for a in Pe.parts())
    noise = []
    for t in ts:
        g = torch.Generator(device=DEV).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(e) * 4_099 + int(t))
        noise.append(tuple(torch.randn(a.shape[1:], device=DEV, generator=g) for a in x0))
    noise = tuple(torch.stack(z) for z in zip(*noise))
    sg = torch.full((len(ts), P + 1), sc["context_signal"], device=DEV, dtype=torch.long); sg[:, -1] = 0
    em = torch.full_like(sg, k.bit_length() - 1)
    tau = [s.float() / k for s in M._parts((sg, sg))]
    zt = tuple((1 - M._bcast(tt, x)) * nn + M._bcast(tt, x) * x for x, nn, tt in zip(x0, noise, tau))
    with TR.autocast(DEV, cfg):
        o = model.denoiser(zt, (M.shortcut_code(sg, em, k), M.shortcut_code(sg, em, k)), stack(c["prev_action"]),
                           (model.belief.memory(stack(bp)), stack(rin)), stack(Pe.fix), stack(c["episode"]), stack(c["valid"]))
    return o[0][:, -1].float()


def policy_s(S_cell, pol):
    s = torch.as_tensor(S_cell[:, WIN].sum(-1))
    return C.policy(torch.zeros_like(s), s, pol).numpy()


def main(n=150):
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, light, lookup, loff, cover = C1.semantics(L)
    dl = S.episodes(L)
    eps = np.sort(np.random.default_rng(20261120).choice(len(L["ep_T"]), n, replace=False))
    rng = np.random.default_rng(7)
    picks = {int(e): np.sort(rng.choice(np.arange(9, int(L["ep_T"][e]) + 1), min(20, int(L["ep_T"][e]) - 8), replace=False))
             for e in eps if L["ep_T"][e] >= 10}
    # S2 units with a recorded next frame (deletion link)
    o = np.load(DL.EV.S2 / "oracle.npz")
    U = DL.units(o, 4000)
    b = np.load(DL.EV.BANK); rows = b["rows"][b["sample"]]
    z = np.load(DL.EV.S2 / "model.npz"); mg, mp = z["m_grid"], z["m_pos"].astype(int)
    units = []
    for u in U:
        w, kind, i, j = u[:4]
        e, t0 = int(rows[w, 0]), int(rows[w, 3])
        if t0 + 1 <= L["ep_T"][e] and t0 + 1 >= 9:
            r = DL.deleted(mg, mp, u)
            if r:
                units.append((w, e, t0 + 1, kind, i, j, np.mean([x[0] for x in r]), u[7]))
    units = units if n == 150 else units[:60]                          # smoke: a few units
    unit_eps = sorted({x[1] for x in units})
    tokens, est = {}, {k: {} for k in CKPTS}
    for k in CKPTS:
        model, cfg = AI.load(RUN / f"ckpt_{k:02d}.pt")
        model.requires_grad_(False).eval()
        todo = sorted(set(picks) | (set(unit_eps) if k == 10 else set()))
        for e in todo:
            st = episode(model, cfg, dl, e)
            ts = sorted(set(picks.get(e, []).tolist()) | ({x[2] for x in units if x[1] == e} if k == 10 else set()))
            if not ts:
                continue
            x = estimates(model, cfg, st, e, ts).cpu().numpy()
            for t, xx in zip(ts, x):
                est[k][(e, t)] = xx.astype(np.float16)
                if k == 10:
                    tokens[(e, t)] = st[1].p[0, t].float().cpu().numpy().astype(np.float16)
        if k == 10:
            pol = dict(cfg["fixation"]["chooser"])
            head = C.Head(pol["history"]).to(DEV)
            head.load_state_dict(torch.load(RUN / "ckpt_10.pt", weights_only=False, map_location="cpu")["fixation_controller"]["head"]); head.eval()
            chooser = {}
            for e in picks:
                T = int(L["ep_T"][e])
                chunk = D.gather([dl[e]], [[(0, 1, t, 4, 4) for t in range(T + 1)]])
                with torch.no_grad(), TR.autocast(DEV, cfg):
                    p, v = C.coarse(model.perception, torch.as_tensor(chunk["frames"], device=DEV).flatten(0, 1))
                xx, y = C.inputs(p.reshape(1, T + 1, 81, 16).float(), v.reshape(1, T + 1, 8).float(), chunk, 8, 8)
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    taken = head(xx); noop = head((xx[0], xx[1], torch.zeros_like(xx[2])))
                    prob = C.policy((taken - noop).square().sum(-1)[:, C.WIN.to(DEV)].sum(-1), (y - taken).square().sum(-1)[:, C.WIN.to(DEV)].sum(-1), pol)
                for t in picks[e]:
                    chooser[(e, int(t))] = prob[int(t) - 8].cpu().numpy()
        del model; torch.cuda.empty_cache()
        print("checkpoint", k, len(est[k]), "estimates", flush=True)
    keys = [(e, int(t)) for e in picks for t in picks[e]]
    tok = np.stack([tokens[q] for q in keys]).astype(np.float32)
    rws = np.array([lookup[loff[e] + t] for e, t in keys])
    night = light[rws] < 0.5
    res = {"windows": len(keys)}
    S10 = ((tok - np.stack([est[10][q] for q in keys]).astype(np.float32)) ** 2).sum(-1)
    p10 = policy_s(S10, pol)
    pch = np.stack([chooser[q] for q in keys])

    def coverage(prob):
        out = {}
        for name, mask in cover.items():
            mk = mask[rws]; present = mk.any(-1); hit = mk[:, WIN].any(-1)
            lw = (prob * hit).sum(-1); uw = hit.mean(-1)
            for tag, sel in (("all", present), ("day", present & ~night), ("night", present & night)):
                out[f"{name}|{tag}"] = [float(lw[sel].mean()), float(uw[sel].mean()), int(sel.sum())]
        return out
    res["coverage"] = {"world_eye_fresh": coverage(p10), "chooser": coverage(pch)}
    res["surprise_share"] = {g: float((S10 * groups[g][rws]).sum() / S10.sum()) for g in C1.GROUPS}
    res["surprise_share_night"] = {g: float((S10[night] * groups[g][rws][night]).sum() / S10[night].sum()) for g in C1.GROUPS}
    res["stale"] = {}
    for k in CKPTS[1:]:
        Sk = ((tok - np.stack([est[k][q] for q in keys]).astype(np.float32)) ** 2).sum(-1)
        pk = policy_s(Sk, pol)
        res["stale"][f"ckpt_{k:02d}"] = {"passes_stale": 10 - k, "tv_to_fresh": float(0.5 * np.abs(pk - p10).sum(-1).mean()),
                                        "argmax_agreement": float((pk.argmax(-1) == p10.argmax(-1)).mean()),
                                        "tv_fresh_vs_chooser": float(0.5 * np.abs(pch - p10).sum(-1).mean()), "coverage": coverage(pk)}
    # deletion link (checkpoint 10)
    dlk = []
    for w, e, t, kind, i, j, rate, nt in units:
        x = ((tokens[(e, t)].astype(np.float32) - est[10][(e, t)].astype(np.float32)) ** 2).sum(-1)       # [81]
        d = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
        r0, c0 = i - d[0], j - d[1]
        cells = [r * 9 + cc for r in range(r0 - 1, r0 + 2) for cc in range(c0 - 1, c0 + 2) if 0 <= r < 7 and 0 <= cc < 9]
        if not cells:
            continue
        pr = policy_s(x[None], pol)[0]
        cov_ = float(sum(pr[wi] for wi in range(49) if set(WIN[wi]) & set(cells)))
        dlk.append((float(x[cells].max()), float(x.mean()), cov_, rate, bool(nt), int(kind)))
    a = np.array(dlk)
    qs = np.quantile(a[:, 0], [0.25, 0.5, 0.75])
    bins = np.digitize(a[:, 0], qs)
    res["deletion_link"] = {"units": len(a), "surprise_quartile_bounds": qs.tolist(),
                            "deletion_by_surprise_quartile": [float(a[bins == q, 3].mean()) for q in range(4)],
                            "eye_coverage_by_surprise_quartile": [float(a[bins == q, 2].mean()) for q in range(4)],
                            "corr_surprise_deletion": float(np.corrcoef(a[:, 0], a[:, 3])[0, 1]),
                            "deleted_vs_kept": {"surprise_if_deleted_any": float(a[a[:, 3] > 0, 0].mean()), "surprise_if_never": float(a[a[:, 3] == 0, 0].mean()),
                                                "neighbourhood_over_frame_mean_if_deleted": float((a[a[:, 3] > 0, 0] / a[a[:, 3] > 0, 1]).mean()),
                                                "neighbourhood_over_frame_mean_if_never": float((a[a[:, 3] == 0, 0] / a[a[:, 3] == 0, 1]).mean())}}
    print(json.dumps({k: v for k, v in res.items() if k != "stale"}, indent=1), flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "coverage"} for k, v in res["stale"].items()}, indent=1), flush=True)
    (HERE / ("c2_eye.json" if n == 150 else f"c2_eye_{n}.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
