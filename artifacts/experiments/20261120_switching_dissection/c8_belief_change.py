"""C8 (PLAN.md): an eye scored by the world's belief change. For each of C2's DEV picks t (t + 1 <= T), P1.5's
zero-signal prediction of frame t + 1 under NOOP, given the real frame t, against the same prediction given the
world's own estimate of frame t (substituted in the denoiser's context, the record's update and Mamba's percept).
Beside it on the same frames: C2's world-eye surprise and the chooser.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c8_belief_change [N_EPISODES]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import s1 as S

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C1 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c1")
C2 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_eye")
HERE = Path(__file__).resolve().parent
CK = HERE.parent / "20261116_p15/run/ckpt_10.pt"
DEV = torch.device("cuda")
WIN = C.WIN.numpy()
NOOP = 0


@torch.no_grad()
def scores(model, cfg, dl, e, ts):
    """-> S [W, 81] (the world-eye's surprise at t), B [W, 81] (belief change), max |bp re-run - C2's bp| (check)."""
    st = C2.episode(model, cfg, dl, e)
    c, Pe, rin, bp = st
    T = Pe.p.shape[1] - 1
    xh = C2.estimates(model, cfg, st, e, ts)                               # the world's estimate of frame t
    S_ = ((Pe.p[0, ts].float() - xh) ** 2).sum(-1)
    with TR.autocast(DEV, cfg):                                             # real percept embeddings; record after t - 1
        state, embs, before = model.record.zero(1, DEV), [], {}
        for a in range(0, T + 1, 512):
            sl = slice(a, min(T + 1, a + 512))
            first = torch.zeros(1, sl.stop - a, dtype=torch.bool, device=DEV); first[:, 0] = a == 0
            _, si = model.record.scan(Pe.p[:, sl], first, state); state = si[:, -1]
            for t in ts:
                if a <= t - 1 < sl.stop:
                    before[t] = si[:, t - 1 - a].clone()
            embs.append(model.belief.percept_embed(M.Percept(Pe.p[:, sl], Pe.g[:, sl], Pe.v[:, sl], Pe.fix[:, sl]), si)); del si
        emb = torch.cat(embs, 1)
        nf = torch.zeros(1, 1, dtype=torch.bool, device=DEV)
        after_cf = {t: model.record.scan(xh[i][None, None], nf, before[t])[1][:, 0] for i, t in enumerate(ts)}   # [1, 81, dim]
        emb_cf = {t: model.belief.percept_embed(M.Percept(xh[i][None, None].to(Pe.p.dtype), Pe.g[:, t:t + 1], Pe.v[:, t:t + 1], Pe.fix[:, t:t + 1]),
                                                after_cf[t][:, None]) for i, t in enumerate(ts)}
        Lm = max(ts) + 1                                                    # Mamba on each prefix 0..t (padded at the end)
        arms = [("real_noop", False, True), ("cf_noop", True, True), ("real_action", False, False)]
        seq_e, seq_a = [], []
        for _, cf, noop in arms:
            for i, t in enumerate(ts):
                x = emb[0, :Lm].clone()
                if cf:
                    x[t] = emb_cf[t][0, 0]
                a_ = c["action"][0, :Lm].clone()
                if noop:
                    a_[t] = NOOP
                seq_e.append(x); seq_a.append(a_)
        X = torch.stack(seq_e); A_ = torch.stack(seq_a); n = len(seq_e)
        y, _ = model.belief(model.belief.tokens(X, c["reward"][:, :Lm].expand(n, -1), c["terminated"][:, :Lm].expand(n, -1), A_), model.belief.initial(n, DEV))
        pred = torch.stack([y[k, 2 * t + 1] for k, (_, t) in enumerate([(a, t) for a in arms for t in ts])])
    W = len(ts)
    bp_real, bp_cf, bp_chk = pred[:W], pred[W:2 * W], pred[2 * W:]
    check = float((bp_chk.float() - bp[0, [t + 1 for t in ts]].float()).abs().max())
    out = {}
    for name, cf, bpx in (("real", False, bp_real), ("cf", True, bp_cf)):
        est = []
        for i, t in enumerate(ts):                                          # one pick at a time: windows overlap
            keep = (Pe.p[0, t].clone(), rin[0, t + 1].clone(), bp[0, t + 1].clone(), c["prev_action"][0, t + 1].clone())
            if cf:
                Pe.p[0, t] = xh[i].to(Pe.p.dtype); rin[0, t + 1] = after_cf[t][0].to(rin.dtype)
            bp[0, t + 1] = bpx[i].to(bp.dtype); c["prev_action"][0, t + 1] = NOOP
            est.append(C2.estimates(model, cfg, st, e, [t + 1])[0])
            Pe.p[0, t], rin[0, t + 1], bp[0, t + 1], c["prev_action"][0, t + 1] = keep
        out[name] = torch.stack(est)
    B_ = ((out["real"] - out["cf"]) ** 2).sum(-1)
    return S_.cpu().numpy(), B_.cpu().numpy(), check


def main(n=150):
    model, cfg = AI.load(CK)
    model.requires_grad_(False).eval()
    pol = dict(cfg["fixation"]["chooser"])
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, light, lookup, loff, cover = C1.semantics(L)
    dl = S.episodes(L)
    eps = np.sort(np.random.default_rng(20261120).choice(len(L["ep_T"]), 150, replace=False))   # C2's picks, exactly
    rng = np.random.default_rng(7)
    picks = {int(e): np.sort(rng.choice(np.arange(9, int(L["ep_T"][e]) + 1), min(20, int(L["ep_T"][e]) - 8), replace=False))
             for e in eps if L["ep_T"][e] >= 10}
    picks = {e: [int(t) for t in v if t + 1 <= L["ep_T"][e]] for e, v in list(picks.items())[:n]}
    Sd, Bd, checks, chooser = {}, {}, [], {}
    head = C.Head(pol["history"]).to(DEV)
    head.load_state_dict(torch.load(CK, weights_only=False, map_location="cpu")["fixation_controller"]["head"]); head.eval()
    for e, ts in picks.items():
        if not ts:
            continue
        S_, B_, chk = scores(model, cfg, dl, e, ts)
        checks.append(chk)
        for i, t in enumerate(ts):
            Sd[(e, t)], Bd[(e, t)] = S_[i], B_[i]
        T = int(L["ep_T"][e])                                               # the chooser on the same frames (C2's code)
        chunk = D.gather([dl[e]], [[(0, 1, t, 4, 4) for t in range(T + 1)]])
        with torch.no_grad(), TR.autocast(DEV, cfg):
            p, v = C.coarse(model.perception, torch.as_tensor(chunk["frames"], device=DEV).flatten(0, 1))
        xx, y = C.inputs(p.reshape(1, T + 1, 81, 16).float(), v.reshape(1, T + 1, 8).float(), chunk, 8, 8)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            taken = head(xx); noop = head((xx[0], xx[1], torch.zeros_like(xx[2])))
            prob = C.policy((taken - noop).square().sum(-1)[:, C.WIN.to(DEV)].sum(-1), (y - taken).square().sum(-1)[:, C.WIN.to(DEV)].sum(-1), pol)
        for t in ts:
            chooser[(e, t)] = prob[t - 8].cpu().numpy()
        print(e, len(Sd), f"check {chk:.2e}", flush=True)
    keys = list(Sd)
    rws = np.array([lookup[loff[e] + t] for e, t in keys]); night = light[rws] < 0.5
    Sm, Bm = np.stack([Sd[q] for q in keys]), np.stack([Bd[q] for q in keys])

    def coverage(prob):
        out = {}
        for name, mask in cover.items():
            mk = mask[rws]; present = mk.any(-1); hit = mk[:, WIN].any(-1)
            lw, uw = (prob * hit).sum(-1), hit.mean(-1)
            for tag, sel in (("all", present), ("day", present & ~night), ("night", present & night)):
                out[f"{name}|{tag}"] = [float(lw[sel].mean()), float(uw[sel].mean()), int(sel.sum())]
        return out
    share = lambda X, sel: {g: float((X[sel] * groups[g][rws][sel]).sum() / X[sel].sum()) for g in C1.GROUPS}
    res = {"windows": len(keys), "mamba_rerun_max_abs_diff": float(max(checks)),
           "coverage": {"belief_change": coverage(C2.policy_s(Bm, pol)), "world_eye_surprise": coverage(C2.policy_s(Sm, pol)),
                        "chooser": coverage(np.stack([chooser[q] for q in keys]))},
           "share": {"belief_change": {"all": share(Bm, np.ones_like(night)), "day": share(Bm, ~night), "night": share(Bm, night)},
                     "world_eye_surprise": {"all": share(Sm, np.ones_like(night)), "day": share(Sm, ~night), "night": share(Sm, night)}}}
    print(json.dumps(res["share"], indent=1))
    print("coverage: belief change | world-eye surprise | chooser | uniform")
    for k in ("any_entity|all", "any_entity|day", "any_entity|night", "cows|all", "zombies|all", "zombies|night", "changed_hud|all", "entering_map|all"):
        r = res["coverage"]
        print(f"  {k:18s} {r['belief_change'][k][0]:.3f} | {r['world_eye_surprise'][k][0]:.3f} | {r['chooser'][k][0]:.3f} | {r['chooser'][k][1]:.3f}", flush=True)
    (HERE / ("c8_belief_change.json" if n == 150 else f"c8_belief_change_{n}.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
