"""Stage A5 of the switching dissection (PLAN.md): a paired fixation intervention on P1.5's step-1 deletion. S2's world
loop (`s2.model_futures`) is copied with H = 1 and the same batches and seeds. Each start's last real frame (state 0)
is re-encoded at a chosen fixation, in that start's own copy of its history: glimpse tokens and fixation replaced;
periphery and lighting do not depend on the fixation. Three arms per start:
- orig: the seeded draw; must reproduce S2's saved step-1 frames.
- on: a window centred on the start's target mob, one isolated inner start mob drawn per start.
- away: the map window farthest from it that excludes it.

Every future keeps its noise across arms, so the comparison is paired.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.a5 [N_STARTS]
"""
import dataclasses, importlib, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
DL = importlib.import_module("artifacts.experiments.20261120_switching_dissection.deletions")
EV = DL.EV
from saccade import data as D

M, S, C, TR, G, DEV, MF, NB = S2.M, S2.S, S2.C, S2.TR, S2.G, S2.DEV, S2.MF, S2.NB
CENTRES = [(r, c) for r in range(1, 8) for c in range(1, 8)]


def targets(n_starts):
    """One isolated inner start mob per start (deletions.units), drawn with a fixed seed: start -> (kind, i, j), and
    each arm's fixation for state 0."""
    o = np.load(EV.S2 / "oracle.npz")
    by = {}
    for u in DL.units(o, n_starts):
        by.setdefault(u[0], []).append(u)
    rng = np.random.default_rng(20261120)
    out = {}
    for w, us in by.items():
        n, k, i, j = us[rng.integers(len(us))][:4]
        far = [c for c in CENTRES if c[0] <= 5 and (abs(c[0] - i) > 1 or abs(c[1] - j) > 1)]
        away = max(far, key=lambda c: (max(abs(c[0] - i), abs(c[1] - j)), -c[0], -c[1]))
        out[int(w)] = {"kind": int(k), "cell": (int(i), int(j)), "on": (int(i), int(j)), "away": away}
    return out


@torch.no_grad()
def step1(model, cfg, R, enc, rows, plans, eps, sub, L, grids, override):
    """s2.model_futures with H = 1; override: start -> fixation (row, col) of its state-0 frame (None = seeded)."""
    state0 = torch.load(S2.CK, weights_only=False, map_location="cpu")["fixation_controller"]
    pol = dict(cfg["fixation"]["chooser"])
    head = C.Head(pol["history"]).to(DEV); head.load_state_dict(state0["head"]); head.eval().requires_grad_(False)
    win, centres, Hh = C.WIN.to(DEV), C.CENTRES.to(DEV), pol["history"]
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    P = cfg["data"]["context_prefix"]
    out, kc = {}, np.minimum(rows[:, 3], P - 1) + 1
    batches = [b for k in np.unique(kc) for b in np.array_split(np.flatnonzero(kc == k), max(1, -(-int((kc == k).sum()) // NB)))]
    dl_eps = S.episodes(L)
    for bi, ws in enumerate(batches):
        rr = [(sub[int(rows[w, 0])], int(rows[w, 3])) for w in ws]; k_ctx = int(kc[ws[0]])
        Bn = len(ws) * MF
        rep = lambda x: x.repeat_interleave(MF, 0)
        own = []                                                         # each start's own (g, fix) at state 0
        for (i, t), w in zip(rr, ws):
            en = enc[i]
            fx = override.get(int(w))
            if fx is None:
                own.append((en["g"][t], en["fix"][t]))
            else:
                c = TR.tensors(D.gather([dl_eps[int(rows[w, 0])]], [[(0, 1, t, *fx)]]), DEV)
                with TR.autocast(DEV, cfg):
                    pc = model.perception(c["frames"], c["fixation"])
                assert torch.allclose(pc.p[0, 0].float().cpu(), en["p"][t].float(), atol=2e-2), "periphery depends on fixation?"
                own.append((pc.g[0, 0].to(en["g"].dtype).cpu(), torch.as_tensor(fx, dtype=en["fix"].dtype)))
        at = lambda key, dt: torch.stack([own[a][0] if key == "g" and dt == 0 else own[a][1] if key == "fix" and dt == 0
                                          else enc[i][key][max(t + dt, 0)] for a, (i, t) in enumerate(rr)]).to(DEV)
        cs, bps, rl = [], [], []
        for a, ((i, t), w) in enumerate(zip(rr, ws)):
            en = enc[i]
            gq, fq = en["g"][:t + 1].clone(), en["fix"][:t + 1].clone()
            gq[t], fq[t] = own[a]
            with TR.autocast(DEV, cfg):
                pp = M.Percept(en["p"][:t + 1].to(DEV).float()[None], gq.to(DEV).float()[None], en["v"][:t + 1].to(DEV).float()[None], fq.to(DEV)[None])
                reads, _, last = S.history(model, pp.p)
                y, c = model.belief(model.belief.tokens(S.embed(model, pp, reads), en["reward"][:t + 1].to(DEV)[None],
                                    en["terminated"][:t + 1].to(DEV)[None], en["action"][:t + 1].to(DEV)[None]), model.belief.initial(1, DEV))
            cs.append(c); bps.append(y[:, -1].float()); rl.append(last)
        carry = tuple(type(cs[0][l])(*(rep(torch.cat([getattr(c[l], f.name) for c in cs])) for f in dataclasses.fields(cs[0][l])))
                      for l in range(len(cs[0])))
        bp, rs = rep(torch.cat(bps)), rep(torch.cat(rl))
        rin = S.window(model, enc, rr, P - 1)
        g = torch.Generator(device=DEV).manual_seed(20261119 + bi); gg = torch.Generator(device=DEV).manual_seed(30261119 + bi)
        cache = model.denoiser.new_cache()
        with TR.autocast(DEV, cfg):
            for dt in range(1 - k_ctx, 1):
                cache = M.context_pass(model.denoiser, tuple(rep(at(k, dt)).float()[:, None] for k in ("p", "g", "v")),
                                       rep(at("prev_action", dt))[:, None],
                                       S.cond(model.belief.memory(rep(at("bp", dt)).float())[:, None], rep(rin[:, P - 1 + dt])[:, None]),
                                       rep(at("fix", dt))[:, None], cache, cfg, g)
        ok = rep(torch.tensor([[t + dt >= 0 for dt in range(1 - Hh, 1)] for _, t in rr], device=DEV))
        hist = {k: rep(torch.stack([torch.stack([enc[i][k][max(t + dt, 0)] for dt in range(1 - Hh, 1)]) for i, t in rr]).to(DEV))
                for k in ("p", "v", "prev_action", "reward", "terminated")}
        hist = {k: x.float() if k != "prev_action" else x for k, x in hist.items()}
        plan = rep(torch.as_tensor(plans[ws], device=DEV))
        fix = rep(at("fix", 0))
        pa = plan[:, 0]
        okf = ok.float()
        ctx = torch.cat([hist["v"] * okf[..., None], F.one_hot((hist["prev_action"] + 1).clamp(0, 17), 18).float() * okf[..., None],
                         (M.SYMLOG(hist["reward"]) * okf)[..., None], (hist["terminated"] * okf)[..., None], okf[..., None]], -1).flatten(1)
        ph = hist["p"] * okf[..., None, None]

        def choose(p_, v_):
            with torch.autocast("cuda", enabled=False):
                taken = head((ph, ctx, pa)); noop = head((ph, ctx, torch.zeros_like(pa)))
                c_ = (taken - noop).square().sum(-1)[:, win].sum(-1)
                s_ = (p_[:, 0].float() - taken).square().sum(-1)[:, win].sum(-1)
                prob = C.policy(c_, s_, pol)
            u = torch.rand(Bn, 1, device=DEV, generator=gg, dtype=torch.float64)
            return centres[torch.searchsorted(prob.double().cumsum(-1), u).squeeze(-1).clamp(max=48)][:, None]
        with TR.autocast(DEV, cfg):
            mem = S.cond(model.belief.memory(bp)[:, None], rs[:, None])
            parts, fx_ = M.sample(model.denoiser, dims, pa[:, None], mem, fix[:, None], cache, cfg, g, choose=choose)
            parts = tuple(x[:, 0].float() for x in parts)
        pos0 = np.repeat(np.stack([grids[int(rows[w, 0])]["pos"][int(rows[w, 3])] for w in ws]), MF, 0)
        gr, hu, sl, po = S2.read(R, torch.stack([rep(at("p", 0)).float(), parts[0]], 1), torch.stack([rep(at("v", 0)).float(), parts[2]], 1), pos0)
        for a, w in enumerate(ws):
            s = slice(MF * a, MF * (a + 1))
            out.setdefault("grid", {})[int(w)] = gr[s]; out.setdefault("pos", {})[int(w)] = po[s]
            out.setdefault("ptok", {})[int(w)] = parts[0][s].half().cpu().numpy()            # the drawn periphery
            out.setdefault("g0", {})[int(w)] = own[a][0].float().cpu().numpy()            # state 0's glimpse
    return {k: np.stack([v[w] for w in range(len(rows))]) for k, v in out.items()}


def main(n=None):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = S2.load(); EV  # noqa
    S2.EV.guard()
    R = S2.load_readers(model, cfg)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, plans = S2.bank(n)
    eps = np.unique(rows[:, 0]); sub = {e: i for i, e in enumerate(eps)}
    enc = S.encode(model, S2.sublabels(L, eps), cfg)
    grids = G.episode_grids(L)
    T = targets(len(rows))
    res, t0 = {}, time.time()
    for arm in ("orig", "on", "away"):
        override = {} if arm == "orig" else {w: v[arm] for w, v in T.items()}
        res[arm] = step1(model, cfg, R, enc, rows, plans, eps, sub, L, grids, override)
        print(arm, f"{time.time() - t0:.0f}s", flush=True)
    saved = np.load(S2.HERE / "model.npz")
    same = (res["orig"]["grid"][:, :, 1] == saved["m_grid"][:len(rows), :, 1]).all((-1, -2))
    print(f"orig vs S2's saved step-1 frames: {same.mean():.4f} of futures identical", flush=True)
    sel = np.array(sorted(T))                                            # the override reached the model?
    for arm in ("on", "away"):
        dg = np.abs(res[arm]["g0"][sel] - res["orig"]["g0"][sel]).mean()
        dp = np.abs(res[arm]["ptok"][sel].astype(np.float32) - res["orig"]["ptok"][sel].astype(np.float32))
        cell = np.array([T[w]["cell"][0] * 9 + T[w]["cell"][1] for w in sel])
        near = np.stack([dp[q, :, [c + d for d in (-10, -9, -8, -1, 0, 1, 8, 9, 10) if 0 <= c + d < 63]].mean() for q, c in enumerate(cell)])
        print(f"{arm} vs orig: state-0 glimpse |diff| {dg:.4f}; drawn step-1 periphery |diff| all cells {dp.mean():.5f},"
              f" around the target {near.mean():.5f}", flush=True)
    for r in res.values():
        r.pop("ptok"); r.pop("g0")
    tag = "" if n is None else "_smoke"
    np.savez_compressed(DL.EV.HERE / f"a5{tag}.npz", **{f"{a}_{k}": v for a, r in res.items() for k, v in r.items()},
                        targets=np.array([[w, v["kind"], *v["cell"], *v["on"], *v["away"]] for w, v in sorted(T.items())]))
    summarise(tag)


def summarise(tag=""):
    z = np.load(DL.EV.HERE / f"a5{tag}.npz")
    o = np.load(EV.S2 / "oracle.npz")
    night = o["lab_light"][:, 0, 0] < 0.5
    b = np.load(EV.BANK); ep = b["rows"][b["sample"]][:, 0]
    rng = np.random.default_rng(20261120)
    rows = {}
    for w, k, i, j, *_ in z["targets"]:
        d = {}
        for arm in ("orig", "on", "away"):
            r = DL.deleted(z[f"{arm}_grid"], z[f"{arm}_pos"].astype(int), (w, k, i, j))
            d[arm] = (sum(x[0] for x in r), len(r))
        key = (EV.KINDS[k - 1], "night" if night[w] else "day", "ring" if i in (1, 5) or j in (1, 7) else "inner")
        rows.setdefault(key, []).append((w, d))
    out = {}
    print("step-1 deletion of the target mob: on / away / orig (paired on - away, episode-bootstrap SE), units")
    for key in sorted(rows):
        if key[0] == "arrows":
            continue
        us = rows[key]
        num = {a: np.array([u[1][a][0] for u in us], float) for a in ("orig", "on", "away")}
        den = {a: np.array([u[1][a][1] for u in us], float) for a in ("orig", "on", "away")}
        e = ep[[u[0] for u in us]]
        rate = {a: num[a].sum() / max(den[a].sum(), 1) for a in num}
        uniq, inv = np.unique(e, return_inverse=True)
        agg = lambda x: np.bincount(inv, weights=x, minlength=len(uniq))
        A, B, Da, Db = agg(num["on"]), agg(num["away"]), agg(den["on"]), agg(den["away"])
        boots = [A[q].sum() / max(Da[q].sum(), 1) - B[q].sum() / max(Db[q].sum(), 1) for q in (rng.integers(0, len(uniq), len(uniq)) for _ in range(1000))]
        out["|".join(key)] = {**{a: float(v) for a, v in rate.items()}, "on_minus_away": float(rate["on"] - rate["away"]), "se": float(np.std(boots)), "units": len(us)}
        print(f"  {key[0]:9s} {key[1]:5s} {key[2]:5s} on {rate['on']:.3f} away {rate['away']:.3f} orig {rate['orig']:.3f}"
              f" | on-away {rate['on'] - rate['away']:+.3f} ({np.std(boots):.3f}) units {len(us)}", flush=True)
    (DL.EV.HERE / f"a5{tag}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
