"""Why recognition rarely proposes the true place (stage A, 11-24; diagnosis, no change to the design).

On panel episodes 0-22 with stage A's evaluation outages (gates' events), at every proposal while the agent is drifted
(true error > the read's width), with simulator positions scoring only:
- the support the winning place gets, the true place's support (cells whose kept candidates imply the true agent
  place), and the drifted place's (implying x_pre, no correction);
- where each support comes from: entries written before the outage began against entries written during or after it
  (which share its drift);
- the key's reach: per world cell, whether any of its top 32 entries by key (before deduplication) implies the true
  place, and whether one of its kept 8 does;
- the winner's entries: their age and whether they were written at the query's own screen cell;
- the true entry's rank: per world cell with any entry at the true place, the best such entry's rank among all
  entries by key and by raw token (L1), and its lighting gap to the query.

    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261124_learned_location.diagnosis_recognition [n_eps]
    LOC_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261124_learned_location.diagnosis_recognition probe
"""
import importlib, json, sys

import numpy as np
import torch

LOC = importlib.import_module("artifacts.experiments.20261124_learned_location.location")
LM, DEV, WIDTH, EXCL, TOPR = LOC.LM, LOC.DEV, LOC.WIDTH, LOC.EXCL, LOC.TOPR


def stats(est, t, p, v, x_true, outage_start, toks, light):
    """propose()'s candidates at frame t, scored against the true place."""
    Fn = t - EXCL
    cells = torch.nonzero(est.m["calib"].world.to(DEV))[:, 0]
    q = est.m["key"](p[cells].to(DEV), v[None].to(DEV).expand(len(cells), 8), cells)
    fi, ci = torch.nonzero(est.wm[:Fn], as_tuple=True)
    sims = q @ est.keys[fi, ci].T
    k = min(32, sims.shape[1])
    top_s, top_i = sims.topk(k, dim=1)
    P_e = est.X[fi[top_i]] + est.off[fi[top_i], ci[top_i]]
    xh = (P_e - est.m["calib"].offset.to(DEV)[cells][:, None]).cpu().numpy()
    frame = fi[top_i].cpu().numpy(); ws = top_s.clamp(min=0).cpu().numpy()
    xt = x_true.cpu().numpy(); xp = est.X[t].cpu().numpy()
    at_true32 = (np.abs(xh - xt).max(-1) < WIDTH).any(1)
    kept = []
    for a in range(len(cells)):
        seen_f, places = set(), []
        for j in range(k):
            if frame[a, j] in seen_f or any(np.abs(xh[a, j] - y).max() < WIDTH for y in places):
                continue
            seen_f.add(frame[a, j]); places.append(xh[a, j]); kept.append((a, j))
            if len(places) == TOPR:
                break
    def support(target, before=None):
        best = {}
        for a, j in kept:
            if np.abs(xh[a, j] - target).max() < WIDTH and (before is None or (frame[a, j] < outage_start) == before):
                best[a] = max(best.get(a, 0), ws[a, j])
        return sum(best.values()), len(best)
    pts = np.array([xh[a, j] for a, j in kept]); w_ = np.array([ws[a, j] for a, j in kept]); cid = np.array([a for a, _ in kept])
    near = np.abs(pts[:, None] - pts[None]).max(-1) < WIDTH
    Mx = np.zeros((len(pts), len(cells)))
    for c in np.unique(cid):
        m = cid == c; Mx[:, c] = (near[:, m] * w_[m][None]).max(1)
    sup = Mx.sum(1); b = int(sup.argmax())
    kept_true = np.zeros(len(cells), bool)
    for a, j in kept:
        if np.abs(xh[a, j] - xt).max() < WIDTH:
            kept_true[a] = True
    win = [(a, j) for a, j in kept if np.abs(xh[a, j] - pts[b]).max() < WIDTH]
    ages = [t - int(frame[a, j]) for a, j in win]
    ce_all = ci[top_i].cpu().numpy()
    same_cell = [int(ce_all[a, j]) == int(cells[a]) for a, j in win]
    P_all = est.X[fi] + est.off[fi, ci]                                   # every entry's world place
    is_true = ((P_all[None] - est.m["calib"].offset.to(DEV)[cells][:, None] - x_true.to(DEV)).abs().amax(-1) < WIDTH)
    has = is_true.any(1)
    best_true = torch.where(is_true, sims, torch.full_like(sims, -9)).amax(1)
    q_tok = toks[t][cells.cpu()].to(DEV); e_tok = toks[fi.cpu(), ci.cpu()].to(DEV)
    dist = torch.zeros_like(sims)
    for k_ in range(16):
        dist += (e_tok[None, :, k_] - q_tok[:, None, k_]).abs()
    best_true_d = torch.where(is_true, dist, torch.full_like(dist, 1e9)).amin(1)
    tok_rank = ((dist <= best_true_d[:, None]) & ~is_true).sum(1)        # other-place entries at least as close (ties count)
    key_rank_ties = ((sims >= best_true[:, None]) & ~is_true).sum(1)
    j_best = torch.where(is_true, sims, torch.full_like(sims, -9)).argmax(1)
    lgap = (light[fi[j_best].cpu()] - light[t]).abs().to(DEV)
    hv = has.cpu().numpy()
    st, nt = support(xt); sd, nd = support(xp)
    st_old, _ = support(xt, before=True); sd_new, _ = support(xp, before=False)
    return {"winner_support": float(sup[b]), "winner_is_true": bool(np.abs(pts[b] - xt).max() < WIDTH),
            "winner_is_drifted": bool(np.abs(pts[b] - xp).max() < WIDTH),
            "true_support": float(st), "true_cells": nt, "true_support_from_entries_before_outage": float(st_old),
            "drifted_support": float(sd), "drifted_cells": nd, "drifted_support_from_entries_after_outage_start": float(sd_new),
            "cells_with_true_in_top32": float(at_true32.mean()), "cells_with_true_in_kept8": float(kept_true.mean()),
            "error": float(np.abs(xt - xp).max()), "winner_entry_age_median": float(np.median(ages)) if ages else None,
            "winner_same_screen_cell_share": float(np.mean(same_cell)) if same_cell else None,
            "entries": int(sims.shape[1]), "cells_with_true_entry": float(hv.mean()),
            "true_key_rank": key_rank_ties.cpu().numpy()[hv].tolist(), "true_token_rank": tok_rank.cpu().numpy()[hv].tolist(),
            "true_light_gap": lgap.cpu().numpy()[hv].tolist()}


def main(n_eps=23):
    parts = LOC.load()
    panel = torch.load(LM.DATA / "tokens_panel.pt")[:n_eps]
    rows = []
    for e, r in enumerate(panel):
        T, pos = r["T"], r["pos"]
        # the gates' events: outages of 8/16/32 steps before returns (drifted_returns' selection)
        evs = []
        first_seen = {}
        for t in range(T + 1):
            for k in range(LM.MAP):
                first_seen.setdefault((int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k])), t)
        t0, Ls = 9, [8, 16, 32]
        while t0 < T - 33:
            if (pos[t0] != pos[t0 - 1]).any():
                for s in range(t0 + 1, t0 + 33):
                    if any(first_seen[(int(pos[s][0]) + int(LM.KR[k]), int(pos[s][1]) + int(LM.KC[k]))] < t0 - 1 for k in range(LM.MAP)
                           if abs(int(pos[s][0]) + int(LM.KR[k]) - int(pos[s - 1][0])) > 3 or abs(int(pos[s][1]) + int(LM.KC[k]) - int(pos[s - 1][1])) > 4):
                        evs.append((t0, Ls[len(evs) % 3], s)); t0 = s + 17; break
            t0 += 1
        m = np.zeros(T + 1, bool)
        start_of = np.full(T + 1, -1)
        for a, L, s in evs:
            m[a:a + L] = True; start_of[a:min(T + 1, s + 17)] = a
        d = LOC.sensed(r); xtrue = torch.cumsum(d, 0).to(DEV)
        est = LOC.Estimator(parts, T, "full", True)
        p, v, act = r["p"].float(), r["v"].float(), r["prev_action"]
        with torch.no_grad():
            for t in range(T + 1):
                if t == 0:
                    x, var = torch.zeros(2, device=DEV), LOC.SENSED_VAR
                else:
                    x, var = est.predict(t, p[t - 1], v[t - 1], act[t], None if m[t] else d[t])
                if t > EXCL and start_of[t] >= 0:
                    est.X[t] = x
                    if float((x - xtrue[t]).abs().max()) > WIDTH:
                        st = stats(est, t, p[t], v[t], xtrue[t], start_of[t], p, r["light"].float()); st.update(ep=e, t=t, light=float(r["light"][t]))
                        rows.append(st)
                est.step(t, p[t], v[t], x, var, not m[t])
        print("episode", e, "rows", len(rows), flush=True)
    allrows = rows
    rows = [x for x in allrows if x["cells_with_true_entry"] > 0]              # frames that revisit: some cell has an entry at its true place
    A = lambda k: np.array([x[k] for x in rows], float)
    res = {"episodes": n_eps, "drifted_proposal_frames": len(allrows), "revisit_frames": len(rows),
           "winner_is_true_all_frames": float(np.mean([x["winner_is_true"] for x in allrows])),
           "winner_is_true": float(A("winner_is_true").mean()), "winner_is_drifted": float(A("winner_is_drifted").mean()),
           "true_support_median": float(np.median(A("true_support"))), "drifted_support_median": float(np.median(A("drifted_support"))),
           "winner_support_median": float(np.median(A("winner_support"))),
           "true_support_share_from_before_outage": float(A("true_support_from_entries_before_outage").sum() / max(A("true_support").sum(), 1e-9)),
           "drifted_support_share_from_after_outage_start": float(A("drifted_support_from_entries_after_outage_start").sum() / max(A("drifted_support").sum(), 1e-9)),
           "cells_with_true_in_top32_mean": float(A("cells_with_true_in_top32").mean()),
           "cells_with_true_in_kept8_mean": float(A("cells_with_true_in_kept8").mean()),
           "frames_true_beats_drifted": float((A("true_support") > A("drifted_support")).mean()),
           "frames_true_support_zero": float((A("true_support") == 0).mean()),
           "winner_entry_age_median": float(np.nanmedian(A("winner_entry_age_median"))),
           "winner_entry_age_q10_q90": [float(np.nanquantile(A("winner_entry_age_median"), .1)), float(np.nanquantile(A("winner_entry_age_median"), .9))],
           "winner_same_screen_cell_share": float(np.nanmean(A("winner_same_screen_cell_share"))),
           "entries_median": float(np.median(A("entries"))), "cells_with_true_entry_mean": float(A("cells_with_true_entry").mean())}
    kr = np.concatenate([x["true_key_rank"] for x in rows]); tr = np.concatenate([x["true_token_rank"] for x in rows])
    lg = np.concatenate([x["true_light_gap"] for x in rows])
    for name, sel in (("all", np.ones(len(lg), bool)), ("light_gap<0.05", lg < 0.05), ("light_gap>=0.05", lg >= 0.05)):
        res[f"true_entry_{name}"] = {"n": int(sel.sum()), "key_rank_median": float(np.median(kr[sel])),
                                     "key_rank_lt32": float((kr[sel] < 32).mean()), "key_rank_eq0": float((kr[sel] == 0).mean()), "token_rank_median": float(np.median(tr[sel])),
                                     "token_rank_lt32": float((tr[sel] < 32).mean()), "token_rank_eq0": float((tr[sel] == 0).mean())}
    for x in rows:
        for k_ in ("true_key_rank", "true_token_rank", "true_light_gap"):
            x.pop(k_)
    print(json.dumps(res, indent=1))
    (LOC.HERE / "diagnosis_recognition.json").write_text(json.dumps({"summary": res, "rows": allrows}, indent=1) + "\n")


def key_probe(n_eps=60):
    """What the trained key depends on, and whether its training positives share a screen cell (11-24)."""
    parts = LOC.load(); key, calib = parts["key"], parts["calib"]
    data = LOC.train_data()[:n_eps]
    cw = torch.nonzero(calib.world)[:, 0]; nW = len(cw)
    g = np.random.default_rng(0)
    same_cell, ages, n = 0.0, [], 0
    toks, lights, cells_q = [], [], []
    for r in data:
        T = r["T"]; P_, _ = LOC.episode_world(r, calib)
        pk = (P_[:, cw] * 4).round().long().numpy(); code = pk[..., 0] * 100003 + pk[..., 1]   # train_key's place code
        for _ in range(40):
            t = int(g.integers(EXCL + 1, T + 1)); a = int(g.integers(nW))
            m = code[: t - EXCL] == code[t, a]
            if not m.any():
                continue
            f, aa = np.nonzero(m)
            same_cell += (aa == a).mean(); n += 1; ages.append(np.median(t - f))
        toks.append(r["p"][::25, cw].float().reshape(-1, 16)); lights.append(r["v"][::25].float().repeat_interleave(nW, 0))
        cells_q.append(cw.repeat(len(r["p"][::25])))
    tok = torch.cat(toks)[:4000].to(DEV); li = torch.cat(lights)[:4000].to(DEV); ce = torch.cat(cells_q)[:4000].to(DEV)
    with torch.no_grad():
        k0 = key(tok, li, ce)
        sim = lambda k: float((k0 * k).sum(-1).mean())
        perm = lambda x, s: x[torch.randperm(len(x), generator=torch.Generator().manual_seed(s)).to(x.device)]
        other = cw.to(DEV)[torch.randint(nW, (len(ce),), generator=torch.Generator().manual_seed(1)).to(DEV)]
        res = {"training_positive_pairs_same_screen_cell_share": same_cell / n, "queries": n,
               "training_positive_age_median": float(np.median(ages)),
               "sim_same_token_light_other_screen_cell": sim(key(tok, li, other)),
               "sim_same_token_cell_other_lighting": sim(key(tok, perm(li, 2), ce)),
               "sim_same_light_cell_other_token": sim(key(perm(tok, 3), li, ce))}
    print(json.dumps(res, indent=1))
    (LOC.HERE / "diagnosis_key_probe.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["probe"]:
        key_probe()
    else:
        main(*(int(x) for x in sys.argv[1:]))
