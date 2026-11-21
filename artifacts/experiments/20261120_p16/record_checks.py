"""P1.6's record checks (PLAN.md, "Record checks"; the user 11-21): how the new eye behaves in training, and what the
foveal record brings at the night deletions, against P1.5.

    python -m artifacts.experiments.20261120_p16.record_checks eye                     F1-F4 (CPU)
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.record_checks stale            F5
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_p16.record_checks content [smoke]  R0-R2
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
TRC = importlib.import_module("artifacts.experiments.20261104_p1f2_review.trace")
SD = "artifacts.experiments.20261120_switching_dissection."
C4, C5 = (importlib.import_module(SD + n) for n in ("c4_sample", "c5_calibration"))
R = importlib.import_module("artifacts.experiments.20261120_p16.readings")
HERE = Path(__file__).resolve().parent
OUT = HERE / "readings_record"
RUNS = {"p15": HERE.parent / "20261116_p15/run", "p16": HERE / "run"}
DEV = torch.device("cuda")
FILL = 13_125                                                             # the world-eye's first update (pass 7)


def labelled(run):
    """C11b's set-up: the trace rows of the 400 labelled training episodes, their label rows, update and semantics."""
    ck = torch.load(run / "ckpt_10.pt", map_location="cpu", weights_only=False)
    raw = np.memmap(run / "gaze.bin", dtype=C.TRACE, mode="r", shape=(ck["fixation_controller"]["trace_rows"],))
    eps = D.load_split(ck["cfg"])["train"]; off = D.frame_offsets(eps)
    L = dict(np.load(S.LABELS / "s0_train400_labels.npz")); keys = {(e.shard, e.slot): i for i, e in enumerate(eps)}
    mapping = np.full(len(eps), -1, int)
    for le, key in enumerate(zip(L["ep_shard"], L["ep_slot"])):
        if tuple(key) in keys:
            mapping[keys[tuple(key)]] = le
    kept, pos = [], []
    for a in range(0, len(raw), 262144):
        q = raw[a:a + 262144]; ei = np.searchsorted(off, q["target"], side="right") - 1
        take = (q["target"] >= 0) & (mapping[ei] >= 0)
        kept.append(np.array(q[take])); pos.append(np.flatnonzero(take) + a)
    q, pos = np.concatenate(kept), np.concatenate(pos)
    ei = np.searchsorted(off, q["target"], side="right") - 1; ep = mapping[ei]; tt = q["target"] - off[ei]
    loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    TRC.A.L = L; TRC.A.E = G.episode_grids(L); TRC.A.OFF = loff; TRC.A.AOFF = np.r_[0, np.cumsum(L["ep_T"])]
    eall = np.concatenate([np.full(int(n), e) for e, n in enumerate(L["ep_T"])])
    tall = np.concatenate([np.arange(1, int(n) + 1) for n in L["ep_T"]]); si = loff[eall] + tall
    sem = TRC.training_semantics(L, eall, tall, si, loff)
    lookup = np.full(int(loff[-1]), -1, int); lookup[si] = np.arange(len(si)); rows = lookup[loff[ep] + tt]
    first = 12_500 * 16 * 64
    update = np.where(pos < first, pos // (16 * 64), 12_500 + (pos - first) // (16 * 256))
    night = L["s_light"][loff[ep] + tt] < 0.5
    return dict(q=q, rows=rows, sem=sem, update=update, night=night, eps=eps, mapping=mapping, off=off, L=L, ep=ep, tt=tt)


def eye():
    """F1-F4 per run, passes 1-6 against 7-10."""
    win, cen = C.WIN.numpy(), C.CENTRES.numpy()
    hud = (win >= 63).any(1); edge = (cen == 1).any(1) | (cen == 7).any(1)
    res = {}
    for name, run in RUNS.items():
        d = labelled(run); q, rows, sem = d["q"], d["rows"], d["sem"]
        w, s_, c_ = q["window"].astype(int), q["S"].astype(int), q["C"].astype(int)
        nxt = np.r_[(q["target"][1:] == q["target"][:-1] + 1) & (q["episode"][1:] == q["episode"][:-1]), False]   # row i+1 follows i
        ent = sem["masks"]["any_entity"][rows]                            # [n, 81]
        hit = lambda mk: mk[np.arange(len(mk))[:, None], win[w]].any(-1)
        out = {"mask_keys": sorted(sem["masks"])}
        for band, sel in (("passes_1_6", d["update"] < FILL), ("passes_7_10", d["update"] >= FILL)):
            r = {"rows": int(sel.sum())}
            for key in sorted(sem["masks"]):
                mk = sem["masks"][key][rows]; present = mk.any(-1); h = hit(mk)
                for lt, lm in (("day", ~d["night"]), ("night", d["night"])):
                    m = sel & present & lm
                    if m.sum() >= 100:
                        r[f"cover|{key}|{lt}"] = float(h[m].mean())
            r["hud_window_share"] = float(hud[w[sel]].mean()); r["edge_window_share"] = float(edge[w[sel]].mean())
            pw = np.bincount(w[sel], minlength=49) / sel.sum(); r["window_entropy_bits"] = float(-(pw[pw > 0] * np.log2(pw[pw > 0])).sum())
            m2 = sel & nxt
            r["repeat_previous_window"] = float((w[1:][m2[:-1]] == w[:-1][m2[:-1]]).mean())
            r["jump_chebyshev"] = float(np.abs(cen[w[1:][m2[:-1]]] - cen[w[:-1][m2[:-1]]]).max(-1).mean())
            he = hit(ent)
            m3 = m2[:-1] & he[:-1] & ent[1:].any(-1)
            r["entity_after_entity_look"] = float(he[1:][m3].mean())
            m4 = m2[:-1] & ~he[:-1] & ent[1:].any(-1)
            r["entity_after_other_look"] = float(he[1:][m4].mean())
            r["chosen_is_S_argmax"] = float((w[sel] == s_[sel]).mean()); r["chosen_is_C_argmax"] = float((w[sel] == c_[sel]).mean())
            r["S_argmax_is_C_argmax"] = float((s_[sel] == c_[sel]).mean())
            r["chosen_probability_mean"] = float(q["probability"][sel].mean())
            r["S_argmax_touches_hud"] = float(hud[s_[sel]].mean()); r["C_argmax_touches_hud"] = float(hud[c_[sel]].mean())
            r["S_argmax_has_entity"] = float(ent[sel][np.arange(sel.sum())[:, None], win[s_[sel]]].any(-1)[ent[sel].any(-1)].mean())
            r["C_argmax_has_entity"] = float(ent[sel][np.arange(sel.sum())[:, None], win[c_[sel]]].any(-1)[ent[sel].any(-1)].mean())
            out[band] = r
        res[name] = out
        print(name, json.dumps(out)[:2000], flush=True)
    return res


def stale():
    """F5: the world's error at checkpoints 6 and 10 on the labelled episodes, same frames and prior fixations."""
    d = labelled(RUNS["p16"]); win = C.WIN.numpy()
    eps = [e for i, e in enumerate(d["eps"]) if d["mapping"][i] >= 0]; idx = [i for i in range(len(d["eps"])) if d["mapping"][i] >= 0]
    off = D.frame_offsets(eps); looks = np.full(int(off[-1]), -1, np.int16)
    errs = {}
    for k in (6, 10):
        model, cfg = AI.load(RUNS["p16"] / f"ckpt_{k:02d}.pt"); model.requires_grad_(False).eval()
        path = OUT / f"stale_ckpt{k:02d}.npy"
        if not path.exists():
            C.world_errors(model, cfg, eps, off, looks, path, 2026112106)
        errs[k] = np.load(path, mmap_mode="r"); del model; torch.cuda.empty_cache()
    cache = np.load(RUNS["p16"] / "eye_cache_06.npy", mmap_mode="r")
    glob = np.concatenate([d["off"][i] + np.arange(eps[j].T + 1) for j, i in enumerate(idx)])
    L, loff = d["L"], np.r_[0, np.cumsum(d["L"]["ep_T"] + 1)]
    labs = np.concatenate([np.full(eps[j].T + 1, d["mapping"][i]) for j, i in enumerate(idx)])
    tts = np.concatenate([np.arange(eps[j].T + 1) for j in range(len(eps))])
    sem = d["sem"]; lookup = np.full(int(loff[-1]), -1, int)
    eall = np.concatenate([np.full(int(n), e) for e, n in enumerate(L["ep_T"])]); tall = np.concatenate([np.arange(1, int(n) + 1) for n in L["ep_T"]])
    lookup[loff[eall] + tall] = np.arange(len(eall)); rows = lookup[loff[labs] + tts]; ok = rows >= 0
    score = lambda e_: np.asarray(e_, dtype=np.float32)[:, win].sum(-1)          # [frames, 49]
    s6, s10, sc = score(errs[6])[ok], score(errs[10])[ok], score(cache[glob])[ok]
    rank = lambda a: a.argsort(1).argsort(1).astype(np.float32)
    spear = lambda a, b: float(np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(rank(a[::50]), rank(b[::50]))]))
    res = {"frames": int(ok.sum()), "argmax_agree_6_10": float((s6.argmax(1) == s10.argmax(1)).mean()),
           "argmax_agree_cache_6": float((sc.argmax(1) == s6.argmax(1)).mean()), "argmax_agree_cache_10": float((sc.argmax(1) == s10.argmax(1)).mean()),
           "spearman_6_10": spear(s6, s10), "spearman_cache_6": spear(sc, s6), "spearman_cache_10": spear(sc, s10),
           "total_error_6": float(s6.sum()), "total_error_10": float(s10.sum())}
    for key in ("any_entity", "changed_hud", "entering_map"):
        if key in sem["masks"]:
            mk = sem["masks"][key][rows[ok]]; present = mk.any(-1)
            for nm, s_ in (("cache", sc), ("ckpt6", s6), ("ckpt10", s10)):
                top = win[s_.argmax(1)]
                res[f"top_window_has_{key}|{nm}"] = float(mk[np.arange(len(mk))[:, None], top].any(-1)[present].mean())
    return res


def fine_reader(model, cfg):
    """R0: S1's token probe on glimpse tokens at their window's map cells (train400, this encoder). Not under no_grad:
    the probe trains (C14's slip)."""
    Ltr = dict(np.load(S.LABELS / "s0_train400_labels.npz"))
    with torch.no_grad():
        enc = S.encode(model, Ltr, cfg)
    cols = {k: [] for k in ("tok", "v", "kind", "terr", "ep")}
    for e, ep in enumerate(G.episode_grids(Ltr)):
        r = enc[e]; cells = M.window(r["fix"]).numpy(); n = len(ep["grid"]); ok = cells < 63   # [n, 9]
        fr = np.repeat(np.arange(n), 9).reshape(n, 9)
        kind = np.where(ok, ep["grid"].reshape(n, 63)[fr, cells.clip(max=62)], -1)
        terr = np.where(ok & (kind == 0), ep["view"].reshape(n, 63)[fr, cells.clip(max=62)], -1)
        cols["tok"].append(r["g"][ok]); cols["v"].append(r["v"][fr[ok]]); cols["kind"].append(torch.as_tensor(kind[ok].astype(np.int64)))
        cols["terr"].append(torch.as_tensor(terr[ok].astype(np.int64))); cols["ep"].append(np.full(int(ok.sum()), e))
    tr = {k: (torch.cat(v) if isinstance(v[0], torch.Tensor) else np.concatenate(v)) for k, v in cols.items()}
    dv = {k: v[:1000] for k, v in tr.items()}
    del enc
    return S.token_probe(tr, dv, 0)[0]


@torch.no_grad()
def unit_pass(model, cfg, dl, e, t, centre):
    """C2.episode over frames 0..t with frame t - 1's fixation set to `centre` (None: the prior's), and the record state
    after frame t - 2."""
    m = cfg["p1"]["monitor"]
    fx = [D.fixation_prior(np.random.default_rng([m["fixed_fixation_seed"], int(e), k]), ()) for k in range(t + 1)]
    if centre is not None:
        fx[t - 1] = np.array(centre)
    c = TR.tensors(D.gather([dl[e]], [[(0, 1, k, *fx[k]) for k in range(t + 1)]]), DEV)
    with TR.autocast(DEV, cfg):
        Pe = M.Percept(*(torch.cat(z, 1) for z in zip(*[model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                                                        for a in range(0, t + 1, 512)])), c["fixation"])
        state, carry, rins, ys, prev = model.record.zero(1, DEV), model.belief.initial(1, DEV), [], [], None
        for a in range(0, t + 1, 512):
            sl = slice(a, min(t + 1, a + 512))
            first = torch.zeros(1, sl.stop - a, dtype=torch.bool, device=DEV); first[:, 0] = a == 0
            ri, si = model.record.scan(model.record.input(Pe.p[:, sl], Pe.g[:, sl], Pe.fix[:, sl]), first, state); state = si[:, -1]
            if a <= t - 2 < sl.stop:
                prev = si[:, t - 2 - a].clone()
            y, carry = model.belief(model.belief.tokens(model.belief.percept_embed(M.Percept(Pe.p[:, sl], Pe.g[:, sl], Pe.v[:, sl], Pe.fix[:, sl]), si),
                                                        c["reward"][:, sl], c["terminated"][:, sl], c["action"][:, sl]), carry)
            rins.append(ri.to(torch.bfloat16)); ys.append(y[:, 1::2])
        y = torch.cat(ys, 1)
    return (c, Pe, torch.cat(rins, 1), torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], 1)), prev


@torch.no_grad()
def write_read(model, cfg, Pe, t, prev, blank):
    """The denoiser's read at frame t from the state after t - 2 and frame t - 1's write (blank: no glimpse written)."""
    g = Pe.g[:, t - 1:t]
    with TR.autocast(DEV, cfg):
        x = model.record.input(Pe.p[:, t - 1:t], torch.zeros_like(g) if blank else g, Pe.fix[:, t - 1:t])
        return model.record.scan(x, torch.zeros(1, 1, dtype=torch.bool, device=DEV), prev)[1][0, 0].to(torch.bfloat16)   # [81, dim]


def content(smoke=False):
    """R0-R2 on C5's night units (PLAN)."""
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L); us_all = R.units(L); m = None
    order = []                                                            # calibration's order (its skip rule)
    for e in sorted({x[0] for x in us_all}):
        for x in [u for u in us_all if u[0] == e]:
            _, t, kind, i, j, night = x
            dpos = L["s_pos"][loff[e] + t] - L["s_pos"][loff[e] + t - 1]
            cells = [r * 9 + c for r in range(i - dpos[0] - 1, i - dpos[0] + 2) for c in range(j - dpos[1] - 1, j - dpos[1] + 2) if 0 <= r < 7 and 0 <= c < 9]
            if cells:
                order.append((x, cells, dpos))
    night_ix = [n for n, (x, _, _) in enumerate(order) if x[5]]
    if smoke:
        eps2 = sorted({order[n][0][0] for n in night_ix})[:2]; night_ix = [n for n in night_ix if order[n][0][0] in eps2]
    res, cov = {"units": len(night_ix), "smoke_checks": {}}, []
    mon = torch.load(RUNS["p16"] / "ckpt_10.pt", map_location="cpu", weights_only=False)["cfg"]["p1"]["monitor"]
    for n in night_ix:                                                    # R1: coverage under the evaluated prior fixations
        (e, t, kind, i, j, _), _, dpos = order[n]
        fx = {k: D.fixation_prior(np.random.default_rng([mon["fixed_fixation_seed"], int(e), k]), ()) for k in range(t - 8, t)}
        world = L["s_pos"][loff[e] + t - 1] + np.array([i, j])
        scr = lambda k: world - L["s_pos"][loff[e] + k]
        inwin = lambda k: (lambda s: 0 <= s[0] < 7 and 0 <= s[1] < 9 and int(s[0] * 9 + s[1]) in M.window(torch.as_tensor(fx[k])).tolist())(scr(k))
        st_now = L["s_pos"][loff[e] + t]
        cov.append({"looked_t1": inwin(t - 1), "covered8": any(inwin(k) for k in range(t - 8, t)),
                    "aligned8": any(inwin(k) and (L["s_pos"][loff[e] + k] == st_now).all() for k in range(t - 8, t))})
    for name in ("p16", "p15"):
        model, cfg = AI.load(RUNS[name] / "ckpt_10.pt"); model.requires_grad_(False).eval()
        Rk = {"kind": S.load_reader(RUNS[name] / "s1/results.reader.pt", RUNS[name] / "ckpt_10.pt", DEV)}
        fine = fine_reader(model, cfg); print(name, "fine reader fitted", flush=True)
        prior = np.load(HERE / f"readings/calibration_{name}.npz")["kept"]
        n0 = M.Record.size(cfg, coarse=True); rows = []
        for n in night_ix:
            (e, t, kind, i, j, _), cells, dpos = order[n]
            centre = (i, j)
            assert i * 9 + j in M.window(torch.as_tensor(np.array(centre))).tolist()
            st, prev = unit_pass(model, cfg, dl, e, t, centre)
            c, Pe, rin, bp = st
            arms = {"A": rin}
            if smoke:                                                     # the prefix pass is C2's; the one-frame read is the pass's
                stp, _ = unit_pass(model, cfg, dl, e, t, None)
                pk = np.stack([(C5.kind_probs(Rk, *C4.samples(model, cfg, stp, e, [t], d)).argmax(-1)[0, cells] == kind).any().cpu().numpy() for d in range(C5.DRAWS)])
                sc = res["smoke_checks"].setdefault(name, {"prior_pass_equal_draws": 0, "draws": 0, "one_frame_scan_max_diff": 0.0})
                sc["prior_pass_equal_draws"] += int((pk == prior[n]).sum()); sc["draws"] += C5.DRAWS
                if model.record.fovea:
                    sc["one_frame_scan_max_diff"] = max(sc["one_frame_scan_max_diff"], float((write_read(model, cfg, Pe, t, prev, False).float() - rin[0, t].float()).abs().max()))
            if model.record.fovea:
                rb = rin.clone(); rb[0, t] = write_read(model, cfg, Pe, t, prev, True); arms["B"] = rb
                rc = rin.clone(); rc[0, t, :, n0:] = 0; arms["C"] = rc
            kept = {a: np.stack([(C5.kind_probs(Rk, *C4.samples(model, cfg, (c, Pe, r_, bp), e, [t], d)).argmax(-1)[0, cells] == kind).any().cpu().numpy()
                                 for d in range(C5.DRAWS)]) for a, r_ in arms.items()}
            slot = M.window(Pe.fix[0, t - 1]).tolist().index(i * 9 + j)
            fs = int(fine(torch.cat([Pe.g[0, t - 1, slot][None].half(), Pe.v[0, t - 1][None].half()], 1))["kind"].argmax(1)) == kind
            pk_ = lambda k, cell: int(C5.kind_probs(Rk, Pe.p[0, k][None].float(), Pe.v[0, k][None].float()).argmax(-1)[0, cell]) == kind
            world = L["s_pos"][loff[e] + t - 1] + np.array([i, j])
            hist = [pk_(k, int(s[0] * 9 + s[1])) for k in range(t - 8, t) for s in [world - L["s_pos"][loff[e] + k]] if 0 <= s[0] < 7 and 0 <= s[1] < 9]
            rows.append({"n": n, "kind": kind, "edge": min(i, 6 - i, j, 8 - j) == 1, "moved": bool((dpos != 0).any()),
                         "fine_shows": fs, "coarse_t1_shows": pk_(t - 1, i * 9 + j), "coarse_hist_shows": any(hist), "prior": prior[n],
                         **{f"kept_{a}": v for a, v in kept.items()}})
            if len(rows) % 50 == 0:
                print(name, len(rows), flush=True)
        res[name] = summarise(rows, cov)
        np.savez_compressed(OUT / f"content_{name}{'_smoke' if smoke else ''}.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]},
                            **{f"cov_{k}": np.array([c_[k] for c_ in cov]) for k in cov[0]})
        print(name, json.dumps(res[name])[:3000], flush=True)
        del model, fine; torch.cuda.empty_cache()
    return res


def summarise(rows, cov):
    """Deletion per arm and paired differences by class; coverage by deletion outcome."""
    arms = [k[5:] for k in rows[0] if k.startswith("kept_")]
    dl = {a: 1 - np.array([r[f"kept_{a}"].mean() for r in rows]) for a in arms + ["prior"] if a != "prior"}
    dl["prior"] = 1 - np.array([r["prior"].mean() for r in rows])
    fs, ct, ch = (np.array([r[k] for r in rows]) for k in ("fine_shows", "coarse_t1_shows", "coarse_hist_shows"))
    kinds, edge, moved = (np.array([r[k] for r in rows]) for k in ("kind", "edge", "moved"))
    classes = {"all": np.ones(len(rows), bool), "adds_evidence": fs & ~ct, "adds_beyond_history": fs & ~ct & ~ch,
               "both_show": fs & ct, "crop_misses": ~fs, "edge": edge, "inner": ~edge, "still": ~moved, "moved": moved,
               "cows": kinds == 2, "zombies": kinds == 1}
    out = {"classes": {}}
    for cname, sel in classes.items():
        if sel.sum() == 0:
            continue
        r = {"units": int(sel.sum()), **{f"deletion_{a}": float(v[sel].mean()) for a, v in dl.items()}}
        pair = lambda a, b: (float((dl[a] - dl[b])[sel].mean()), float((dl[a] - dl[b])[sel].std(ddof=1) / np.sqrt(sel.sum())) if sel.sum() > 1 else None)
        for a, b in (("B", "A"), ("C", "A"), ("A", "prior")):
            if a in dl and b in dl:
                r[f"{a}_minus_{b}"] = pair(a, b)
        out["classes"][cname] = r
    always = np.array([r["prior"].sum() == 0 for r in rows]); never = np.array([r["prior"].all() for r in rows])
    cv = {k: np.array([c_[k] for c_ in cov]) for k in cov[0]}
    out["coverage"] = {g: {"units": int(s.sum()), **{k: int(v[s].sum()) for k, v in cv.items()}}
                       for g, s in (("deleted_all_8", always), ("deleted_some", ~always & ~never), ("deleted_none", never))}
    return out


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    what = sys.argv[1]; smoke = "smoke" in sys.argv[2:]
    if what != "eye":
        assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    res = {"eye": eye, "stale": stale, "content": lambda: content(smoke)}[what]()
    (OUT / f"{what}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1)[:5000], flush=True)
