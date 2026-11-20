"""C9 (PLAN.md): the world-eye's cache, its cheaper route, and what a stale eye looks at, on C2's DEV frames.
- Eyes: fresh (C2's), estimate cache (checkpoint k's estimate vs checkpoint 10's tokens), score cache (checkpoint k's
  estimate vs its own tokens), drift only (checkpoint 10's vs checkpoint k's tokens), and the chooser-proxy belief
  (checkpoint 10's estimate with Mamba fed the chooser's sampled fixations instead of the monitor's prior).
- Policies: contingency off (C2-C8), and deployed (the chooser's own C with the world's S); the chooser is its C + S.
- Readings: coverage, TV / argmax agreement to fresh, where looks land by region, look mass on windows the current
  world predicts well, the estimate cache against drift only; a figure; the cheaper route's cost.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261120_switching_dissection.c9_eye_cache [smoke]
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
CC = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c2_cost")
HERE = Path(__file__).resolve().parent
RUN = C2.RUN
CKPTS = (10, 9, 8, 5, 2)
LATE = (9, 8, 5)                                                          # checkpoint 2: the early-training check only
DEV = torch.device("cuda")
WIN = C.WIN.numpy()


@torch.no_grad()
def episode_at(model, cfg, dl, e, fix):
    """C2.episode with the fixation centres fix [T + 1, 2] instead of the monitor's prior."""
    T = int(dl[e].T)
    c = TR.tensors(D.gather([dl[e]], [[(0, 1, t, int(fix[t, 0]), int(fix[t, 1])) for t in range(T + 1)]]), DEV)
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
def chooser_scores(model, cfg, head, dl, e):
    """The chooser's per-window C and S for targets t = 8 .. T ([T - 7, 49] each), and the episode's frames."""
    T = int(dl[e].T)
    chunk = D.gather([dl[e]], [[(0, 1, t, 4, 4) for t in range(T + 1)]])
    with TR.autocast(DEV, cfg):
        p, v = C.coarse(model.perception, torch.as_tensor(chunk["frames"], device=DEV).flatten(0, 1))
    xx, y = C.inputs(p.reshape(1, T + 1, 81, 16).float(), v.reshape(1, T + 1, 8).float(), chunk, 8, 8)
    with torch.autocast("cuda", enabled=False):
        taken = head(xx); noop = head((xx[0], xx[1], torch.zeros_like(xx[2])))
    w = C.WIN.to(DEV)
    return ((taken - noop).square().sum(-1)[:, w].sum(-1).cpu(), (y - taken).square().sum(-1)[:, w].sum(-1).cpu(), chunk["frames"][0])


@torch.no_grad()
def cost(model, cfg):
    """The cheaper route at the trainer's shapes: the perception / record / belief pass and two chunked denoiser forwards
    at the context signal (the stand-in for the two-stream pass), without gradients."""
    d, prefix = cfg["data"], cfg["data"]["context_prefix"]
    ck = torch.load(RUN / "ckpt_10.pt", map_location="cpu", weights_only=False)
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], prefix, seed=CC.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    sc = cfg["edm"]["shortcut"]; k, tc = sc["k_max"], sc["context_signal"]
    out = {}
    for Sx in (64, 256):
        micro = d["accumulate_at_64"] if Sx == d["chunk_steps"][0] else d["accumulate_at_256"]
        ti, td = [], []
        torch.cuda.reset_peak_memory_stats()
        for rep in range(4):                                              # 1 warm-up, 3 timed
            chunk = streams.next(Sx)
            t0 = CC.sync(); c, P, rin, bp = CC.inputs(model, cfg, chunk, DEV); t1 = CC.sync()
            for _ in range(2):
                for r in np.array_split(np.arange(d["streams"]), micro):
                    r = torch.as_tensor(r, device=DEV)
                    x = tuple((1 - tc / k) * torch.randn_like(a[r].float()) + tc / k * a[r].float() for a in P.parts())
                    code = torch.full(P.p[r].shape[:2], M.shortcut_code(tc, k.bit_length() - 1, k), device=DEV)
                    with TR.autocast(DEV, cfg):
                        model.denoiser(x, (code, code), c["prev_action"][r], (model.belief.memory(bp[r]), rin[r]), P.fix[r], c["episode"][r], c["valid"][r])
            t2 = CC.sync()
            if rep:
                ti.append(t1 - t0); td.append(t2 - t1)
        out[Sx] = {"inputs_sec": float(np.mean(ti)), "two_forwards_sec": float(np.mean(td)),
                   "fraction_of_update": (float(np.mean(ti)) + float(np.mean(td))) / CC.BASE[Sx][0],
                   "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        print("cost", Sx, out[Sx], flush=True)
    out["projected_extra_hours"] = sum((out[s]["inputs_sec"] + out[s]["two_forwards_sec"]) * CC.BASE[s][1] for s in (64, 256)) / 3600
    out["fresh_in_loop_extra_hours_c2"] = 12.2
    return out


def figure(keys, frames, probs, cover, rws, night, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ent = cover["any_entity"][rws].any(-1)
    rng = np.random.default_rng(11)
    pick = [*rng.choice(np.where(ent & ~night)[0], 3, replace=False), *rng.choice(np.where(ent & night)[0], 3, replace=False)]
    names = list(probs)
    fig, ax = plt.subplots(len(pick), len(names), figsize=(2.6 * len(names), 2.6 * len(pick)))
    for i, n in enumerate(pick):
        img = frames[keys[n]]; H, W = img.shape[:2]
        for j, name in enumerate(names):
            pr = probs[name][n]; cellp = np.zeros(81)
            for wi in range(49):
                cellp[WIN[wi]] += pr[wi]
            a = ax[i, j]; a.imshow(img); a.imshow(cellp.reshape(9, 9), extent=(0, W, H, 0), cmap="magma", alpha=0.55, vmin=0, vmax=max(cellp.max(), 1e-6))
            r, cc = (C.CENTRES[int(pr.argmax())] - 1).tolist()
            a.add_patch(Rectangle((cc * W / 9, r * H / 9), 3 * W / 9, 3 * H / 9, fill=False, lw=2, ec="cyan"))
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(name, fontsize=9)
            if j == 0:
                a.set_ylabel(f"{'night' if night[n] else 'day'} e{keys[n][0]} t{keys[n][1]}", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(smoke=False):
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    groups, light, lookup, loff, cover = C1.semantics(L)
    dl = S.episodes(L)
    eps = np.sort(np.random.default_rng(20261120).choice(len(L["ep_T"]), 150, replace=False))   # C2's frames, exactly
    rng = np.random.default_rng(7)
    picks = {int(e): np.sort(rng.choice(np.arange(9, int(L["ep_T"][e]) + 1), min(20, int(L["ep_T"][e]) - 8), replace=False))
             for e in eps if L["ep_T"][e] >= 10}
    if smoke:
        picks = dict(list(picks.items())[:3])
    tok, est = {k: {} for k in CKPTS}, {k: {} for k in CKPTS}
    est_px, cs, ss, frames = {}, {}, {}, {}
    for k in CKPTS:
        model, cfg = AI.load(RUN / f"ckpt_{k:02d}.pt")
        model.requires_grad_(False).eval()
        if k == 10:
            pol = dict(cfg["fixation"]["chooser"]); m = cfg["p1"]["monitor"]
            head = C.Head(pol["history"]).to(DEV)
            head.load_state_dict(torch.load(RUN / "ckpt_10.pt", weights_only=False, map_location="cpu")["fixation_controller"]["head"]); head.eval()
        for e, ts in picks.items():
            ts = [int(t) for t in ts]
            st = C2.episode(model, cfg, dl, e)
            x = C2.estimates(model, cfg, st, e, ts).cpu().numpy()
            for t, xx in zip(ts, x):
                est[k][(e, t)] = xx.astype(np.float16); tok[k][(e, t)] = st[1].p[0, t].float().cpu().numpy().astype(np.float16)
            if k == 10:
                c, s, fr = chooser_scores(model, cfg, head, dl, e)
                T = int(L["ep_T"][e])
                fix = np.array([D.fixation_prior(np.random.default_rng([m["fixed_fixation_seed"], int(e), t]), ()) for t in range(T + 1)], dtype=np.int64)
                choice = torch.multinomial(C.policy(c, s, pol).double(), 1, generator=torch.Generator().manual_seed(int(e))).squeeze(1)
                fix[8:] = C.CENTRES[choice].numpy()                         # the chooser's own sampled looks from t = 8
                xp = C2.estimates(model, cfg, episode_at(model, cfg, dl, e, fix), e, ts).cpu().numpy()
                for t, xx in zip(ts, xp):
                    est_px[(e, t)] = xx.astype(np.float16); cs[(e, t)] = c[t - 8].numpy(); ss[(e, t)] = s[t - 8].numpy(); frames[(e, t)] = fr[t]
        if k == 10:
            res_cost = cost(model, cfg)
        del model; torch.cuda.empty_cache()
        print("checkpoint", k, len(est[k]), flush=True)
    keys = [(e, int(t)) for e in picks for t in picks[e]]
    rws = np.array([lookup[loff[e] + t] for e, t in keys]); night = light[rws] < 0.5
    arr = lambda d: np.stack([d[q] for q in keys]).astype(np.float32)
    T10 = arr(tok[10])
    cell = lambda a, b: ((a - b) ** 2).sum(-1)                              # [n, 81]
    wsum = lambda s: s[:, WIN].sum(-1)                                       # [n, 49]
    Cw, Sch = arr(cs), arr(ss)
    scores = {"fresh": cell(T10, arr(est[10])), "proxy_belief": cell(T10, arr(est_px))}
    for k in LATE:
        scores[f"estimate_cache_{10 - k}"] = cell(T10, arr(est[k]))
        scores[f"score_cache_{10 - k}"] = cell(arr(tok[k]), arr(est[k]))
        scores[f"drift_only_{10 - k}"] = cell(T10, arr(tok[k]))
    form = {"off": lambda sw: C.policy(torch.zeros(sw.shape), torch.as_tensor(sw), pol).numpy(),
            "deployed": lambda sw: C.policy(torch.as_tensor(Cw), torch.as_tensor(sw), pol).numpy()}
    fresh_w = wsum(scores["fresh"]); well = fresh_w <= np.median(fresh_w, -1, keepdims=True)

    def coverage(prob):
        out = {}
        for name, mask in cover.items():
            mk = mask[rws]; present = mk.any(-1); hit = mk[:, WIN].any(-1)
            lw = (prob * hit).sum(-1); uw = hit.mean(-1)
            for tag, sel in (("all", present), ("day", present & ~night), ("night", present & night)):
                out[f"{name}|{tag}"] = [float(lw[sel].mean()), float(uw[sel].mean()), int(sel.sum())]
        return out

    def looks(prob):
        cellp = np.zeros((len(prob), 81))
        for wi in range(49):
            cellp[:, WIN[wi]] += prob[:, wi:wi + 1] / 9
        return {g: {tag: float((cellp * groups[g][rws])[sel].sum(-1).mean()) for tag, sel in (("all", np.ones_like(night)), ("day", ~night), ("night", night))}
                for g in C1.GROUPS}

    def read(prob, ref):
        return {"coverage": coverage(prob), "looks": looks(prob), "tv_to_fresh": float(0.5 * np.abs(prob - ref).sum(-1).mean()),
                "argmax_agreement": float((prob.argmax(-1) == ref.argmax(-1)).mean()), "mass_on_well_predicted": float((prob * well).sum(-1).mean())}
    res = {"frames": len(keys), "night_frames": int(night.sum()), "eyes": {}, "uniform_looks": looks(np.full((len(keys), 49), 1 / 49))}
    probs = {f: {name: form[f](wsum(s)) for name, s in scores.items()} for f in form}
    chooser = C.policy(torch.as_tensor(Cw), torch.as_tensor(Sch), pol).numpy()
    for f in form:
        res["eyes"][f] = {name: read(p, probs[f]["fresh"]) for name, p in probs[f].items()}
        res["eyes"][f]["chooser"] = read(chooser, probs[f]["fresh"])
        for k in LATE:
            a, b = probs[f][f"estimate_cache_{10 - k}"], probs[f][f"drift_only_{10 - k}"]
            res["eyes"][f][f"estimate_cache_{10 - k}"]["tv_to_drift_only"] = float(0.5 * np.abs(a - b).sum(-1).mean())
    off = form["off"]                                                       # early training: checkpoint 2's caches read at checkpoint 5
    fresh5 = off(wsum(cell(arr(tok[5]), arr(est[5]))))
    res["early_3_passes_at_ckpt5"] = {"fresh_at_5": read(fresh5, fresh5),
                                      "score_cache_3": read(off(wsum(cell(arr(tok[2]), arr(est[2])))), fresh5),
                                      "estimate_cache_3": read(off(wsum(cell(arr(tok[5]), arr(est[2])))), fresh5)}
    res["mean_cell_error"] = {name: float(s.mean()) for name, s in scores.items()}
    res["cost"] = res_cost
    tag = "_smoke" if smoke else ""
    figure(keys, frames, {"chooser": chooser, "fresh world-eye": probs["deployed"]["fresh"], "5-stale estimate cache": probs["deployed"]["estimate_cache_5"],
                          "5-stale score cache": probs["deployed"]["score_cache_5"]}, cover, rws, night, HERE / f"c9_fixations{tag}.png")
    (HERE / f"c9_eye_cache{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    for f in form:
        print(f, flush=True)
        for name, r in res["eyes"][f].items():
            cv = r["coverage"]
            print(f"  {name:22s} tv {r['tv_to_fresh']:.3f} agree {r['argmax_agreement']:.2f} ent {cv['any_entity|all'][0]:.3f} zom {cv['zombies|all'][0]:.3f} "
                  f"cow {cv['cows|all'][0]:.3f} hud {cv['changed_hud|all'][0]:.3f} enter {cv['entering_map|all'][0]:.3f} well {r['mass_on_well_predicted']:.3f}"
                  + (f" tv_drift {r['tv_to_drift_only']:.3f}" if "tv_to_drift_only" in r else ""), flush=True)
    for name, r in res["early_3_passes_at_ckpt5"].items():
        print(f"  early {name:16s} tv {r['tv_to_fresh']:.3f} agree {r['argmax_agreement']:.2f} ent {r['coverage']['any_entity|all'][0]:.3f}", flush=True)
    print(json.dumps(res["cost"], indent=1), flush=True)


if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "smoke")
