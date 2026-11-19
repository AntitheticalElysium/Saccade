"""S2 stage 1b on P1.5 (PLAN.md "Stage 1b"): the readings stage 1 missed, and the route localisation of the mob loss.
s2.py's readers and rollout, unchanged in arithmetic; the plan's first action now also enters the prefix belief (equal
to the recorded one on stage 1's bank).

    python -m artifacts.experiments.20261119_s2.s2b bank                 # CPU: the competent bank
    python -m artifacts.experiments.20261119_s2.s2b oracle  [N]          # the simulator's futures of it, read
    python -m artifacts.experiments.20261119_s2.s2b model   [N]          # the world's futures of it, read
    python -m artifacts.experiments.20261119_s2.s2b routes  [N]          # stage 1's bank: which route's self-feeding
    python -m artifacts.experiments.20261119_s2.s2b gates   [N]          # CPU

bank    competent starts (DEV episodes of collector epsilon 0.1 / 0.25 / 0.5, start step >= 7, admitted by position or
        by the start state only, never by the future): (a) 1,000 uniform starts with fresh uniform random 16-step plans;
        (b) hold pairs: starts where the player is awake and faces a block it can collect (tree; stone, coal with a wood
        pickaxe; iron with a stone pickaxe; diamond with an iron pickaxe; water), the material below 9, up to 150 per
        material, each run under two plans, DO or NOOP first, then the same 15 uniform random actions (Codex: mine
        against wait). The reference is the simulator under the same plan: the recorded future is not used.
routes  stage 1's terminal-inclusive bank (1,000 starts, 4 futures): free running; the denoiser's context fed the
        recorded frames (ctx_real); Mamba and the record fed them (mem_real); both (teacher). The frame scored is
        always the generated one; recorded inputs only while the recording lives.
gates   on the competent random-plan starts: L1, the S3 bookkeeping blocker, L2b, consistency; on the holds: the
        action's effect (DO minus NOOP) on the collected quantity by steps 1 / 4 / 16 and on other outcomes, world
        against simulator; on both banks: spread (randomised PIT of each simulator future among the world's 8; the null
        PIT of oracle B among oracle A beside it) and revisits (TEST_SPEC S2.2: a cell re-entering view whose terrain was
        clearly seen before, in the real 64-frame prefix or the rollout, agreeing, by gap 1-16 / 17-64 / 65-80).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MAMBA_DETERMINISTIC", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import dataclasses, importlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from saccade import chooser as C, model as M, train as TR
from saccade.eval import gates as G, oracle as O, s1 as S
from saccade.eval.probe import KEEP

S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
HERE, DEV, H, MF, NB = S2.HERE, S2.DEV, S2.H, S2.MF, S2.NB
DIRS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])            # constants.DIRECTIONS[:5]
# faced block -> (HUD index of the collected quantity, HUD index of the tool needed or None)
COLLECT = {5: (4, None), 4: (5, 10), 8: (6, 10), 9: (7, 11), 10: (8, 12), 3: (2, None)}
MATERIAL = {5: "tree", 4: "stone", 8: "coal", 9: "iron", 10: "diamond", 3: "water"}
KEEP_T = torch.as_tensor(KEEP)


def build_bank():
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rng = np.random.default_rng(20261120)
    eps = np.flatnonzero(L["ep_eps"] < 0.99)
    cand = np.array([(e, t) for e in eps for t in range(7, int(L["ep_T"][e]))])
    rows, plans, kind, pair, mat = [], [], [], [], []
    base = lambda e, t: (e, int(L["ep_shard"][e]), int(L["ep_slot"][e]), t, int(L["ep_T"][e]), 1)
    for e, t in cand[np.sort(rng.choice(len(cand), 1000, replace=False))]:
        rows.append(base(int(e), int(t))); plans.append(rng.integers(17, size=H)); kind.append(0); pair.append(-1); mat.append(-1)
    s = so[cand[:, 0]] + cand[:, 1]
    d = DIRS[L["s_dir"][s]]
    face = L["s_view"][s, 3 + d[:, 0], 4 + d[:, 1]]
    hud = np.concatenate([L["s_body"], L["s_inv"]], 1)[s]
    counts = {}
    for b, (q, tool) in COLLECT.items():
        ok = (face == b) & ~L["s_sleep"][s].astype(bool) & (hud[:, q] < 9) & ((hud[:, tool] > 0) if tool is not None else True)
        idx = np.flatnonzero(ok)
        idx = np.sort(rng.choice(idx, min(150, len(idx)), replace=False)) if len(idx) else idx
        counts[MATERIAL[b]] = {"eligible": int(ok.sum()), "taken": int(len(idx))}
        for i in idx:
            tail, pid = rng.integers(17, size=H - 1), int(max(pair) + 1 if pair else 0)
            for first, k in ((5, 1), (0, 2)):                               # DO, NOOP: consecutive rows
                rows.append(base(int(cand[i, 0]), int(cand[i, 1]))); plans.append(np.r_[first, tail])
                kind.append(k); pair.append(pid); mat.append(b)
    rows, plans = np.array(rows, np.int32), np.array(plans, np.int8)
    np.savez_compressed(HERE / "bank_competent.npz", rows=rows, plans=plans, kind=np.array(kind, np.int8),
                        pair=np.array(pair, np.int32), material=np.array(mat, np.int8))
    out = {"random_plan_starts": int((np.array(kind) == 0).sum()), "hold_pairs": counts,
           "episodes": int(len(np.unique(rows[:, 0]))), "epsilon": {float(x): int((L["ep_eps"][rows[:, 0]] == x).sum()) for x in np.unique(L["ep_eps"][rows[:, 0]])},
           "achievements_at_start_median": float(np.median(L["s_ach"][so[rows[:, 0]] + rows[:, 3]].sum(1)))}
    (HERE / "bank_competent.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def competent(n=None):
    b = np.load(HERE / "bank_competent.npz")
    sl = slice(None) if n is None else np.r_[0:n // 2, 1000:1000 + 2 * (n // 4)]   # smoke: random-plan rows and hold pairs
    return b["rows"][sl], b["plans"][sl].astype(np.int64), {k: b[k][sl] for k in ("kind", "pair", "material")}


@torch.no_grad()
def terrain(R, p, v):
    """The entity reader's terrain head on map cells [N, S, 7, 9], -1 where it reads an entity or at the player."""
    N, Sn = p.shape[:2]
    x = torch.cat([p[:, :, :63].reshape(-1, 16), v[:, :, None].expand(-1, -1, 63, -1).reshape(-1, 8)], -1).half()
    o = [R["kind"](c) for c in x.split(262144)]
    terr = torch.cat([q["terr"].argmax(1) for q in o]).reshape(N, Sn, 63)
    kind = torch.cat([q["kind"].argmax(1) for q in o]).reshape(N, Sn, 63)
    terr = torch.where((kind > 0) | ~KEEP_T.to(kind.device), torch.full_like(terr, -1), terr)
    return terr.reshape(N, Sn, 7, 9).cpu().numpy().astype(np.int8)


def oracle(n=None):
    import jax, jax.numpy as jnp
    model, cfg = S2.load()
    R = S2.load_readers(model, cfg)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, plans, _ = competent(n)
    samp, out, pend = S2.sim_sampler(), {}, []

    def flush():
        fr = torch.as_tensor(np.stack([f for _, f, _ in pend]), device=DEV)
        with TR.autocast(DEV, cfg):
            p, v = C.coarse(model.perception, fr.flatten(0, 1))
        pos0 = np.concatenate([np.repeat(pp[None], 16, 0) for _, _, pp in pend])
        r = S2.read(R, p.float(), v.float(), pos0)
        te = terrain(R, p.float(), v.float())
        for j, (w, _, _) in enumerate(pend):
            S2.store(out, "rd", w, *(x[16 * j:16 * (j + 1)] for x in r), None, None, None)
            out.setdefault("rd_terr", {})[w] = te[16 * j:16 * (j + 1)]
        pend.clear()
    t_start = time.time()
    for w, (e, sh, slt, t0, T, _) in enumerate(rows):
        state = O.RP.advance_to(int(sh), int(slt), int(t0))
        keys = jax.random.split(jax.random.PRNGKey(3 * 10 ** 6 + w), O.M * H).reshape(O.M, H, 2)
        Rr, fr, rr, dd = jax.tree.map(np.asarray, samp(state, jnp.asarray(plans[w].astype(np.int32)), keys))
        lab = [O.to_obs(jax.tree.map(lambda q: q[i], Rr)) for i in range(O.M)]
        S2.store(out, "lab", w, np.stack([g for g, _, _ in lab]),
                 np.minimum(np.concatenate([Rr["body"], Rr["inv"]], -1), 9).astype(np.int8), Rr["sleep"].astype(bool),
                 np.stack([p for _, _, p in lab]).astype(np.int16), rr.astype(np.float32), dd.astype(bool), Rr["light"].astype(np.float32))
        pend.append((w, fr, lab[0][2][0]))
        if len(pend) == 8:
            flush()
        if (w + 1) % 200 == 0:
            print(f"start {w + 1}/{len(rows)} {time.time() - t_start:.0f}s", flush=True)
    if pend:
        flush()
    np.savez_compressed(HERE / ("oracle_competent.npz" if n is None else "oracle_competent_smoke.npz"), **S2.stack(out, len(rows)))


def prefix_terrain(model, cfg, R, enc, rows, sub, grids, back=64):
    """Per start: the real prefix's last `back` + 1 frames read by the terrain head, and their true positions."""
    out = []
    for e, _, _, t0, _, _ in rows:
        en, a = enc[sub[int(e)]], max(0, int(t0) - back)
        te = terrain(R, en["p"][a:t0 + 1].to(DEV).float()[None], en["v"][a:t0 + 1].to(DEV).float()[None])[0]
        out.append((te, grids[int(e)]["pos"][a:t0 + 1].astype(int)))
    return out


def rollout(model, cfg, R, head, pol, enc, sub, grids, rows, plans, mode="free", mf=MF, seed=0, gaze="chooser"):
    """s2.model_futures' loop for one bank. mode (recorded banks only): which routes read the recorded frame instead of
    the generated one while the recording lives: ctx_real (the denoiser's cache), mem_real (Mamba and the record),
    teacher (both); the split (11-19): record_real (the record alone: the denoiser's per-cell read and Mamba's read of
    it), mamba_real (Mamba's percept input alone); the readers (11-19): dread_real (only the denoiser's per-cell read
    of the record reads the recorded frames), mread_real (only Mamba's read of it, record_in); there the two readers
    keep their own record states, otherwise one record (rm is rs); every arm feeds the belief the world's own sampled reward and termination, so the arms differ in
    frames only (Codex's review). gaze: "chooser" (S2's contract) or "recorded" (the recording's own fixation schedule,
    so a recorded glimpse and a generated one sit at the same crop: the route arms)."""
    win, centres, Hh = C.WIN.to(DEV), C.CENTRES.to(DEV), pol["history"]
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    P = cfg["data"]["context_prefix"]
    out, kc = {}, np.minimum(rows[:, 3], P - 1) + 1
    batches = [b for k in np.unique(kc) for b in np.array_split(np.flatnonzero(kc == k), max(1, -(-int((kc == k).sum()) // NB)))]
    t_start = time.time()
    for bi, ws in enumerate(batches):
        rr = [(sub[int(rows[w, 0])], int(rows[w, 3])) for w in ws]; k_ctx = int(kc[ws[0]])
        Bn = len(ws) * mf
        rep = lambda x: x.repeat_interleave(mf, 0)
        at = lambda key, dt: torch.stack([enc[i][key][min(max(t + dt, 0), len(enc[i][key]) - 1)] for i, t in rr]).to(DEV)
        cs, bps, rl = [], [], []
        for (i, t), w in zip(rr, ws):
            en = enc[i]
            act = en["action"][:t + 1].clone(); act[t] = int(plans[w, 0])   # the plan's first action
            with TR.autocast(DEV, cfg):
                pp = M.Percept(*(en[k][:t + 1].to(DEV).float()[None] for k in ("p", "g", "v")), en["fix"][:t + 1].to(DEV)[None])
                reads, _, last = S.history(model, pp.p)
                y, c = model.belief(model.belief.tokens(S.embed(model, pp, reads), en["reward"][:t + 1].to(DEV)[None],
                                    en["terminated"][:t + 1].to(DEV)[None], act.to(DEV)[None]), model.belief.initial(1, DEV))
            err = float((y[0, -2].float() - en["b"][t].to(DEV).float()).square().mean().sqrt())   # the corrected belief at t0
            assert err < 0.1, ("prefix belief differs from encode's", err)
            out.setdefault("m_prefix_error", {})[int(w)] = np.float32(err)
            cs.append(c); bps.append(y[:, -1].float()); rl.append(last)
        carry = tuple(type(cs[0][l])(*(rep(torch.cat([getattr(c[l], f.name) for c in cs])) for f in dataclasses.fields(cs[0][l])))
                      for l in range(len(cs[0])))
        bp, rs = rep(torch.cat(bps)), rep(torch.cat(rl))
        rm = rs                                                          # Mamba's record state (scan never writes in place)
        split = mode in ("dread_real", "mread_real")
        rin = S.window(model, enc, rr, P - 1)
        g = torch.Generator(device=DEV).manual_seed(20261119 + seed + bi); gg = torch.Generator(device=DEV).manual_seed(30261119 + seed + bi)
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
        fix = rep(at("fix", 0)); alive = torch.ones(Bn, dtype=torch.bool, device=DEV)
        lives = rep(torch.as_tensor([[t + j <= int(rows[w, 4]) for j in range(H + 1)] for (_, t), w in zip(rr, ws)], device=DEV))
        ps, vs = [rep(at("p", 0)).float()], [rep(at("v", 0)).float()]
        rec = {k: [] for k in ("reward", "done", "fix", "bhud", "bsleep")}
        for j in range(1, H + 1):
            pa = plan[:, j - 1]
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
                if gaze == "chooser":
                    parts, fx = M.sample(model.denoiser, dims, pa[:, None], mem, fix[:, None], cache, cfg, g, choose=choose)
                    fix = fx[:, 0]
                else:
                    fix = rep(at("fix", j))
                    parts = M.sample(model.denoiser, dims, pa[:, None], mem, fix[:, None], cache, cfg, g)
                parts = tuple(x[:, 0].float() for x in parts)
                real = lives[:, j] if mode != "free" else torch.zeros(Bn, dtype=torch.bool, device=DEV)
                actual = tuple(rep(at(k, j)).float() for k in ("p", "g", "v")) if mode != "free" else parts
                mix = lambda use: tuple(torch.where(use.view(-1, *([1] * (a.dim() - 1))), a, b) for a, b in zip(actual, parts))
                rec_ev = mix(real & torch.tensor(mode in ("mem_real", "teacher", "record_real", "dread_real"), device=DEV))
                mrec_ev = mix(real & torch.tensor(mode in ("mem_real", "teacher", "record_real", "mread_real"), device=DEV))
                mam_ev = mix(real & torch.tensor(mode in ("mem_real", "teacher", "mamba_real"), device=DEV))
                cx = mix(real & torch.tensor(mode in ("ctx_real", "teacher"), device=DEV))
                reads0, _, _ = S.history(model, parts[0], state=rm)
                h = model.outcome.features(bp, S.embed(model, M.Percept(*parts, fix), reads0)).float()
                r = model.outcome.sample_reward(h, g)
                done = torch.rand(Bn, device=DEV, generator=g) > torch.sigmoid(model.outcome.cont_logit(h, r).float())
                r_in, d_in = r, done
                reads, _, next_rm = S.history(model, mrec_ev[0], state=rm)
                next_rs = S.history(model, rec_ev[0], state=rs)[2] if split else next_rm
                emb = S.embed(model, M.Percept(*mam_ev, fix), reads)
                nxt = plan[:, j] if j < H else torch.zeros_like(pa)
                yy, carry = model.belief(model.belief.tokens(emb[:, None], r_in[:, None], d_in[:, None], nxt[:, None]), carry)
                cache = M.context_pass(model.denoiser, tuple(a[:, None] for a in cx), pa[:, None], mem, fix[:, None], cache, cfg, g)
                ob = R["belief"](yy[:, 0].float())
            alive = alive & ~done
            rec["reward"].append(r.float()); rec["done"].append(~alive); rec["fix"].append(fix)
            rec["bhud"].append(ob["hud"].reshape(Bn, 16, 10).argmax(-1)); rec["bsleep"].append(ob["sleep"].argmax(-1).bool())
            ps.append(parts[0]); vs.append(parts[2])
            bp, rs, rm = yy[:, 1].float(), next_rs, next_rm
            for k, x in (("p", cx[0]), ("v", cx[2]), ("prev_action", pa), ("reward", r_in.float()), ("terminated", d_in.float())):
                hist[k] = torch.cat([hist[k][:, 1:], x[:, None].to(hist[k].dtype)], 1)
            ok = torch.cat([ok[:, 1:], torch.ones_like(ok[:, :1])], 1)
        pos0 = np.repeat(np.stack([grids[int(rows[w, 0])]["pos"][int(rows[w, 3])] for w in ws]), mf, 0)
        pst, vst = torch.stack(ps, 1), torch.stack(vs, 1)
        gr, hu, sl, po = S2.read(R, pst, vst, pos0)
        te = terrain(R, pst, vst)
        z = {k: torch.stack(x, 1).cpu().numpy() for k, x in rec.items()}
        for a, w in enumerate(ws):
            s = slice(mf * a, mf * (a + 1))
            S2.store(out, "m", int(w), gr[s], hu[s], sl[s], po[s], z["reward"][s], z["done"][s], None)
            for k, x in (("terr", te[s]), *((k, z[k][s]) for k in ("fix", "bhud", "bsleep"))):
                out.setdefault(f"m_{k}", {})[int(w)] = x
        print(f"{mode} batch {bi + 1}/{len(batches)} {time.time() - t_start:.0f}s", flush=True)
    return out


def setup():
    model, cfg = S2.load(); S2.EV.guard()
    R = S2.load_readers(model, cfg)
    st = torch.load(S2.CK, weights_only=False, map_location="cpu")["fixation_controller"]
    pol = dict(cfg["fixation"]["chooser"])
    head = C.Head(pol["history"]).to(DEV); head.load_state_dict(st["head"]); head.eval().requires_grad_(False)
    return model, cfg, R, head, pol


def model_competent(n=None):
    model, cfg, R, head, pol = setup()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, plans, _ = competent(n)
    eps = np.unique(rows[:, 0]); sub = {e: i for i, e in enumerate(eps)}
    enc = S.encode(model, S2.sublabels(L, eps), cfg)
    grids = G.episode_grids(L)
    out = rollout(model, cfg, R, head, pol, enc, sub, grids, rows, plans)
    pt = prefix_terrain(model, cfg, R, enc, rows, sub, grids)
    np.savez_compressed(HERE / (f"model_competent{S2.TAG}.npz" if n is None else f"model_competent{S2.TAG}_smoke.npz"), **S2.stack(out, len(rows)))
    np.savez_compressed(HERE / ("prefix_competent.npz" if n is None else "prefix_competent_smoke.npz"),
                        **{f"terr_{w}": te for w, (te, _) in enumerate(pt)}, **{f"pos_{w}": po for w, (_, po) in enumerate(pt)})


def routes(n=None, modes=("free", "ctx_real", "mem_real", "teacher"), name="routes"):
    model, cfg, R, head, pol = setup()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, plans = S2.bank(1000 if n is None else n)
    eps = np.unique(rows[:, 0]); sub = {e: i for i, e in enumerate(eps)}
    enc = S.encode(model, S2.sublabels(L, eps), cfg)
    grids = G.episode_grids(L)
    res = {}
    for mode in modes:
        o = S2.stack(rollout(model, cfg, R, head, pol, enc, sub, grids, rows, plans, mode=mode, mf=4, seed=500, gaze="recorded"), len(rows))
        res.update({f"{mode}_{k}": v for k, v in o.items() if k in ("m_grid", "m_done")})
    np.savez_compressed(HERE / (f"{name}{S2.TAG}.npz" if n is None else f"{name}{S2.TAG}_smoke.npz"), **res)


# ---------------------------------------------------------------- gates (CPU)
def counts_by_step(grid, done):
    a = np.concatenate([np.ones(done.shape[:-1] + (1,), bool), ~np.maximum.accumulate(done.astype(bool), -1)], -1)
    return {kn: (((grid == k).sum((-1, -2)) * a).sum((0, 1)) / np.maximum(a.sum((0, 1)), 1)).round(4).tolist()
            for k, kn in enumerate(G.KINDS, 1)}


def scalars(o):
    n = O.n_alive(o) - 1
    hd = o["hud"][:n + 1]; d = np.diff(hd[:, :4], axis=0)
    return [float((d[:, 0] < 0).sum()), float(np.abs(o["pos"][n] - o["pos"][0]).sum()), float((o["grid"][n] > 0).sum()),
            float((d[:, 1] < 0).sum()), float((d[:, 2] < 0).sum()), float((d[:, 3] < 0).sum()),
            float(o["reward"][:n].sum()), float(np.argmax(o["done"]) + 1 if o["done"].any() else H + 1)]


SCALARS = ("health_drops", "displacement", "mobs_at_end", "food_drops", "drink_drops", "energy_drops", "return", "death_step")


def spread(Ts, Ref, rng):
    """Randomised PIT of every reference future's scalars among the 8 futures of Ts, pooled: quartile shares (uniform =
    calibrated; heavy outer quartiles = the world too narrow)."""
    u = {k: [] for k in SCALARS}
    for ts, rs in zip(Ts, Ref):
        xs = np.array([scalars(o) for o in ts])
        for o in rs:
            y = scalars(o)
            for i, k in enumerate(SCALARS):
                u[k].append(O.pit(xs[:, i], y[i], rng))
    return {k: (np.histogram(v, bins=[0, .25, .5, .75, 1])[0] / len(v)).round(4).tolist() for k, v in u.items()}


def revisits(Ts, prefix):
    """TEST_SPEC S2.2 on each trajectory: cells entering view whose world cell's terrain was clearly seen before (the
    real prefix's reads, then the rollout's own), agreement with the last clear sighting, by gap. terr -1 = covered."""
    rr, cc = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
    bins = ((1, 16), (17, 64), (65, 10 ** 9))
    num, den = np.zeros(3), np.zeros(3)
    for ts, (pte, ppo) in zip(Ts, prefix):
        base = {}
        for t in range(len(ppo)):
            for i, j in zip(*np.nonzero(pte[t] >= 0)):
                base[(ppo[t][0] + rr[i, j], ppo[t][1] + cc[i, j])] = (t, int(pte[t][i, j]))
        T0 = len(ppo) - 1
        for o in ts:
            last, n = dict(base), O.n_alive(o)
            prev = {(o["pos"][0][0] + rr[i, j], o["pos"][0][1] + cc[i, j]) for i in range(7) for j in range(9)}
            for s in range(1, n):
                cur = set()
                for i in range(7):
                    for j in range(9):
                        w = (o["pos"][s][0] + rr[i, j], o["pos"][s][1] + cc[i, j]); cur.add(w)
                        v = int(o["terr"][s][i, j])
                        if v < 0:
                            continue
                        if w not in prev and w in last:
                            gap = T0 + s - last[w][0]
                            b = 0 if gap <= 16 else 1 if gap <= 64 else 2
                            num[b] += v == last[w][1]; den[b] += 1
                        last[w] = (T0 + s, v)
                prev = cur
    return {f"gap_{lo}_{hi if hi < 10 ** 9 else 'more'}": {"agree": float(num[k] / max(den[k], 1)), "events": int(den[k])}
            for k, (lo, hi) in enumerate(bins)}


def attach(T, z, src, sel=slice(None)):
    te = z[f"{src}_terr"]
    for w, ts in enumerate(T):
        for f, o in zip(range(te.shape[1])[sel], ts):
            o["terr"] = te[w, f]
    return T


def gates(n=None):
    rng, tag = np.random.default_rng(20261120), "" if n is None else "_smoke"
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    res = {}
    # 1. the route localisation (stage 1's bank)
    if not (HERE / f"routes{S2.TAG}{tag}.npz").exists():               # another world read without the route test
        return gates_competent(n, rng, tag, L, res)
    zr = dict(np.load(HERE / f"routes{S2.TAG}{tag}.npz"))
    oz = dict(np.load(HERE / "oracle.npz"))
    k = len(zr["free_m_grid"])
    rows1, _ = S2.bank(k)
    res["routes_counts_by_step"] = {"simulator_B": counts_by_step(oz["rd_grid"][:k, 8:], oz["lab_done"][:k, 8:]),
                                    **{m: counts_by_step(zr[f"{m}_m_grid"], zr[f"{m}_m_done"]) for m in ("free", "ctx_real", "mem_real", "teacher")}}
    alive = lambda dn: np.concatenate([np.ones(dn.shape[:-1] + (1,), bool), ~np.maximum.accumulate(dn.astype(bool), -1)], -1)
    per = lambda m: (((zr[f"{m}_m_grid"] > 0).sum((-1, -2)) * alive(zr[f"{m}_m_done"])).sum(1), alive(zr[f"{m}_m_done"]).sum(1).astype(float))
    nf, df = per("free")
    res["routes_paired_vs_free"] = {}
    for m in ("ctx_real", "mem_real", "teacher"):
        nm, dm = per(m)
        d, se = O.paired_diff(nm, dm, nf, df, rows1[:, 0], rng)
        res["routes_paired_vs_free"][m] = {"mobs_diff_by_step": np.round(d, 4)[[1, 4, 8, 16]].tolist(), "se": np.round(se, 4)[[1, 4, 8, 16]].tolist(),
                                           "alive_by_step": (dm.sum(0) / (4 * k)).round(4)[[1, 4, 8, 16]].tolist()}
    res["routes_paired_vs_free"]["free_alive_by_step"] = (df.sum(0) / (4 * k)).round(4)[[1, 4, 8, 16]].tolist()
    return gates_competent(n, rng, tag, L, res)


def gates_competent(n, rng, tag, L, res):
    # 2. competent starts
    z = dict(np.load(HERE / f"oracle_competent{tag}.npz")); z.update(np.load(HERE / f"model_competent{S2.TAG}{tag}.npz"))
    pz = np.load(HERE / f"prefix_competent{tag}.npz")
    rows, plans, meta = competent(n)
    assert len(rows) == len(z["rd_grid"]) == len(z["m_grid"]), "bank and rollouts differ"
    prefix = [(pz[f"terr_{w}"], pz[f"pos_{w}"]) for w in range(len(rows))]
    light = z["lab_light"][:, 0]
    A, B = attach(S2.trajs(z, "rd", light, slice(0, 8)), z, "rd", slice(0, 8)), attach(S2.trajs(z, "rd", light, slice(8, 16)), z, "rd", slice(8, 16))
    Mo = attach(S2.trajs(z, "m", light), z, "m")
    Bl = S2.trajs(z, "lab", light, slice(8, 16))
    a_lab, a_rd = z["lab_hud"].reshape(-1, 16), z["rd_hud"].reshape(-1, 16)
    res["reader_error_competent_simulator_frames"] = {"hud_exact": dict(zip(S2.HUD, (a_lab == a_rd).mean(0).round(4).tolist())),
        "entity_cells": {kn: {"recall": float(((z["rd_grid"] == q) & (z["lab_grid"] == q)).sum() / max(1, (z["lab_grid"] == q).sum()))}
                         for q, kn in enumerate(G.KINDS, 1)}}
    rp = np.flatnonzero(meta["kind"] == 0)
    sel = lambda Ts, idx: [Ts[i] for i in idx]
    ep_r = rows[rp, 0]
    agg = lambda Ts, f: tuple(np.array([np.sum([f(o)[i] for o in t], 0) for t in Ts]) for i in range(4))

    def pd(na, da, nb, db, ep):
        d, se = O.paired_diff(na, da, nb, db, ep, rng)
        return {"diff": np.round(d, 5).tolist(), "se": np.round(se, 5).tolist(), "rejects": bool(np.any(np.abs(d) > 3 * se + 1e-9))}
    comp = {}
    for name, X in (("null_A_vs_B", sel(A, rp)), ("model_vs_B", sel(Mo, rp))):
        p, pb = agg(X, O.per_traj), agg(sel(B, rp), O.per_traj)
        comp[name] = {"persistence": pd(p[0], p[1], pb[0], pb[1], ep_r), "count": pd(p[2], p[3], pb[2], pb[3], ep_r),
                      "level": {"persistence": (p[0].sum(0) / p[1].sum(0)).round(4).tolist(), "count": float(p[2].sum() / p[3].sum()),
                                "reference_persistence": (pb[0].sum(0) / pb[1].sum(0)).round(4).tolist(), "reference_count": float(pb[2].sum() / pb[3].sum())}}
        blk = {}
        for q, kn in enumerate(G.KINDS, 1):
            pm, pr = agg(X, lambda o: S2.per_kind(o, q)), agg(sel(B, rp), lambda o: S2.per_kind(o, q))
            cm, cb = pm[2].sum() / pm[3].sum(), pr[2].sum() / pr[3].sum()
            c = pd(pm[2], pm[3], pr[2], pr[3], ep_r)
            p4m, p4r = pm[0][:, 3].sum() / max(pm[1][:, 3].sum(), 1), pr[0][:, 3].sum() / max(pr[1][:, 3].sum(), 1)
            blk[kn] = {"count": [round(float(cm), 4), round(float(cb), 4)], "persistence4": [round(float(p4m), 4), round(float(p4r), 4)],
                       "fires": bool((cb and abs(cm / cb - 1) > 0.30 and abs(cm - cb) > 3 * c["se"]) or (kn != "arrows" and p4r - p4m > 0.10))}
        comp[name]["S3_bookkeeping_blocker"] = blk
    res["competent_random_plans"] = comp
    cons = lambda Ts: tuple(np.array([np.sum([S2.consistency(o)[i] for o in t], 0) for t in Ts]) for i in range(2))
    (cmn, cmd), (cbn, cbd) = cons(sel(Mo, rp)), cons(sel(B, rp))
    d, se = O.paired_diff(cmn, cmd, cbn, cbd, ep_r, rng)
    res["competent_consistency"] = {kk: {"model": round(float(cmn[:, i].sum() / max(cmd[:, i].sum(), 1)), 5),
                                         "reference": round(float(cbn[:, i].sum() / max(cbd[:, i].sum(), 1)), 5), "rejects": bool(abs(d[i]) > 3 * se[i] + 1e-9)}
                                    for i, kk in enumerate(S2.CONS) if cmn[:, i].sum() + cbn[:, i].sum() > 0}
    FE = lambda Ts, idx: [[S2.first_events(o, False) for o in Ts[w]] for w in idx]
    fm, fb = FE(Mo, rp), FE(B, rp)
    l2 = {}
    for fam in ("damage", "death", "food_drop", "drink_drop", "energy_drop", "unlock"):
        cur = lambda F_: np.array([[sum(f[fam] <= kk for f in fw) for kk in range(1, H + 1)] for fw in F_], float)
        dm, db = cur(fm), cur(fb)
        dd, ss = O.paired_diff(dm, np.full_like(dm, MF), db, np.full_like(db, 8.), ep_r, rng)
        l2[fam] = {"model_by_k": (dm.sum(0) / (MF * len(rp))).round(4)[[0, 3, 7, 15]].tolist(),
                   "reference_by_k": (db.sum(0) / (8 * len(rp))).round(4)[[0, 3, 7, 15]].tolist(),
                   "rejects_at_k": [int(q + 1) for q in np.flatnonzero(np.abs(dd) > 3 * ss + 1e-9)]}
    res["competent_time_to_event"] = l2
    # 3. holds: the action's effect, DO minus NOOP, world against simulator
    hp = {}
    for b, (q, _) in COLLECT.items():
        do = np.flatnonzero((meta["kind"] == 1) & (meta["material"] == b))
        if not len(do):
            continue
        no = do + 1
        assert (meta["pair"][do] == meta["pair"][no]).all() and (rows[do, 3] == rows[no, 3]).all()
        eff = {}
        for kk in (1, 4, 16):
            gain = lambda Ts, w: np.array([float(o["hud"][min(kk, O.n_alive(o) - 1), q] - o["hud"][0, q]) for o in Ts[w]])
            em = np.array([gain(Mo, a).mean() - gain(Mo, c).mean() for a, c in zip(do, no)])
            es = np.array([gain(B, a).mean() - gain(B, c).mean() for a, c in zip(do, no)])
            el = np.array([gain(Bl, a).mean() - gain(Bl, c).mean() for a, c in zip(do, no)])   # the reader's attenuation
            dd, ss = O.paired_diff(em, np.ones(len(em)), es, np.ones(len(es)), rows[do, 0], rng)
            eff[f"step_{kk}"] = {"world": round(float(em.mean()), 4), "simulator": round(float(es.mean()), 4),
                                 "simulator_labels": round(float(el.mean()), 4), "diff_se": round(float(ss), 4),
                                 "rejects": bool(abs(dd) > 3 * ss + 1e-9)}
        hp[MATERIAL[b]] = {"pairs": int(len(do)), **eff}
    res["holds_effect_of_DO"] = hp
    # 4. spread and revisits on both banks
    res["spread_competent"] = {"model_vs_B": spread(sel(Mo, rp), sel(B, rp), rng), "null_A_vs_B": spread(sel(A, rp), sel(B, rp), rng)}
    res["revisits_competent"] = {"model": revisits(sel(Mo, rp), [prefix[i] for i in rp]), "simulator_B": revisits(sel(B, rp), [prefix[i] for i in rp])}
    z1 = dict(np.load(HERE / "oracle.npz")); z1.update(np.load(HERE / f"model{S2.TAG}.npz"))
    l1 = z1["lab_light"][:, 0]
    sp = lambda src, s_: S2.trajs(z1, src, l1, s_)
    res["spread_stage1_bank"] = {"model_vs_B": spread(sp("m", slice(None)), sp("rd", slice(8, 16)), rng),
                                 "null_A_vs_B": spread(sp("rd", slice(0, 8)), sp("rd", slice(8, 16)), rng)}
    (HERE / f"gates_1b{S2.TAG}{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("routes_counts_by_step", "holds_effect_of_DO") if k in res}, indent=1), flush=True)


def split_gates(name="routes_split", modes=("record_real", "mamba_real")):
    """Mamba against the record (11-19): the two split arms paired against stage 1b's free arm (the same starts, seeds and
    fixation schedule: the first generated frame must agree exactly), beside mem_real and teacher; name / modes: the
    readers split (routes_readers: dread_real, mread_real), beside record_real as well."""
    rng = np.random.default_rng(20261121)
    zr, zs = dict(np.load(HERE / "routes.npz")), dict(np.load(HERE / f"{name}.npz"))
    k = len(zr["free_m_grid"])
    rows1, _ = S2.bank(k)
    for m in modes:
        assert np.array_equal(zs[f"{m}_m_grid"][:, :, :2], zr["free_m_grid"][:, :, :2]), "the first frame differs from free running"
    z = {**zr, **dict(np.load(HERE / "routes_split.npz")), **zs}
    modes = tuple(dict.fromkeys((*modes, "record_real", "mamba_real")))
    alive = lambda dn: np.concatenate([np.ones(dn.shape[:-1] + (1,), bool), ~np.maximum.accumulate(dn.astype(bool), -1)], -1)
    per = lambda m, kind=None: ((((z[f"{m}_m_grid"] > 0) if kind is None else (z[f"{m}_m_grid"] == kind)).sum((-1, -2))
                                 * alive(z[f"{m}_m_done"])).sum(1), alive(z[f"{m}_m_done"]).sum(1).astype(float))
    res = {}
    for kind, kn in ((None, "all"), *enumerate(G.KINDS, 1)):
        nf, df = per("free", kind)
        res[kn] = {"free": (nf.sum(0) / df.sum(0)).round(4)[[1, 4, 8, 16]].tolist()}
        for m in (*modes, "mem_real", "teacher"):
            nm, dm = per(m, kind)
            d, se = O.paired_diff(nm, dm, nf, df, rows1[:, 0], rng)
            res[kn][m] = {"level": (nm.sum(0) / dm.sum(0)).round(4)[[1, 4, 8, 16]].tolist(),
                          "diff": np.round(d, 4)[[1, 4, 8, 16]].tolist(), "se": np.round(se, 4)[[1, 4, 8, 16]].tolist()}
    res["alive_at_16"] = {m: float(alive(z[f"{m}_m_done"])[..., 16].mean()) for m in ("free", *modes, "mem_real", "teacher")}
    (HERE / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["all"], indent=1), json.dumps(res["alive_at_16"]), flush=True)


if __name__ == "__main__":
    stage, n = sys.argv[1], (int(sys.argv[2]) if len(sys.argv) > 2 else None)
    {"bank": build_bank, "oracle": lambda: oracle(n), "model": lambda: model_competent(n), "routes": lambda: routes(n),
     "routes_split": lambda: routes(n, ("record_real", "mamba_real"), "routes_split"), "split_gates": split_gates,
     "routes_readers": lambda: routes(n, ("dread_real", "mread_real"), "routes_readers"),
     "readers_gates": lambda: split_gates("routes_readers", ("dread_real", "mread_real")),
     "gates": lambda: gates(n)}[stage]()
