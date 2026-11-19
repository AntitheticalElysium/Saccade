"""S2 on P1.5, stage 1 (PLAN.md): every trajectory, simulated or generated, read through one set of frozen readers.

    python -m artifacts.experiments.20261119_s2.s2 readers|oracle|model|gates [N_STARTS]   (N_STARTS: smoke runs only)

readers  HUD (16 quantities capped at 9, asleep) from the HUD rows' tokens + lighting; camera shift (5 classes) from two
         frames' map tokens; the same HUD targets from the corrected belief (N5). S0 probe protocol on train-400.
oracle   the simulator's 16 futures per bank start (oracle_l1.py's keys: the same futures), rendered and read, beside
         their labels; the recorded future read the same way.
model    the world's 8 futures per start: real prefix (belief, record, up to 8 context frames), 16 free-running steps
         under the precommitted actions, the checkpoint's chooser choosing each fixation between the cascade's stages,
         sampled reward / termination, absorbing per branch; read through the same readers; the belief's HUD read.
gates    instrument checks, L1, TEST_SPEC S3's bookkeeping blocker, L2b curves, internal consistency, N5 (CPU).
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEV = S.DEV
P15 = ROOT / "artifacts/experiments/20261116_p15/run/ckpt_10.pt"
CK = Path(os.environ.get("S2_CKPT", P15)).resolve()              # another world (11-19: the switch arm); P1.5 by default
TAG = os.environ.get("S2_TAG", "")                               # its outputs: model{TAG}.npz, gates{TAG}.json
BANK = ROOT / "artifacts/experiments/20261114_s2_harness/dev_bank_h16.npz"
AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
EV = importlib.import_module("artifacts.experiments.20261116_p15.evaluation")
H, MF, NB = 16, 8, 16                      # horizon, model futures per start, starts per GPU batch
SHIFT = {(0, 0): 0, **{v: k for k, v in G.MOVE.items()}}                 # camera shift (row, col) -> class
DELTA = np.array([(0, 0)] + [G.MOVE[k] for k in (1, 2, 3, 4)])
HUD = ("health", "food", "drink", "energy", "wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe",
       "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")


def load():
    model, cfg = AI.load(CK)
    if os.environ.get("S2_STEPS"):                                  # the sampler's K (20261120_switching_dissection's sweep)
        cfg["edm"]["sampler"]["steps"] = int(os.environ["S2_STEPS"])
    return model.requires_grad_(False), cfg


def bank(n=None):
    b = np.load(BANK)
    pick = b["sample"] if n is None else b["sample"][:n]
    return b["rows"][pick], b["plans"][pick].astype(np.int64)


def sublabels(L, eps):
    so, ao = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    return {k: v[eps] if k.startswith("ep_") else np.concatenate([v[(so if k.startswith("s_") else ao)[e]:
            (so if k.startswith("s_") else ao)[e + 1]] for e in eps]) for k, v in L.items()}


def hud_x(p, v):
    return torch.cat([p[..., 63:, :].flatten(-2), v], -1)


def shift_x(p0, p1):
    return torch.cat([p0[..., :63, :].flatten(-2), p1[..., :63, :].flatten(-2)], -1)


def tables(enc, L):
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    p, v, b = (torch.cat([r[k] for r in enc]) for k in ("p", "v", "b"))
    i0 = np.concatenate([np.arange(so[e], so[e + 1] - 1) for e in range(len(L["ep_T"]))])
    pos = L["s_pos"].astype(int)
    ep = np.repeat(np.arange(len(L["ep_T"])), L["ep_T"] + 1)
    return {"hx": hud_x(p, v), "bx": b, "sx": shift_x(p[i0], p[i0 + 1]), "ep": ep, "ep_t": ep[i0],
            "hud": torch.as_tensor(np.minimum(np.concatenate([L["s_body"], L["s_inv"]], 1), 9).astype(np.int64)),
            "sleep": torch.as_tensor(L["s_sleep"].astype(np.int64)),
            "shift": torch.as_tensor(np.array([SHIFT[tuple(d)] for d in pos[i0 + 1] - pos[i0]], np.int64))}


def hud_loss(o, y):
    return F.cross_entropy(o["hud"].reshape(-1, 16, 10).transpose(1, 2), y["hud"]) + F.cross_entropy(o["sleep"], y["sleep"])


def fit(X, Y, ep, heads, loss_fn):
    held = torch.as_tensor(ep % 10 == 9)
    Xs = (X[~held].to(DEV), X[held].to(DEV))
    Ys = ({k: y[~held].to(DEV) for k, y in Y.items()}, {k: y[held].to(DEV) for k, y in Y.items()})
    return S.fit(lambda i, val: Xs[val][i], lambda o, i, val: loss_fn(o, {k: y[i] for k, y in Ys[val].items()}),
                 len(Xs[0]), len(Xs[1]), heads, 0)


def readers():
    model, cfg = load()
    Ltr, Ldv = dict(np.load(S.LABELS / "s0_train400_labels.npz")), dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, _ = bank()
    Lb = sublabels(Ldv, np.unique(rows[:, 0]))
    tr, dv = tables(S.encode(model, Ltr, cfg), Ltr), tables(S.encode(model, Lb, cfg), Lb)
    heads = {"hud": 160, "sleep": 2}
    hud, lh = fit(tr["hx"], {"hud": tr["hud"], "sleep": tr["sleep"]}, tr["ep"], heads, hud_loss)
    bel, lb = fit(tr["bx"], {"hud": tr["hud"], "sleep": tr["sleep"]}, tr["ep"], heads, hud_loss)
    sh, ls = fit(tr["sx"], {"shift": tr["shift"]}, tr["ep_t"], {"shift": 5}, lambda o, y: F.cross_entropy(o["shift"], y["shift"]))
    res = {"fit_logs": {"hud": lh, "belief": lb, "shift": ls}, "dev_episodes": len(Lb["ep_T"])}
    with torch.no_grad():
        run = lambda m, x: {k: torch.cat([m(c.to(DEV))[k].cpu() for c in x.split(65536)]) for k in m.heads}
        for name, m, x in (("hud", hud, dv["hx"]), ("belief", bel, dv["bx"])):
            o = run(m, x)
            ok = o["hud"].reshape(-1, 16, 10).argmax(-1) == dv["hud"]
            res[f"{name}_exact"] = dict(zip(HUD, ok.float().mean(0).round(decimals=4).tolist()))
            res[f"{name}_asleep_accuracy"] = float((o["sleep"].argmax(-1) == dv["sleep"]).float().mean())
        o = run(sh, dv["sx"])["shift"].argmax(-1)
        res["shift_accuracy"] = float((o == dv["shift"]).float().mean())
        res["shift_recall_by_class"] = [float((o[dv["shift"] == c] == c).float().mean()) for c in range(5)]
    torch.save({"hud": hud.state_dict(), "belief": bel.state_dict(), "shift": sh.state_dict(),
                "checkpoint_sha256": TR.file_sha256(CK), "script_sha256": TR.file_sha256(__file__)}, HERE / "readers.pt")
    (HERE / "readers.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "fit_logs"}, indent=1), flush=True)


def load_readers(model, cfg):
    saved = torch.load(HERE / "readers.pt", weights_only=False, map_location="cpu")
    assert saved["checkpoint_sha256"] == TR.file_sha256(P15)       # the readers read P1.5's frozen encoder's tokens:
    if CK != P15.resolve():                                        # another world must share that encoder exactly
        a, b = (torch.load(c, weights_only=False, map_location="cpu")["model"] for c in (P15, CK))
        assert all(torch.equal(a[k], b[k]) for k in a if k.startswith("perception.")), "the encoder differs from P1.5's"
    out = {"kind": S.load_reader(P15.parent / "s1/results.reader.pt", P15, DEV)}
    for k, heads in (("hud", {"hud": 160, "sleep": 2}), ("belief", {"hud": 160, "sleep": 2}), ("shift", {"shift": 5})):
        m = S.MLP(saved[k]["body.0.weight"].shape[1], heads); m.load_state_dict(saved[k])
        out[k] = m.to(DEV).eval().requires_grad_(False)
    return out


@torch.no_grad()
def read(R, p, v, pos0):
    """p [N, S, 81, 16], v [N, S, 8] for S consecutive states, pos0 [N, 2] the first state's true position -> kind grid
    [N, S, 7, 9], HUD [N, S, 16], asleep [N, S], positions [N, S, 2] (pos0 plus the read camera shifts)."""
    N, Sn = p.shape[:2]
    x = torch.cat([p[:, :, :63].reshape(-1, 16), v[:, :, None].expand(-1, -1, 63, -1).reshape(-1, 8)], -1).half()
    kind = torch.cat([R["kind"](c)["kind"].argmax(1) for c in x.split(262144)]).reshape(N, Sn, 63)
    kind[..., ~torch.as_tensor(KEEP, device=kind.device)] = 0
    o = R["hud"](hud_x(p, v).reshape(-1, 296).half())
    hud, sleep = o["hud"].reshape(N, Sn, 16, 10).argmax(-1), o["sleep"].argmax(-1).reshape(N, Sn)
    sh = R["shift"](shift_x(p[:, :-1], p[:, 1:]).reshape(-1, 2016).half())["shift"].argmax(-1).reshape(N, Sn - 1).cpu().numpy()
    pos = pos0[:, None] + np.concatenate([np.zeros((N, 1, 2), int), np.cumsum(DELTA[sh], 1)], 1)
    return kind.reshape(N, Sn, 7, 9).cpu().numpy().astype(np.int8), hud.cpu().numpy().astype(np.int8), \
        sleep.cpu().numpy().astype(bool), pos.astype(np.int16)


def sim_sampler():
    """O.sampler (the same env, step and keys) with inventory, sleep and the rendered frame of every state."""
    import jax, jax.numpy as jnp
    env, params, _, _, frame = O.RP.env_and_render()
    inv = lambda s: jnp.stack([getattr(s.inventory, f) for f in s.inventory.__dataclass_fields__])
    rec = lambda s: {"pos": s.player_position, "light": s.light_level, "sleep": s.is_sleeping, "inv": inv(s),
                     "body": jnp.stack([s.player_health, s.player_food, s.player_drink, s.player_energy]),
                     **{f"{m}_p": getattr(s, m).position for m in G.KINDS}, **{f"{m}_m": getattr(s, m).mask for m in G.KINDS}}

    def one(state, actions, keys):
        def body(s, xs):
            k, a = xs
            _, nxt, r, d, _ = env.step(k, s, a, params)
            return nxt, (rec(nxt), frame(nxt), r, d)
        _, (seq, fr, r, d) = jax.lax.scan(body, state, (keys, actions))
        cat = lambda f, q: jnp.concatenate([f[None], q])
        return jax.tree.map(cat, rec(state), seq), cat(frame(state), fr), r, d
    return jax.jit(jax.vmap(one, in_axes=(None, None, 0)))


def store(out, src, i, grid, hud, sleep, pos, reward, done, light):
    for k, x in (("grid", grid), ("hud", hud), ("sleep", sleep), ("pos", pos), ("reward", reward), ("done", done), ("light", light)):
        if x is not None:
            out.setdefault(f"{src}_{k}", {})[i] = x


def stack(out, n):
    return {k: np.stack([d[i] for i in range(n)]) for k, d in out.items()}


def oracle(n=None):
    import jax, jax.numpy as jnp
    model, cfg = load()
    R = load_readers(model, cfg)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, plans = bank(n)
    eps = np.unique(rows[:, 0]); sub = {e: i for i, e in enumerate(eps)}
    enc = S.encode(model, sublabels(L, eps), cfg)
    grids, so = G.episode_grids(L), np.r_[0, np.cumsum(L["ep_T"] + 1)]
    samp, out, pend = sim_sampler(), {}, []

    def flush():
        fr = torch.as_tensor(np.stack([f for _, f, _ in pend]), device=DEV)          # [b, 16, 17, 63, 63, 3]
        with TR.autocast(DEV, cfg):
            p, v = C.coarse(model.perception, fr.flatten(0, 1))
        r = read(R, p.float(), v.float(), np.concatenate([np.repeat(pp[None], 16, 0) for _, _, pp in pend]))
        for j, (w, _, _) in enumerate(pend):
            store(out, "rd", w, *(x[16 * j:16 * (j + 1)] for x in r), None, None, None)
        pend.clear()
    t_start = time.time()
    for w, (e, sh, sl, t0, T, _) in enumerate(rows):
        state = O.RP.advance_to(int(sh), int(sl), int(t0))
        keys = jax.random.split(jax.random.PRNGKey(2 * 10 ** 6 + w), O.M * H).reshape(O.M, H, 2)
        Rr, fr, rr, dd = jax.tree.map(np.asarray, samp(state, jnp.asarray(plans[w].astype(np.int32)), keys))
        lab = [O.to_obs(jax.tree.map(lambda q: q[i], Rr)) for i in range(O.M)]
        store(out, "lab", w, np.stack([g for g, _, _ in lab]),
              np.minimum(np.concatenate([Rr["body"], Rr["inv"]], -1), 9).astype(np.int8), Rr["sleep"].astype(bool),
              np.stack([p for _, _, p in lab]).astype(np.int16), rr.astype(np.float32), dd.astype(bool), Rr["light"].astype(np.float32))
        pend.append((w, fr, lab[0][2][0]))
        en, ep, m = enc[sub[e]], grids[e], min(H, int(T) - int(t0))                 # the recorded future, absorbing
        pad = lambda x, k: np.concatenate([x, np.repeat(x[-1:], k - len(x), 0)]) if len(x) < k else x
        g_, h_, s_, p_ = read(R, en["p"][t0:t0 + m + 1].to(DEV).float()[None], en["v"][t0:t0 + m + 1].to(DEV).float()[None], ep["pos"][t0][None])
        s0 = so[e] + t0
        store(out, "rec_rd", w, pad(g_[0], H + 1), pad(h_[0], H + 1), pad(s_[0], H + 1), pad(p_[0], H + 1), None, None, None)
        store(out, "rec_lab", w, pad(ep["grid"][t0:t0 + m + 1], H + 1),
              pad(np.minimum(np.concatenate([L["s_body"], L["s_inv"]], 1)[s0:s0 + m + 1], 9).astype(np.int8), H + 1),
              pad(L["s_sleep"][s0:s0 + m + 1].astype(bool), H + 1), pad(ep["pos"][t0:t0 + m + 1], H + 1).astype(np.int16),
              np.r_[ep["reward"][t0:t0 + m], np.zeros(H - m)].astype(np.float32), np.r_[ep["done"][t0:t0 + m], np.ones(H - m, bool)],
              pad(L["s_light"][s0:s0 + m + 1].astype(np.float32), H + 1))
        if len(pend) == 8:
            flush()
        if (w + 1) % 200 == 0:
            print(f"start {w + 1}/{len(rows)} {time.time() - t_start:.0f}s", flush=True)
    if pend:
        flush()
    np.savez_compressed(HERE / ("oracle.npz" if n is None else "oracle_smoke.npz"), **stack(out, len(rows)))


def model_futures(n=None):
    model, cfg = load(); EV.guard()
    R = load_readers(model, cfg)
    state0 = torch.load(CK, weights_only=False, map_location="cpu")["fixation_controller"]
    pol = dict(cfg["fixation"]["chooser"])
    head = C.Head(pol["history"]).to(DEV); head.load_state_dict(state0["head"]); head.eval().requires_grad_(False)
    win, centres, Hh = C.WIN.to(DEV), C.CENTRES.to(DEV), pol["history"]
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, plans = bank(n)
    eps = np.unique(rows[:, 0]); sub = {e: i for i, e in enumerate(eps)}
    enc = S.encode(model, sublabels(L, eps), cfg)
    grids = G.episode_grids(L)
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    P = cfg["data"]["context_prefix"]
    out, kc = {}, np.minimum(rows[:, 3], P - 1) + 1                              # real context frames committed
    batches = [b for k in np.unique(kc) for b in np.array_split(np.flatnonzero(kc == k), max(1, -(-int((kc == k).sum()) // NB)))]
    t_start = time.time()
    for bi, ws in enumerate(batches):
        rr = [(sub[int(rows[w, 0])], int(rows[w, 3])) for w in ws]; k_ctx = int(kc[ws[0]])
        Bn = len(ws) * MF
        rep = lambda x: x.repeat_interleave(MF, 0)
        at = lambda key, dt: torch.stack([enc[i][key][max(t + dt, 0)] for i, t in rr]).to(DEV)
        cs, bps, rl = [], [], []
        for (i, t), w in zip(rr, ws):
            en = enc[i]
            assert int(en["action"][t]) == int(plans[w, 0])
            with TR.autocast(DEV, cfg):
                pp = M.Percept(*(en[k][:t + 1].to(DEV).float()[None] for k in ("p", "g", "v")), en["fix"][:t + 1].to(DEV)[None])
                reads, _, last = S.history(model, pp.p)
                y, c = model.belief(model.belief.tokens(S.embed(model, pp, reads), en["reward"][:t + 1].to(DEV)[None],
                                    en["terminated"][:t + 1].to(DEV)[None], en["action"][:t + 1].to(DEV)[None]), model.belief.initial(1, DEV))
            err = float((y[0, -1].float() - en["bp"][t + 1].to(DEV).float()).square().mean().sqrt())
            assert err < 0.1, ("prefix belief differs from encode's", err)               # a gross misalignment guard:
            out.setdefault("m_prefix_error", {})[int(w)] = np.float32(err)              # bf16 floor, prefix_check.json
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
        # the chooser's own 8-frame history (chooser.inputs), real up to the start
        ok = rep(torch.tensor([[t + dt >= 0 for dt in range(1 - Hh, 1)] for _, t in rr], device=DEV))
        hist = {k: rep(torch.stack([torch.stack([enc[i][k][max(t + dt, 0)] for dt in range(1 - Hh, 1)]) for i, t in rr]).to(DEV))
                for k in ("p", "v", "prev_action", "reward", "terminated")}
        hist = {k: x.float() if k != "prev_action" else x for k, x in hist.items()}
        plan = rep(torch.as_tensor(plans[ws], device=DEV))
        fix = rep(at("fix", 0)); alive = torch.ones(Bn, dtype=torch.bool, device=DEV)
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
                parts, fx = M.sample(model.denoiser, dims, pa[:, None], mem, fix[:, None], cache, cfg, g, choose=choose)
                parts = tuple(x[:, 0].float() for x in parts); fix = fx[:, 0]
                reads, _, next_rs = S.history(model, parts[0], state=rs)
                emb = S.embed(model, M.Percept(*parts, fix), reads)
                h = model.outcome.features(bp, emb).float()
                r = model.outcome.sample_reward(h, g)
                done = torch.rand(Bn, device=DEV, generator=g) > torch.sigmoid(model.outcome.cont_logit(h, r).float())
                nxt = plan[:, j] if j < H else torch.zeros_like(pa)
                yy, carry = model.belief(model.belief.tokens(emb[:, None], r[:, None], done[:, None], nxt[:, None]), carry)
                cache = M.context_pass(model.denoiser, tuple(a[:, None] for a in parts), pa[:, None], mem, fix[:, None], cache, cfg, g)
                ob = R["belief"](yy[:, 0].float())
            alive = alive & ~done
            rec["reward"].append(r.float()); rec["done"].append(~alive); rec["fix"].append(fix)
            rec["bhud"].append(ob["hud"].reshape(Bn, 16, 10).argmax(-1)); rec["bsleep"].append(ob["sleep"].argmax(-1).bool())
            ps.append(parts[0]); vs.append(parts[2])
            bp, rs = yy[:, 1].float(), next_rs
            for k, x in (("p", parts[0]), ("v", parts[2]), ("prev_action", pa), ("reward", r.float()), ("terminated", done.float())):
                hist[k] = torch.cat([hist[k][:, 1:], x[:, None].to(hist[k].dtype)], 1)
            ok = torch.cat([ok[:, 1:], torch.ones_like(ok[:, :1])], 1)
        pos0 = np.repeat(np.stack([grids[int(rows[w, 0])]["pos"][int(rows[w, 3])] for w in ws]), MF, 0)
        gr, hu, sl, po = read(R, torch.stack(ps, 1), torch.stack(vs, 1), pos0)
        z = {k: torch.stack(x, 1).cpu().numpy() for k, x in rec.items()}
        for a, w in enumerate(ws):
            s = slice(MF * a, MF * (a + 1))
            store(out, "m", int(w), gr[s], hu[s], sl[s], po[s], z["reward"][s], z["done"][s], None)
            for k in ("fix", "bhud", "bsleep"):
                out.setdefault(f"m_{k}", {})[int(w)] = z[k][s]
        print(f"batch {bi + 1}/{len(batches)} starts {sum(len(b) for b in batches[:bi + 1])}/{len(rows)} {time.time() - t_start:.0f}s", flush=True)
    # N5's real control: the encoded belief of the recorded future against its read frames
    for w, (e, _, _, t0, T, _) in enumerate(rows):
        en, m = enc[sub[int(e)]], min(H, int(T) - int(t0))
        ob = R["belief"](en["b"][t0 + 1:t0 + m + 1].to(DEV).float())
        bh = np.zeros((H, 16), np.int8); bs = np.zeros(H, bool); valid = np.zeros(H, bool)
        bh[:m] = ob["hud"].reshape(m, 16, 10).argmax(-1).cpu().numpy(); bs[:m] = ob["sleep"].argmax(-1).cpu().numpy(); valid[:m] = True
        for k, x in (("bhud", bh), ("bsleep", bs), ("valid", valid)):
            out.setdefault(f"rec_{k}", {})[w] = x
    np.savez_compressed(HERE / (f"model{TAG}.npz" if n is None else f"model{TAG}_smoke.npz"), **stack(out, len(rows)))


# ---------------------------------------------------------------- gates (CPU)
def trajs(z, src, light, sel=slice(None)):
    """[start][future] observable trajectories in oracle.traj's format (body = the first four HUD quantities). A read
    source without outcomes (the simulator's or the recording's frames through the readers) takes its labels' reward
    and termination: the same futures."""
    out = {k: z[f"{src}_{k}"] if f"{src}_{k}" in z else z[f"{src.replace('rd', 'lab')}_{k}"]
           for k in ("grid", "hud", "pos", "reward", "done", "sleep")}
    if out["grid"].ndim == 4:                                             # one future per start
        out = {k: x[:, None] for k, x in out.items()}
    return [[{"grid": out["grid"][w, f].astype(np.int8), "body": out["hud"][w, f, :, :4].astype(int),
              "hud": out["hud"][w, f].astype(int), "pos": out["pos"][w, f].astype(int),
              "reward": out["reward"][w, f].astype(float), "done": np.maximum.accumulate(out["done"][w, f].astype(bool)),
              "sleep": out["sleep"][w, f], "light": light[w]} for f in range(out["grid"].shape[1])[sel]]
            for w in range(len(out["grid"]))]


def per_kind(o, k):
    q = dict(o); q["grid"] = np.where(o["grid"] == k, k, 0).astype(np.int8)
    return O.per_traj(q)


def first_events(o, asleep0):
    """First step (1..16, 17 = never) of each L2b family in one trajectory, up to its termination."""
    n = O.n_alive(o) - 1                                                  # transitions before the absorbing state
    hd = o["hud"]; d = np.diff(hd[:n + 1, :4], axis=0) if n > 0 else np.zeros((0, 4))
    first = lambda m: int(np.argmax(m)) + 1 if m.any() else H + 1
    ul = np.round(o["reward"][:n] - 0.1 * d[:, 0]) == 1 if n > 0 else np.zeros(0, bool)
    ev = {"damage": first(d[:, 0] < 0), "food_drop": first(d[:, 1] < 0), "drink_drop": first(d[:, 2] < 0),
          "energy_drop": first(d[:, 3] < 0), "unlock": first(ul),
          "death": int(np.argmax(o["done"])) + 1 if o["done"].any() else H + 1}
    if asleep0:
        ev["wake"] = first(~o["sleep"][1:n + 1])
    return ev


def consistency(o):
    """Per trajectory (numerators, denominators): health read 0 while alive, health read > 0 at termination, reward
    inconsistent with the read health change, per inventory item increments / decrements / jumps >= 2."""
    n = O.n_alive(o) - 1
    hd = o["hud"][:n + 1]; d = np.diff(hd, axis=0)
    alive_states = n if o["done"].any() else n + 1
    num = [float((hd[1:alive_states, 0] == 0).sum()), float(o["done"].any() and hd[n, 0] > 0),
           float((~np.isin(np.round(o["reward"][:n] - 0.1 * d[:, 0]), (0, 1))).sum())]
    den = [float(max(alive_states - 1, 0)), float(o["done"].any()), float(n)]
    for i in range(4, 16):
        num += [float((d[:, i] == 1).sum()), float((d[:, i] < 0).sum()), float((d[:, i] >= 2).sum())]
        den += [float(n)] * 3
    return np.array(num), np.array(den)


CONS = ["health0_alive", "terminated_with_health", "reward_inconsistent"] + \
       [f"{q}_{e}" for q in HUD[4:] for e in ("inc", "dec", "jump")]


def gates(n=None):
    rng, tag = np.random.default_rng(20261119), "" if n is None else "_smoke"
    z = dict(np.load(HERE / f"oracle{tag}.npz")); z.update(np.load(HERE / f"model{TAG}{tag}.npz"))
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    rows, plans = bank(len(z["rd_grid"]))
    ep_of, light = rows[:, 0], z["lab_light"][:, 0]
    A, B = trajs(z, "rd", light, slice(0, 8)), trajs(z, "rd", light, slice(8, 16))
    Al, Bl = trajs(z, "lab", light, slice(0, 8)), trajs(z, "lab", light, slice(8, 16))
    Mo, Rec, RecL = trajs(z, "m", light), trajs(z, "rec_rd", light), trajs(z, "rec_lab", light)
    grids, so = G.episode_grids(L), np.r_[0, np.cumsum(L["ep_T"] + 1)]
    since0 = []
    for e, _, _, t0, _, _ in rows:
        drops = np.flatnonzero(np.diff(grids[e]["body"][:t0 + 1, 0]) < 0)
        since0.append(int(min(t0 - drops[-1], 10)) if len(drops) else 10)
    vec = lambda o: G.summary_vector(o["grid"], o["body"], o["reward"], o["done"], o["pos"])
    scale = np.stack([vec(t[0]) for t in Rec]).std(0); scale[scale == 0] = 1.0
    agg = lambda Ts, f: tuple(np.array([np.sum([f(o)[i] for o in t], 0) for t in Ts]) for i in range(4))
    term = lambda Ts: (np.array([sum(o["done"].any() for o in t) for t in Ts], float),
                       np.array([sum(float(np.argmax(o["done"]) + 1) if o["done"].any() else H + 1. for o in t) for t in Ts]),
                       np.array([len(t) for t in Ts], float))

    def events(Ts):
        parts = [G.traj_pairs(o["grid"][:n], o["body"][:n], o["pos"][:n], plans[w], o["light"][:n], since0[w], ep_of[w])
                 for w, ts in enumerate(Ts) for o in ts for n in [O.n_alive(o)]]
        return G.all_estimates({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    ref_ev = {"B": events(B)}

    def pd(na, da, nb, db):
        d, se = O.paired_diff(na, da, nb, db, ep_of, rng)
        return {"diff": np.round(d, 5).tolist(), "se": np.round(se, 5).tolist(), "rejects": bool(np.any(np.abs(d) > 3 * se + 1e-9))}

    def l1(Ts, Ref=B, ref="B", Y=Rec):
        p, t = agg(Ts, O.per_traj), term(Ts)
        pb, tb = agg(Ref, O.per_traj), term(Ref)
        out = {"persistence": pd(p[0], p[1], pb[0], pb[1]), "count": pd(p[2], p[3], pb[2], pb[3]),
               "terminated": pd(t[0], t[2], tb[0], tb[2]), "death_step": pd(t[1], t[2], tb[1], tb[2]),
               "level": {"persistence": (p[0].sum(0) / p[1].sum(0)).round(4).tolist(), "count": float(p[2].sum() / p[3].sum()),
                         "terminated": float(t[0].sum() / t[2].sum()),
                         "reference_persistence": (pb[0].sum(0) / pb[1].sum(0)).round(4).tolist(),
                         "reference_count": float(pb[2].sum() / pb[3].sum()), "reference_terminated": float(tb[0].sum() / tb[2].sum())}}
        if ref not in ref_ev:
            ref_ev[ref] = events(Ref)
        ev = events(Ts)
        rej, unm = G.compare(ref_ev[ref], ev)
        out["event_gates_rejecting"] = {k: [round(ref_ev[ref][k]["est"], 4), round(ev[k]["est"], 4)] for k in rej}
        out["event_gates_unmeasured"] = unm
        es = lambda X: np.array([G.energy_score(np.stack([vec(o) / scale for o in t]), vec(y[0]) / scale) for t, y in zip(X, Y)])
        out["energy_score"] = {"model": float(es(Ts).mean()) if len(Ts[0]) > 1 else None, "reference": float(es(Ref).mean())}
        out["rejected_by"] = [k for k in ("persistence", "count", "terminated", "death_step") if out[k]["rejects"]] + (["events"] if rej else [])
        return out

    res = {"starts": len(rows), "episodes": int(len(np.unique(ep_of)))}
    # 1. instrument: the reader on simulator frames, the null, the sanity check, observable-level power
    a_lab, a_rd = z["lab_hud"].reshape(-1, 16), z["rd_hud"].reshape(-1, 16)
    res["reader_error_on_simulator_frames"] = {
        "hud_exact": dict(zip(HUD, (a_lab == a_rd).mean(0).round(4).tolist())),
        "asleep_accuracy": float((z["lab_sleep"] == z["rd_sleep"]).mean()),
        "entity_cells": {kn: {"recall": float(((z["rd_grid"] == k) & (z["lab_grid"] == k)).sum() / max(1, (z["lab_grid"] == k).sum())),
                              "precision": float(((z["rd_grid"] == k) & (z["lab_grid"] == k)).sum() / max(1, (z["rd_grid"] == k).sum()))}
                         for k, kn in enumerate(G.KINDS, 1)},
        "position_exact_at_16": float((z["rd_pos"][:, :, -1] == z["lab_pos"][:, :, -1]).all(-1).mean())}
    res["label_level_null"] = l1(Al, Bl, "Bl", RecL)                       # reproduces oracle_l1.json's null
    res["null_A_vs_B_through_reader"] = l1(A)
    res["sanity_recorded_vs_B_through_reader"] = l1(Rec)
    res["power_through_reader"] = {}
    for c in ("fake_consequences", "immortal", "sudden_death"):
        X = [[OL.extra(c, o, rng) for o in t] for t in A]
        res["power_through_reader"][c] = l1(X)["rejected_by"]
    # 2. L1 and the bookkeeping blocker
    res["L1_model_vs_B"] = l1(Mo)
    res["L1_model_vs_recorded"] = l1(Mo, Rec, "Rec")
    blk = {}
    for k, kn in enumerate(G.KINDS, 1):
        pm, pb = agg(Mo, lambda o: per_kind(o, k)), agg(B, lambda o: per_kind(o, k))
        cm, cb = pm[2].sum() / pm[3].sum(), pb[2].sum() / pb[3].sum()
        c = pd(pm[2], pm[3], pb[2], pb[3]); pr = pd(pm[0], pm[1], pb[0], pb[1])
        blk[kn] = {"count_model": float(cm), "count_reference": float(cb), "count_relative": float(cm / cb - 1) if cb else None,
                   "count_se": c["se"], "persistence4_model": float(pm[0][:, 3].sum() / max(pm[1][:, 3].sum(), 1)),
                   "persistence4_reference": float(pb[0][:, 3].sum() / max(pb[1][:, 3].sum(), 1)), "persistence4_se": pr["se"][3]}
        blk[kn]["fires"] = bool((cb and abs(cm / cb - 1) > 0.30 and abs(cm - cb) > 3 * c["se"]) or
                                (kn != "arrows" and blk[kn]["persistence4_reference"] - blk[kn]["persistence4_model"] > 0.10))
    res["S3_bookkeeping_blocker"] = blk
    # 3. L2b: time-to-event curves (share of futures with the event by step k), stratified as PLAN.md says
    s0 = so[rows[:, 0]] + rows[:, 3]
    asleep0 = L["s_sleep"][s0].astype(bool)
    body = L["s_body"]
    last_drop = lambda c: np.array([min(int(t0 - np.flatnonzero(np.diff(body[so[e]:so[e] + t0 + 1, c]) < 0)[-1]), 99)
                                    if (np.diff(body[so[e]:so[e] + t0 + 1, c]) < 0).any() else 99 for e, _, _, t0, _, _ in rows])
    pos, zp, zm = L["s_pos"][s0].astype(int), L["s_zombies_p"][s0].astype(int), L["s_zombies_m"][s0]
    near = (zm & (np.abs(zp - pos[:, None]).sum(-1) <= 2)).any(1)
    strata = {"all": np.ones(len(rows), bool), "asleep_at_start": asleep0, "zombie_within_2": near, "no_zombie_within_2": ~near}
    use = {"damage": ["all", "zombie_within_2", "no_zombie_within_2"], "death": ["all", "zombie_within_2", "no_zombie_within_2"],
           "wake": ["asleep_at_start"], "unlock": ["all"]}
    for c, q in ((1, "food"), (2, "drink"), (3, "energy")):
        ld, use[f"{q}_drop"] = last_drop(c), ["all"]
        for lo, hi in ((0, 9), (10, 19), (20, 98), (99, 99)):
            strata[f"{q}_since_last_drop_{lo}_{hi}"] = (ld >= lo) & (ld <= hi)
            use[f"{q}_drop"].append(f"{q}_since_last_drop_{lo}_{hi}")
    fam = {}
    FE = lambda Ts: [[first_events(o, asleep0[w]) for o in t] for w, t in enumerate(Ts)]
    fm, fb = FE(Mo), FE(B)
    for name, snames in use.items():
        for sname in snames:
            idx = np.flatnonzero(strata[sname])
            if not len(idx):
                continue
            cur = lambda F_: np.array([[sum(f[name] <= k for f in F_[w]) for k in range(1, H + 1)] for w in idx], float)
            dm, db = cur(fm), cur(fb)
            nm, nb = np.full(len(idx), float(MF)), np.full(len(idx), 8.)
            d, se = O.paired_diff(dm, nm[:, None].repeat(H, 1), db, nb[:, None].repeat(H, 1), ep_of[idx], rng)
            fam[f"{name}|{sname}"] = {"starts": int(len(idx)), "model_by_k": (dm.sum(0) / nm.sum()).round(4)[[0, 3, 7, 15]].tolist(),
                                      "reference_by_k": (db.sum(0) / nb.sum()).round(4)[[0, 3, 7, 15]].tolist(),
                                      "max_abs_diff": float(np.abs(d).max()), "rejects_at_k": [int(k + 1) for k in np.flatnonzero(np.abs(d) > 3 * se + 1e-9)]}
    res["L2b_time_to_event"] = fam
    ret = lambda Ts: np.array([sum(float(o["reward"][:O.n_alive(o) - 1].sum()) for o in t) for t in Ts])
    res["L2b_return_per_future"] = pd(ret(Mo), np.full(len(rows), float(MF)), ret(B), np.full(len(rows), 8.)) | \
        {"model": float(ret(Mo).sum() / (MF * len(rows))), "reference": float(ret(B).sum() / (8 * len(rows)))}
    # 4. internal consistency
    cons = lambda Ts: tuple(np.array([np.sum([consistency(o)[i] for o in t], 0) for t in Ts]) for i in range(2))
    (cmn, cmd), (cbn, cbd) = cons(Mo), cons(B)
    d, se = O.paired_diff(cmn, cmd, cbn, cbd, ep_of, rng)
    res["L2b_internal_consistency"] = {k: {"model": float(cmn[:, i].sum() / max(cmd[:, i].sum(), 1)), "reference": float(cbn[:, i].sum() / max(cbd[:, i].sum(), 1)),
                                           "diff": float(d[i]), "se": float(se[i]), "rejects": bool(abs(d[i]) > 3 * se[i] + 1e-9)} for i, k in enumerate(CONS)}
    # 5. N5: the generated belief's HUD against its own trajectory's read HUD; the real recorded future's alike
    alive_m = ~np.concatenate([np.zeros_like(z["m_done"][..., :1]), z["m_done"][..., :-1]], -1)      # frame j generated before termination
    okm = (z["m_bhud"] == z["m_hud"][:, :, 1:]) & alive_m[..., None]
    okr = (z["rec_bhud"] == z["rec_rd_hud"][:, 1:]) & z["rec_valid"][..., None]
    n5 = {}
    for gi, (gname, cols) in enumerate((("health", [0]), ("food", [1]), ("drink", [2]), ("energy", [3]), ("inventory", list(range(4, 16))))):
        for k in (1, 4, 8, 16):
            nm_ = okm[:, :, k - 1][..., cols].sum((1, 2)).astype(float); dm_ = (alive_m[:, :, k - 1].sum(1) * len(cols)).astype(float)
            nr_ = okr[:, k - 1][..., cols].sum(-1).astype(float); dr_ = (z["rec_valid"][:, k - 1] * len(cols)).astype(float)
            dd, ss = O.paired_diff(nm_, dm_, nr_, dr_, ep_of, rng)
            n5[f"{gname}@{k}"] = {"generated": float(nm_.sum() / max(dm_.sum(), 1)), "real": float(nr_.sum() / max(dr_.sum(), 1)),
                                  "diff": float(dd), "se": float(ss)}
    res["N5_belief_hud_exact"] = n5
    (HERE / f"gates{TAG}{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("reader_error_on_simulator_frames", "S3_bookkeeping_blocker")}, indent=1), flush=True)


OL = importlib.import_module("artifacts.experiments.20261114_s2_harness.oracle_l1")

if __name__ == "__main__":
    stage, n = sys.argv[1], (int(sys.argv[2]) if len(sys.argv) > 2 else None)
    {"readers": readers, "oracle": lambda: oracle(n), "model": lambda: model_futures(n), "gates": lambda: gates(n)}[stage]()
