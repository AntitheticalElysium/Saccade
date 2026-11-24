"""Can history change a prediction immediately? PLAN.md (2026-11-23, decision 7): P1.5's denoiser fine-tuned on TRAIN
episodes replayed with a thirst period drawn per episode (16 and 25 held out), judged by when the next drink drop is
drawn at each episode's own period.

    JAX_PLATFORMS=cpu python -m artifacts.experiments.20261123_thirst_period.thirst simulate K N   (shard K of N)
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261123_thirst_period.thirst smoke | train | eval U
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import functools, importlib, inspect, json, sys, time, types
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

FC = importlib.import_module("artifacts.experiments.20261122_rule_change.fast_channel")
C4 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c4_sample")
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
HERE = Path(__file__).resolve().parent
DATA, RUN = HERE / "data", HERE / "run"
DEV = torch.device("cuda")
SEED, NATIVE, HELD = 20261123, 21, (16, 25)
TRAINED = [p for p in range(12, 31) if p not in (16, 21, 25)]
TEST = (16, 25, 21, 13, 29)
THIRSTY = -2                                                            # the synthetic shard of the replayed episodes


def periods(n):
    rng = np.random.default_rng(SEED)
    return np.where(rng.random(n) < 0.3, NATIVE, rng.choice(TRAINED, n))


@functools.cache
def stepper(p):
    """Craftax-Classic's step with both `new_thirst > 20` comparisons at `> p - 1` (clock_curriculum_eda's patch)."""
    import jax
    from craftax.craftax_classic import game_logic as game
    src = inspect.getsource(game.update_player_intrinsics)
    assert src.count("new_thirst > 20") == 2
    ns = dict(vars(game))
    exec(compile(src.replace("new_thirst > 20", f"new_thirst > {p - 1}"), f"<thirst period {p}>", "exec"), ns)
    step = types.FunctionType(game.craftax_step.__code__, ns)
    env, params, reset, _, frame = FC.RP.env_and_render()
    static = env.static_env_params
    return jax.jit(lambda k, s, a: step(k, s, a, params, static)), jax.jit(lambda s: env.is_terminal(s, params)), reset, frame


def replay(shard, slot, p):
    """The recorded actions under the collector's keys with thirst period p -> frames, drink, sleep, reward, term, acts."""
    import jax.numpy as jnp
    step, terminal, reset, frame = stepper(p)
    reset_key, env_keys = FC.RP._slot_keys(shard, slot)
    acts = FC.RP.episode_fields(shard, slot)["actions_taken"].numpy()
    _, s = reset(reset_key)
    frames, drink, sleep, rew, term = [np.asarray(frame(s))], [int(s.player_drink)], [bool(s.is_sleeping)], [], []
    for t, a in enumerate(acts):
        s, r = step(env_keys[t], s, jnp.int32(a))
        d = bool(terminal(s))
        frames.append(np.asarray(frame(s))); drink.append(int(s.player_drink)); sleep.append(bool(s.is_sleeping))
        rew.append(float(r)); term.append(d)
        if d:
            break
    return np.stack(frames), np.array(drink), np.array(sleep), np.array(rew, np.float32), np.array(term), acts[:len(rew)].astype(np.int64)


def test_episodes():
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    return L, sorted(np.random.default_rng(SEED).choice(np.flatnonzero(L["ep_T"] >= 120), 60, replace=False).tolist())


def simulate(k=0, n=1):
    for sub in ("train", "test"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    cfg = torch.load(S2.P15, map_location="cpu", weights_only=False)["cfg"]
    train = D.load_split(cfg)["train"]
    per = periods(len(train))
    if k == 0:                                                           # the native patch reproduces the stored frames
        diff = 0
        for ep in train[:3]:
            frames = replay(ep.shard, ep.slot, NATIVE)[0]
            stored = FC.RP.episode_fields(ep.shard, ep.slot)["observations"].numpy()
            diff = max(diff, int(np.abs(frames.astype(int) - stored[:len(frames)].astype(int)).max()))
        print("native patch max pixel diff", diff, flush=True)
        assert diff <= 1
    L, test = test_episodes()
    dl = S.episodes(L)
    for i in range(k, len(train), n):
        f = DATA / "train" / f"{i:05d}.npz"
        if not f.exists():
            frames, drink, sleep, rew, term, acts = replay(train[i].shard, train[i].slot, int(per[i]))
            np.savez_compressed(f, frames=frames, drink=drink, sleep=sleep, rew=rew, term=term, acts=acts, period=int(per[i]))
        if i % 500 < n:
            print("train", i, flush=True)
    for j, e in enumerate(test):
        if j % n != k:
            continue
        for p in TEST:
            frames, drink, sleep, rew, term, acts = replay(dl[e].shard, dl[e].slot, p)
            np.savez_compressed(DATA / "test" / f"{e}_{p}.npz", frames=frames, drink=drink, sleep=sleep, rew=rew, term=term, acts=acts)
    print("shard", k, "done", flush=True)


_FIELDS, _CACHE = D.RP.episode_fields, {}


def fields(shard, slot):
    if shard != THIRSTY:
        return _FIELDS(shard, slot)
    if slot not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        z = np.load(DATA / "train" / f"{slot:05d}.npz")
        _CACHE[slot] = {"observations": torch.from_numpy(z["frames"]), "actions_taken": torch.from_numpy(z["acts"]),
                        "rewards": torch.from_numpy(z["rew"]), "terminated": torch.from_numpy(z["term"])}
    return _CACHE[slot]


D.RP.episode_fields = fields


def train(updates=2000, saves=(1000, 2000), out=RUN):
    """Stage 1's / test 3's recipe: the denoiser only, G1, over fresh streams of the replayed episodes."""
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(S2.P15, map_location=DEV, weights_only=False)
    model, cfg = S2.AI.load(S2.P15)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    for prm in model.parameters():
        prm.requires_grad_(False)
    den = model.denoiser
    params = list(den.parameters())
    for prm in params:
        prm.requires_grad_(True)
    model.gaze_forecast = None
    model.train()
    den.checkpoint = cfg["p1"]["activation_checkpointing"]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    n_train = len(D.load_split(cfg)["train"])
    eps = [D.Episode(THIRSTY, i, int(len(np.load(DATA / "train" / f"{i:05d}.npz")["acts"])), f"thirst{i}") for i in range(n_train)]
    streams = D.Streams(eps, d["streams"], P, seed=SEED)
    carry, cache, record = ck["carry"], ck["cache"].to(DEV), ck["record"].to(DEV)
    base, micro = ck["state"]["update"], d["accumulate_at_64"]
    log, t0, losses = [], time.time(), []
    for u in range(1, updates + 1):
        chunk = streams.next(64)
        boot = TR.bootstrap_rows(cfg, ck["world_seed"], base + u)
        for grp in opt.param_groups:
            grp["lr"] = 1e-4 * min(1.0, u / 100)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), micro):
            terms, c_new, cache_new, rec_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                            None, cfg, boot[rows], base + u, None, record[rows], None)
            (terms["g1"] / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows], record[rows] = cache_new, rec_new
            losses.append(float(terms["g1"].detach()))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if u % 100 == 0 or u == updates:
            log.append({"update": u, "g1": float(np.mean(losses[-100 * micro:])), "sec_per_update": (time.time() - t0) / u})
            print(json.dumps(log[-1]), flush=True)
            (out / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
        if u in saves:
            torch.save({"denoiser": den.state_dict(), "update": u}, out / f"denoiser_{u}.pt")


@torch.no_grad()
def evaluate(update, out=RUN, n_eps=None, draws=4):
    """P(drawn drop) at each target frame u whose lag since the last drop is in {p - 2 .. p + 2} or 21, on awake runs
    with at least two drops seen, digit >= 1 and no drink at u. Reader-correct events only."""
    model, cfg = FC.AI.load(FC.C5.CK); model.requires_grad_(False).eval()
    if update:
        model.denoiser.load_state_dict(torch.load(out / f"denoiser_{update}.pt", map_location=DEV, weights_only=False)["denoiser"])
    R = S2.load_readers(model, cfg)
    _, test = test_episodes()
    rec = []
    for e in test[:n_eps]:
        for p in TEST:
            z = np.load(DATA / "test" / f"{e}_{p}.npz")
            frames, drink, sleep, rew, term, acts = (z[k] for k in ("frames", "drink", "sleep", "rew", "term", "acts"))
            drops = [t for t in range(1, len(drink)) if drink[t] == drink[t - 1] - 1]
            units = []
            for u in range(FC.P, len(drink)):
                before = [t for t in drops if t < u]
                if len(before) < 2:
                    continue
                lag = u - before[-1]
                if not ((p - 2 <= lag <= p + 2) or lag == NATIVE):
                    continue
                if sleep[before[-2] - 1:u + 1].any() or drink[u - 1] < 1 or drink[u] > drink[u - 1]:
                    continue
                units.append((u, lag))
            if not units:
                continue
            us = [u for u, _ in units]
            st = FC.episode_state(model, cfg, frames, rew, term, acts, e)
            real = FC.drink_digit(R, st[1].p[0, us], st[1].v[0, us])
            dr = np.stack([FC.drink_digit(R, *C4.samples(model, cfg, st, e, us, k)) for k in range(draws)], 1)
            for i, (u, lag) in enumerate(units):
                rec.append({"e": e, "p": p, "u": u, "lag": lag, "prev": int(drink[u - 1]), "true": int(drink[u]),
                            "real_read": int(real[i]), "drawn": dr[i].tolist()})
        print("episode", e, flush=True)
    res = {"update": update, "units": len(rec), "by_period": {}}
    for p in TEST:
        g = {}
        for lag in sorted({r["lag"] for r in rec if r["p"] == p}):
            us = [r for r in rec if r["p"] == p and r["lag"] == lag and r["real_read"] == r["true"]]
            n_all = sum(r["p"] == p and r["lag"] == lag for r in rec)
            if us:
                g[str(lag)] = {"events": len(us), "reader_floor": len(us) / n_all, "due": lag == p,
                               "p_drawn_drop": float(np.mean([[d == r["prev"] - 1 for d in r["drawn"]] for r in us]))}
        res["by_period"][str(p)] = g
    (out / f"eval_{update}.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / f"eval_{update}_units.json").write_text(json.dumps(rec) + "\n")
    print(json.dumps({p: {l: round(v["p_drawn_drop"], 3) for l, v in g.items()} for p, g in res["by_period"].items()}), flush=True)


def smoke():
    out = HERE / "run_smoke"
    if not (out / "denoiser_20.pt").exists():
        train(updates=20, saves=(20,), out=out)
    evaluate(0, out=out, n_eps=2)
    evaluate(20, out=out, n_eps=2)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"simulate": lambda: simulate(int(a[1]), int(a[2])), "smoke": smoke, "train": lambda: train(),
     "eval": lambda: evaluate(int(a[1]))}[a[0]]()
