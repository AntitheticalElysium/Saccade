"""Does one observation of a changed rule change the frozen world's next prediction? (PLAN.md) Rule B: drinking water
gives +2 drink instead of +1. DEV episodes are replayed exactly (stored actions, the collector's keys) under rule B, and
under rule A as a check against the stored frames. P1.5 checkpoint 10 reads each; at every drink event its native 4-step
sample of that frame (the deletion panel's sampler and seeds, 8 draws) is read by S2's HUD reader.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261122_rule_change.fast_channel [smoke]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import importlib, json, sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

RP = importlib.import_module("artifacts.eda.replay")
AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
C4 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c4_sample")
C5 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c5_calibration")
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda")
DRINK, DRAWS, P, N_EPISODES = 2, 8, 8, 150          # s_body's drink index; draws per event; context prefix; episodes


def replay(shard, slot, extra):
    """The stored actions under the collector's keys; extra = 1 is rule B (applied after the game's own step).
    -> frames [T + 1, 63, 63, 3] uint8, drink [T + 1], reward [T], terminated [T], actions [T]."""
    _, _, reset, step, frame = RP.env_and_render()
    reset_key, env_keys = RP._slot_keys(shard, slot)
    acts = RP.episode_fields(shard, slot)["actions_taken"].numpy()
    _, s = reset(reset_key)
    frames, drink, rew, term = [np.asarray(frame(s))], [int(s.player_drink)], [], []
    for t, a in enumerate(acts):
        _, ns, r, d, _ = step(env_keys[t], s, jnp.int32(a))
        if extra and int(ns.player_drink) > int(s.player_drink):
            ns = ns.replace(player_drink=jnp.minimum(9, s.player_drink + 1 + extra).astype(s.player_drink.dtype))
        s = ns
        frames.append(np.asarray(frame(s))); drink.append(int(s.player_drink)); rew.append(float(r)); term.append(bool(d))
        if bool(d):
            break
    return np.stack(frames), np.array(drink), np.array(rew, np.float32), np.array(term), acts[:len(rew)]


@torch.no_grad()
def episode_state(model, cfg, frames, rew, term, acts, e):
    """D.gather's arrays for frames 0..T of one episode, then C2.episode's real pass -> (c, Pe, rin, bp)."""
    T, m = len(frames) - 1, cfg["p1"]["monitor"]
    ts = np.arange(T + 1); prev = ts > 0
    fix = np.array([D.fixation_prior(np.random.default_rng([m["fixed_fixation_seed"], int(e), int(t)]), ()) for t in ts])
    c = {"frames": frames[None], "action": np.where(ts < T, acts[np.minimum(ts, T - 1)], 0)[None],
         "prev_action": np.where(prev, acts[np.maximum(ts - 1, 0)], -1)[None],
         "reward": np.where(prev, rew[np.maximum(ts - 1, 0)], 0.0).astype(np.float32)[None],
         "terminated": (prev & term[np.maximum(ts - 1, 0)])[None], "first": (ts == 0)[None],
         "episode": np.ones((1, T + 1), np.int64), "valid": np.ones((1, T + 1), bool), "fixation": fix[None].astype(np.int64)}
    c = TR.tensors(c, DEV)
    with TR.autocast(DEV, cfg):
        Pe = M.Percept(*(torch.cat(z, 1) for z in zip(*[model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                                                        for a in range(0, T + 1, 512)])), c["fixation"])
        state, carry, rins, ys = model.record.zero(1, DEV), model.belief.initial(1, DEV), [], []
        for a in range(0, T + 1, 512):
            sl = slice(a, min(T + 1, a + 512))
            first = torch.zeros(1, sl.stop - a, dtype=torch.bool, device=DEV); first[:, 0] = a == 0
            ri, si = model.record.scan(model.record.input(Pe.p[:, sl], Pe.g[:, sl], Pe.fix[:, sl]), first, state); state = si[:, -1]
            y, carry = model.belief(model.belief.tokens(model.belief.percept_embed(M.Percept(Pe.p[:, sl], Pe.g[:, sl], Pe.v[:, sl], Pe.fix[:, sl]), si),
                                                        c["reward"][:, sl], c["terminated"][:, sl], c["action"][:, sl]), carry)
            rins.append(ri.to(torch.bfloat16)); ys.append(y[:, 1::2])
        y = torch.cat(ys, 1)
    return c, Pe, torch.cat(rins, 1), torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], 1)


def drink_digit(R, p, v):
    return R["hud"](S2.hud_x(p.float(), v.float()).half())["hud"].reshape(len(p), 16, 10).argmax(-1)[:, DRINK].cpu().numpy()


def main(smoke=False):
    model, cfg = AI.load(C5.CK); model.requires_grad_(False).eval()
    R = S2.load_readers(model, cfg)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dl = S.episodes(L)
    rises = [int((np.diff(L["s_body"][so[e]:so[e + 1], DRINK].astype(int)) > 0).sum()) for e in range(len(dl))]
    eps = [e for e in range(len(dl)) if rises[e] >= 2][:3 if smoke else N_EPISODES]
    rec, checks = [], {"ruleA_replay_max_pixel_diff": 0}
    for n_, e in enumerate(eps):
        ep = dl[e]
        out = {}
        for rule, extra in (("A", 0), ("B", 1)):
            frames, drink, rew, term, acts = replay(ep.shard, ep.slot, extra)
            if rule == "A":
                stored = RP.episode_fields(ep.shard, ep.slot)["observations"].numpy()
                checks["ruleA_replay_max_pixel_diff"] = max(checks["ruleA_replay_max_pixel_diff"],
                                                            int(np.abs(frames.astype(int) - stored[:len(frames)].astype(int)).max()))
            ups = [t for t in range(1, len(drink)) if drink[t] > drink[t - 1]]
            plus2 = [t for t in ups if drink[t] - drink[t - 1] == 2]
            units = [t for t in ups if t >= P and drink[t - 1] <= (7 if rule == "B" else 8)]
            if not units:
                continue
            st = episode_state(model, cfg, frames, rew, term, acts, e)
            real = drink_digit(R, st[1].p[0, units], st[1].v[0, units])
            draws = np.stack([drink_digit(R, *C4.samples(model, cfg, st, e, units, dr)) for dr in range(DRAWS)], 1)   # [U, D]
            for i, t in enumerate(units):
                before = [u for u in plus2 if u < t]
                out.setdefault(rule, []).append({"t": t, "prev": int(drink[t - 1]), "true": int(drink[t]), "real_read": int(real[i]),
                                                 "drawn": draws[i].tolist(), "plus2_before": len(before),
                                                 "gap": (t - before[-1]) if before else None})
        rec.append({"e": e, **out})
        print(n_, e, {k: len(v) for k, v in out.items()}, flush=True)
    res = {"episodes": len(rec), "draws": DRAWS, "checks": checks, "groups": {}}
    groups = {"A": lambda u, r: r == "A",
              "B_first": lambda u, r: r == "B" and u["plus2_before"] == 0,
              "B_recent_evidence": lambda u, r: r == "B" and u["plus2_before"] > 0 and u["gap"] <= 8,
              "B_older_evidence": lambda u, r: r == "B" and u["plus2_before"] > 0 and u["gap"] > 8}
    for name, sel in groups.items():
        us = [u for x in rec for r in ("A", "B") for u in x.get(r, []) if sel(u, r)]
        if not us:
            continue
        inc = np.array([[d - u["prev"] for d in u["drawn"]] for u in us])
        res["groups"][name] = {"events": len(us), "reader_floor": float(np.mean([u["real_read"] == u["true"] for u in us])),
                               "true_increment": float(np.mean([u["true"] - u["prev"] for u in us])),
                               "share_plus2": float((inc == 2).mean()), "share_plus1": float((inc == 1).mean()),
                               "share_zero": float((inc == 0).mean()), "share_other": float(((inc != 0) & (inc != 1) & (inc != 2)).mean())}
    b = res["groups"]
    if "B_first" in b:
        res["reading"] = {k: b[k]["share_plus2"] - b["B_first"]["share_plus2"] for k in ("B_recent_evidence", "B_older_evidence") if k in b}
    (HERE / f"fast_channel{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    (HERE / f"fast_channel_units{'_smoke' if smoke else ''}.json").write_text(json.dumps(rec) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main("smoke" in sys.argv[1:])
