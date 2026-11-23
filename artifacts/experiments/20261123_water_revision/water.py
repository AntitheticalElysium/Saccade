"""How fast do the weights absorb a permanent rule change? PLAN.md (2026-11-23, decision 7). P1.5 checkpoint 10's own
P1 loop continues as decision 4's life: stream 0 lives 859 DEV episodes re-simulated with water giving +2, streams
1-15 replay uniformly; then an unannounced return to +1. Read on the rule-change panel (fast_channel's protocol).

    JAX_PLATFORMS=cpu python -m artifacts.experiments.20261123_water_revision.water simulate
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261123_water_revision.water smoke
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261123_water_revision.water run
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import importlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import s1 as S

FC = importlib.import_module("artifacts.experiments.20261122_rule_change.fast_channel")
C4 = importlib.import_module("artifacts.experiments.20261120_switching_dissection.c4_sample")
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
HERE = Path(__file__).resolve().parent
DATA, RUN = HERE / "data", HERE / "run"
DEV = torch.device("cuda")
LIFE, SEED, RAMP = -1, 20261123, 100                                    # the life's synthetic shard; ramp updates
READ_AT, RETURN_AT = (200, 400), (100, 250)                              # plus the end of phase +2


def episodes():
    """DEV labels, the panel (fast_channel.main's 150 episodes) and the life (the other DEV episodes, shuffled)."""
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rises = [int((np.diff(L["s_body"][so[e]:so[e + 1], FC.DRINK].astype(int)) > 0).sum()) for e in range(len(L["ep_T"]))]
    panel = [e for e in range(len(rises)) if rises[e] >= 2][:FC.N_EPISODES]
    life = np.random.default_rng(SEED).permutation([e for e in range(len(rises)) if e not in set(panel)]).tolist()
    return L, panel, life


def simulate():
    """Exact replays: the panel under +1 and +2 (checked against the stored frames under +1), the life under +2."""
    L, panel, life = episodes(); dl = S.episodes(L)
    for sub in ("panel", "life"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    diff = 0
    for e in panel:
        for rule, extra in (("A", 0), ("B", 1)):
            frames, drink, rew, term, acts = FC.replay(dl[e].shard, dl[e].slot, extra)
            if rule == "A":
                stored = FC.RP.episode_fields(dl[e].shard, dl[e].slot)["observations"].numpy()
                diff = max(diff, int(np.abs(frames.astype(int) - stored[:len(frames)].astype(int)).max()))
            np.savez_compressed(DATA / "panel" / f"{e}_{rule}.npz", frames=frames, drink=drink, rew=rew, term=term, acts=acts)
    index = []
    for k, e in enumerate(life):
        frames, drink, rew, term, acts = FC.replay(dl[e].shard, dl[e].slot, 1)
        np.savez_compressed(DATA / "life" / f"{k:04d}.npz", frames=frames, drink=drink, rew=rew, term=term, acts=acts)
        index.append({"k": k, "source": int(e), "T": int(len(acts)), "plus2": int((np.diff(drink) == 2).sum())})
    summary = {"panel": len(panel), "life": len(life), "ruleA_max_pixel_diff": diff, "life_steps": sum(x["T"] for x in index),
               "life_plus2": sum(x["plus2"] for x in index)}
    (DATA / "life_index.json").write_text(json.dumps({"summary": summary, "episodes": index}) + "\n")
    print(json.dumps(summary), flush=True)


_FIELDS, _CACHE = D.RP.episode_fields, {}


def fields(shard, slot):
    """RP.episode_fields, plus the life's re-simulated episodes on the synthetic shard."""
    if shard != LIFE:
        return _FIELDS(shard, slot)
    if slot not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        z = np.load(DATA / "life" / f"{slot:04d}.npz")
        _CACHE[slot] = {"observations": torch.from_numpy(z["frames"]), "actions_taken": torch.from_numpy(z["acts"].astype(np.int64)),
                        "rewards": torch.from_numpy(z["rew"].astype(np.float32)), "terminated": torch.from_numpy(z["term"])}
    return _CACHE[slot]


D.RP.episode_fields = fields


class LifeStreams(D.Streams):
    """P1's streams under decision 4's replay: stream 0 walks `order` (the life) in turn; streams 1-15 draw uniformly,
    with replacement, from the corpus (eps[:corpus]) plus the life episodes stream 0 has completed. Also emits each
    frame's episode index and time."""

    def __init__(self, eps, n, prefix, seed, corpus, order):
        super().__init__(eps, n, prefix, seed)
        self.corpus, self.order, self.done, self.k = corpus, order, [], 0

    def pull(self, i):
        self.uid += 1
        if i == 0:
            self.k += 1
            return self.order[self.k - 1], self.uid
        j = int(self.rng.integers(self.corpus + len(self.done)))
        return (j if j < self.corpus else self.done[j - self.corpus]), self.uid

    def next(self, steps):
        recs = []
        for i in range(len(self.cur)):
            new = []
            for _ in range(steps):
                (e, uid), t = self.cur[i], self.t[i]
                new.append((e, uid, t, *(int(x) for x in D.fixation_prior(self.rng, ()))))
                if t == self.eps[e].T:
                    if i == 0 and e >= self.corpus:
                        self.done.append(e)
                    self.cur[i], self.t[i] = self.pull(i), 0
                else:
                    self.t[i] = t + 1
            pad = [None] * (self.prefix - len(self.hist[i]))
            recs.append(pad + self.hist[i] + new)
            self.hist[i] = (self.hist[i] + new)[-self.prefix:]
        c = D.gather(self.eps, recs, self.offsets)
        c["ep_index"] = np.array([[-1 if r is None else r[0] for r in row] for row in recs], np.int64)
        c["ts"] = np.array([[0 if r is None else r[2] for r in row] for row in recs], np.int64)
        return c


@torch.no_grad()
def reading(model, cfg, R, panel, tag, counts, split, out_dir, n_windows=None):
    """fast_channel.main's protocol on the cached panel replays, the P1 monitor, the counts -> reading_{tag}.json."""
    was = model.training
    model.eval()
    t0, rec = time.time(), []
    for e in panel:
        out = {}
        for rule in ("A", "B"):
            z = np.load(DATA / "panel" / f"{e}_{rule}.npz")
            frames, drink, rew, term, acts = (z[k] for k in ("frames", "drink", "rew", "term", "acts"))
            ups = [t for t in range(1, len(drink)) if drink[t] > drink[t - 1]]
            plus2 = [t for t in ups if drink[t] - drink[t - 1] == 2]
            units = [t for t in ups if t >= FC.P and drink[t - 1] <= (7 if rule == "B" else 8)]
            if not units:
                continue
            st = FC.episode_state(model, cfg, frames, rew, term, acts, e)
            real = FC.drink_digit(R, st[1].p[0, units], st[1].v[0, units])
            draws = np.stack([FC.drink_digit(R, *C4.samples(model, cfg, st, e, units, dr)) for dr in range(FC.DRAWS)], 1)
            for i, t in enumerate(units):
                before = [u for u in plus2 if u < t]
                out.setdefault(rule, []).append({"t": t, "prev": int(drink[t - 1]), "true": int(drink[t]), "real_read": int(real[i]),
                                                 "drawn": draws[i].tolist(), "plus2_before": len(before),
                                                 "gap": (t - before[-1]) if before else None})
        rec.append({"e": e, **out})
    groups = {"A": lambda u, r: r == "A", "B_first": lambda u, r: r == "B" and u["plus2_before"] == 0,
              "B_recent_evidence": lambda u, r: r == "B" and u["plus2_before"] > 0 and u["gap"] <= 8,
              "B_older_evidence": lambda u, r: r == "B" and u["plus2_before"] > 0 and u["gap"] > 8}
    res = {"tag": tag, "counts": dict(counts), "episodes": len(rec), "groups": {}}
    for name, sel in groups.items():
        us = [u for x in rec for r in ("A", "B") for u in x.get(r, []) if sel(u, r)]
        ok = [u for u in us if u["real_read"] == u["true"]]
        if not us:
            continue
        g = {"events": len(us), "reader_floor": len(ok) / len(us), "reader_correct_events": len(ok)}
        for part, vs in (("all", us), ("reader_correct", ok)):
            inc = np.array([[d - u["prev"] for d in u["drawn"]] for u in vs]) if vs else np.zeros((0, FC.DRAWS))
            g[part] = {"share_plus2": float((inc == 2).mean()) if len(vs) else None, "share_plus1": float((inc == 1).mean()) if len(vs) else None,
                       "share_zero": float((inc == 0).mean()) if len(vs) else None}
        res["groups"][name] = g
    res["monitor"] = TR.monitor(model, cfg, split["monitor"], n_windows)
    res["seconds"] = time.time() - t0
    (out_dir / f"reading_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    (out_dir / f"reading_{tag}_units.json").write_text(json.dumps(rec) + "\n")
    print("reading", tag, json.dumps({k: v["reader_correct"]["share_plus2"] for k, v in res["groups"].items()}),
          "plus1_A", res["groups"].get("A", {}).get("reader_correct", {}).get("share_plus1"), "monitor_g1", res["monitor"]["g1"],
          f"{res['seconds']:.0f}s", flush=True)
    model.train(was)
    return res


def run(smoke=False):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    out_dir = HERE / ("run_smoke" if smoke else "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(S2.P15, map_location=DEV, weights_only=False)
    cfg, ws, base = ck["cfg"], ck["world_seed"], ck["state"]["update"]
    d, P, o = cfg["data"], cfg["data"]["context_prefix"], cfg["optimizer"]
    split = D.load_split(cfg)
    corpus, nC = split["train"], len(split["train"])
    _, panel, _ = episodes()
    index = json.loads((DATA / "life_index.json").read_text())["episodes"]
    lifeB = [D.Episode(LIFE, x["k"], x["T"], f"life{x['k']}") for x in index]
    drinks = [np.load(DATA / "life" / f"{x['k']:04d}.npz")["drink"] for x in index]
    order = list(range(nC, nC + len(lifeB))) + np.random.default_rng(SEED + 1).permutation(nC).tolist()
    streams = LifeStreams(corpus + lifeB, d["streams"], P, ws, nC, order)
    streams.load_state(ck["streams"])                                    # P1.5's streams 1-15 continue where they were
    streams.cur[0], streams.t[0] = streams.pull(0), 0                    # stream 0 starts the life
    model = M.M0(cfg).to(DEV)
    model.load_state_dict(ck["model"])
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    opt.load_state_dict(ck["opt"])                                       # the moments continue (our fine-tune rule)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("perception.")}
    for n, p in model.named_parameters():
        if n in frozen:
            p.requires_grad_(False)                                      # the readers read P1.5's encoder (PLAN)
    controller = C.Controller(model, cfg, out_dir, DEV)
    controller.load_state(dict(ck["fixation_controller"], trace_rows=0, log_bytes=0))
    carry, cache, record = ck["carry"], ck["cache"].to(DEV), ck["record"].to(DEV)
    torch.set_rng_state(ck["rng"]["cpu"].cpu()); torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
    R = S2.load_readers(model, cfg)
    model.train()
    counts = {"updates": 0, "life_steps": 0, "distinct_plus2": 0, "replayed_plus2": 0, "life_episodes_done": 0}
    if smoke:
        r0 = reading(model, cfg, R, panel[:3], "u0", counts, split, out_dir, n_windows=20)
        ref = {x["e"]: x for x in json.loads((HERE.parent / "20261122_rule_change/fast_channel_units.json").read_text())}
        mine = json.loads((out_dir / "reading_u0_units.json").read_text())
        same = all(a.get(r) == ref[a["e"]].get(r) for a in mine for r in ("A", "B"))
        print("smoke: reading at update 0 reproduces fast_channel's draws:", same, flush=True)
    S_, micro = d["chunk_steps"][1], d["accumulate_at_256"]
    phase_end, log, t0, u = None, open(out_dir / "log.jsonl", "a"), time.time(), 0
    while True:
        u += 1
        chunk = streams.next(S_)
        if phase_end is None and streams.k > len(lifeB):
            phase_end = u
        controller.prepare(chunk, streams)
        f = min(1.0, 0.1 + 0.9 * u / RAMP)
        for g in opt.param_groups:
            g["lr"] = o["lr"] * g["lr_scale"] * f
        opt.zero_grad(set_to_none=True)
        step = base + u - 1
        boot = TR.bootstrap_rows(cfg, ws, step)
        fw = np.random.default_rng([cfg["heads"]["forecast"]["seed"], ws, step]).integers(49, size=(d["streams"], S_)) \
            if cfg["heads"].get("forecast") else None
        logs = []
        for rows in np.array_split(np.arange(d["streams"]), micro):
            out = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], None, cfg, boot[rows], step,
                               None if fw is None else fw[rows], record[rows], TR.gaze_rows(controller, rows, S_))
            terms, c_new, cache_new = out[:3]
            ((model.harmoniser(TR.groups(terms)) + terms["noise"]) / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows], record[rows] = cache_new, out[3]
            logs.append({k: float(v.detach()) for k, v in terms.items()})
        del terms, c_new, cache_new, out
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"], error_if_nonfinite=True))
        opt.step()
        controller.learn(int(chunk["valid"][:, P:].sum()))
        ei, ts, val = chunk["ep_index"][:, P:], chunk["ts"][:, P:], chunk["valid"][:, P:]
        for i in range(d["streams"]):
            for e, t, v in zip(ei[i], ts[i], val[i]):
                if v and e >= nC:
                    counts["life_steps"] += i == 0
                    if t >= 1 and drinks[e - nC][t] - drinks[e - nC][t - 1] == 2:
                        counts["distinct_plus2" if i == 0 else "replayed_plus2"] += 1
        counts["updates"], counts["life_episodes_done"] = u, len(streams.done)
        if u % 25 == 0 or u == 1:
            rec = {"update": u, "lr": o["lr"] * f, "grad_norm": gn, **counts,
                   **{k: float(np.mean([l[k] for l in logs])) for k in logs[0]}, "sec_per_update": (time.time() - t0) / u,
                   "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(json.dumps(rec), flush=True)
        tags = []
        if smoke and u == 3:
            tags = ["smoke3"]
        elif not smoke:
            if phase_end is None and u in READ_AT:
                tags = [f"u{u}"]
            if u == phase_end:
                tags = [f"u{u}_end_plus2"]
            if phase_end is not None and u - phase_end in RETURN_AT:
                tags = [f"u{u}_return{u - phase_end}"]
        for tag in tags:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
            torch.save({"model": model.state_dict(), "update": u, "counts": counts}, out_dir / f"model_{tag}.pt")
            reading(model, cfg, R, panel[:3] if smoke else panel, tag, counts, split, out_dir, n_windows=20 if smoke else None)
        if smoke and u == 3:
            break
        if phase_end is not None and u - phase_end >= RETURN_AT[-1]:
            break
    controller.close()


if __name__ == "__main__":
    {"simulate": simulate, "smoke": lambda: run(smoke=True), "run": lambda: run()}[sys.argv[1]]()
