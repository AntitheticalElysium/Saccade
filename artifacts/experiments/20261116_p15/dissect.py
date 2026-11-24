"""P1.5's fixed readings outside the queue (PLAN, readings), rerun with committed code. Gemini's checks 15 and 17 saved
no code, and check 15's inventory rows measured cells that do not change on their events (inventory_tokens.py).
  catalogue  check 15: use_check.catalogue's classes, rows, fixations, seeds and measures on P1.5 (one arm), the
             record in the belief's input and in the denoiser's conditioning (S helpers; through the previous frame at
             every context pass and at the draw). F6.1's arm is in ../20261115_f61_dissection/catalogue.json, same rows.
  schedule   check 17: schedule_pairs.read_bank's banks, seeds and 2 x 2 swap, m = the belief stream (each member's
             own record in Mamba's input), c = the record the denoiser reads (the place of F6.1's recency input).
  drift      the loss-share drift and token variance (../20261116_loss_share/drift.py, its frames and measure) for
             P1.5, F6.1 beside it as the reproduction check against that study's drift.json.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_p15.dissect catalogue
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_p15.dissect schedule
    GAZE_CPU=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261116_p15.dissect drift
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CK = Path(os.environ.get("DISSECT_CKPT", HERE / "run/ckpt_10.pt"))   # another world (11-19: the switching arm)
TAG = os.environ.get("DISSECT_TAG", "")                                 # its outputs: catalogue{TAG}.json, schedule{TAG}.json


@torch.no_grad()
def catalogue(events=128, quiet=64, batch=8):
    UC = importlib.import_module("artifacts.experiments.20261115_f61_dissection.use_check")
    M, C, D, TR, DS = UC.M, UC.C, UC.D, UC.TR, UC.DS
    S = UC.G.S
    model, cfg = DS.NP.IF.load(CK.resolve())
    model.requires_grad_(False)
    model.eval()
    den, dev = model.denoiser, DS.DEV
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    after = UC.G.cohorts(L)["after_reset"]
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    eps = S.episodes(L)
    ep_of = np.repeat(np.arange(len(eps)), L["ep_T"] + 1)
    cls = UC.classes(L, after)
    xs = {}                                                             # pass 1: this world's inventory cells
    for e in range(4, len(eps), 5):
        ft = torch.as_tensor(D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"].numpy(), device=dev)
        xs[e] = UC.change(torch.cat([C.coarse(model.perception, z)[0] for z in ft.split(512)]).float().cpu().numpy())
    X = np.concatenate([xs[e] for e in range(4, len(eps), 5)])
    rows_g = np.concatenate([off[e] + np.arange(len(xs[e])) for e in range(4, len(eps), 5)])
    pos = np.full(len(L["s_body"]), -1); pos[rows_g] = np.arange(len(rows_g))
    cellmap = {}
    for name, (ev, no, cell, cue) in list(cls.items()):
        if cell is None:
            mean = X[pos[np.flatnonzero(ev)], 63:].mean(0)
            cell = 63 + int(mean.argmax())
            cellmap[name] = {"cell": cell, "change": float(mean.max()), "next": float(np.sort(mean)[-2])}
            cls[name] = (ev, no, cell, cue)
    rows = []                                                           # (state g, class, event?): check 15's
    for ci_, (name, (ev, no, cell, cue)) in enumerate(cls.items()):
        r = np.random.default_rng(1515 + ci_)
        for flag, mask, k in ((1, ev, events), (0, no, quiet)):
            idx = np.flatnonzero(mask)
            rows += [(int(g), name, flag) for g in r.choice(idx, min(k, len(idx)), replace=False)]
    rec, by_ep = [], {}
    for j, (g, name, flag) in enumerate(rows):
        by_ep.setdefault(ep_of[g], []).append(j)
    R = DS.DRAWS
    for e, js in sorted(by_ep.items()):
        ef = D.RP.episode_fields(eps[e].shard, eps[e].slot)
        f = ef["observations"].numpy()
        n = len(f)
        a = np.r_[ef["actions_taken"].numpy(), 0]
        rw, td = np.r_[0., ef["rewards"].numpy()], np.r_[False, ef["terminated"].numpy()]
        assert len(a) == len(rw) == len(td) == n == L["ep_T"][e] + 1
        fix = torch.as_tensor(D.fixation_prior(np.random.default_rng([15000, int(e)]), (1, n)), device=dev)
        act = torch.as_tensor(a[None], device=dev)
        prev = torch.cat([act.new_full((1, 1), -1), act[:, :-1]], 1)
        ft = torch.as_tensor(f[None], device=dev)
        need = sorted({int(rows[j][0] - off[e]) - 8 + k for j in js for k in range(9)})
        where = {t: i for i, t in enumerate(need)}
        with TR.autocast(dev, cfg):
            parts = [model.perception(ft[:, s:s + 512], fix[:, s:s + 512]) for s in range(0, n, 512)]
            parts = tuple(torch.cat(z, 1) for z in zip(*(q.parts() for q in parts)))
            pp = M.Percept(*parts, fix)
            reads, kept, _ = S.history(model, parts[0], keep=need)       # from the episode start, as the belief
            rin = lambda frames: kept[0, [where[t] for t in frames]]     # the record through the frame before each
            y, _ = model.belief(model.belief.tokens(S.embed(model, pp, reads), torch.as_tensor(rw[None], device=dev, dtype=torch.float32),
                                                    torch.as_tensor(td[None], device=dev), act), model.belief.initial(1, dev))
            bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
            shapes = [tuple(z.shape[2:]) for z in parts]
            for b0 in range(0, len(js), batch):
                jb = js[b0:b0 + batch]
                ts = [int(rows[j][0] - off[e]) for j in jb]
                cells = [cls[rows[j][1]][2] for j in jb]
                B = len(ts)
                last = torch.stack([parts[0][0, t - 1, c].float() for t, c in zip(ts, cells)])
                delta = torch.stack([parts[0][0, t, c].float() for t, c in zip(ts, cells)]) - last
                seed = 1_000_003 * int(e) + b0
                rng = torch.Generator(device=dev).manual_seed(seed)
                cache = den.new_cache()
                for k in range(8):
                    fi = [t - 8 + k for t in ts]
                    cache = M.context_pass(den, tuple(z[0, fi][:, None] for z in parts), prev[0, fi][:, None],
                                           S.cond(model.belief.memory(bp[0, fi][:, None]), rin(fi)[:, None]), fix[0, fi][:, None],
                                           cache, cfg, rng)
                cache = {"t": cache["t"], "kv": [tuple(z.repeat_interleave(R, 0) for z in kv) for kv in cache["kv"]]}
                pz, vz, xp = UC.physical(den, shapes, prev[0, ts][:, None].repeat_interleave(R, 0),
                                         S.cond(model.belief.memory(bp[0, ts][:, None]).repeat_interleave(R, 0),
                                                rin(ts)[:, None].repeat_interleave(R, 0)),
                                         fix[0, ts][:, None].repeat_interleave(R, 0), cache, cfg,
                                         torch.Generator(device=dev).manual_seed(seed + 1))
                for r_, (j, c) in enumerate(zip(jb, cells)):
                    sl = slice(r_ * R, (r_ + 1) * R)
                    fin = pz[sl, 0, c]
                    rec.append({"row": j, "class": rows[j][1], "event": rows[j][2],
                                "kappa": [float(UC.kappa(z[sl, 0, c], last[r_], delta[r_]).mean()) for z in xp],
                                "kappa_final": float(UC.kappa(fin, last[r_], delta[r_]).mean()),
                                "drawn_toward": float((UC.kappa(fin, last[r_], delta[r_]) > .5).float().mean()),
                                "drawn_change": float(((fin.float() - last[r_]).square().sum(-1) > 1).float().mean()),
                                "true_change": float(delta[r_].square().sum())})
        print("episode", e, "rows", len(js), flush=True)
    res = {"cells": {k: int(v[2]) for k, v in cls.items()}, "cue": {k: v[3] for k, v in cls.items()},
           "inventory_cells": cellmap, "p15": {}}
    for name in cls:
        ev = [r for r in rec if r["class"] == name and r["event"]]
        no = [r for r in rec if r["class"] == name and not r["event"]]
        if ev:
            res["p15"][name] = {"events": len(ev), "first_step": UC.ci([r["kappa"][0] for r in ev]),
                                "steps": np.mean([r["kappa"] for r in ev], 0).tolist(),
                                "kappa_final": float(np.mean([r["kappa_final"] for r in ev])),
                                "drawn_toward": UC.ci([r["drawn_toward"] for r in ev]),
                                "drawn_change_event": float(np.mean([r["drawn_change"] for r in ev])),
                                "drawn_change_quiet": float(np.mean([r["drawn_change"] for r in no])) if no else None,
                                "true_change": float(np.mean([r["true_change"] for r in ev]))}
            print(name, json.dumps(res["p15"][name]), flush=True)
    (HERE / f"catalogue{TAG}.json").write_text(json.dumps(res, indent=1) + "\n")


@torch.no_grad()
def schedule():
    SP = importlib.import_module("artifacts.experiments.20261115_f61_dissection.schedule_pairs")
    UC, DS, D, M, C, TR, TJ = SP.UC, SP.DS, SP.D, SP.M, SP.C, SP.TR, SP.TJ
    S = UC.G.S
    model, cfg = DS.NP.IF.load(CK.resolve())
    model.requires_grad_(False)
    model.eval()
    den, dev = model.denoiser, DS.DEV
    readers = TJ.readers(model, cfg, slots=(64, 66))

    def read_bank(rows, slot, sign, reader):
        cells = []
        for i, (prefix, frames, rewards, terms, q, acts, truth) in enumerate(rows):
            old_f, old_a, old_r, old_d = prefix
            n = len(old_f)
            f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
            a = np.concatenate([np.repeat(old_a[None], 2, 0), acts], 1)
            rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
            td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
            fix = torch.as_tensor(np.repeat(D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=dev)
            query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=dev)
            act = torch.as_tensor(a, device=dev)
            prev = torch.cat([act.new_full((2, 1), -1), act[:, :-1]], 1)
            ft = torch.as_tensor(f, device=dev)
            cell = {}
            with TR.autocast(dev, cfg):
                p = model.perception(ft, fix)
                reads, rin, after = S.history(model, p.p, keep=range(n + q - 7, n + q + 1))   # each member's own record
                y, _ = model.belief(model.belief.tokens(S.embed(model, p, reads), torch.as_tensor(rw, device=dev, dtype=torch.float32),
                                                        torch.as_tensor(td, device=dev), act), model.belief.initial(2, dev))
                bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
                nxt = model.perception(torch.as_tensor(frames[0:1, q + 1:q + 2], device=dev), query[None, None])
                last, delta = p.p[0, -1, slot].float(), nxt.p[0, 0, slot].float() - p.p[0, -1, slot].float()
                for m, c in ((0, 0), (1, 1), (0, 1), (1, 0)):
                    rng = torch.Generator(device=dev).manual_seed(14000 + i)
                    cache = den.new_cache()
                    for t in range(n + q - 7, n + q + 1):
                        cache = M.context_pass(den, tuple(z[0:1, t:t + 1] for z in p.parts()), prev[0:1, t:t + 1],
                                               S.cond(model.belief.memory(bp[m:m + 1, t:t + 1]), rin[c:c + 1, t - n - q + 7:t - n - q + 8]),
                                               fix[0:1, t:t + 1], cache, cfg, rng)
                    cache = {"t": cache["t"], "kv": [tuple(z.expand(DS.DRAWS, *z.shape[1:]) for z in kv) for kv in cache["kv"]]}
                    pz, vz, xs = UC.physical(den, [tuple(x.shape[2:]) for x in p.parts()], act.new_zeros(DS.DRAWS, 1),
                                             S.cond(model.belief.memory(y[m:m + 1, -1:]).expand(DS.DRAWS, -1, -1, -1),
                                                    after[c:c + 1, None].expand(DS.DRAWS, -1, -1, -1)),
                                             query[None, None].expand(DS.DRAWS, 1, -1), cache, cfg,
                                             torch.Generator(device=dev).manual_seed(16000 + i))
                    digits = reader(torch.cat([pz[:, 0, slot], vz[:, 0]], 1).float())["y"].argmax(-1).cpu().numpy()
                    cell[f"m{m}c{c}"] = (float(((digits < truth[0, 0]) if sign < 0 else (digits > truth[0, 0])).mean()),
                                         float(UC.kappa(xs[0][:, 0, slot], last, delta).mean()))
            cells.append(cell)
        r = {k: np.array([c[k][0] for c in cells]) for k in cells[0]}
        k0 = {k: np.array([c[k][1] for c in cells]) for k in cells[0]}
        return {"pairs": len(cells), "rate": {k: float(v.mean()) for k, v in r.items()},
                "first_step": {k: float(v.mean()) for k, v in k0.items()},
                "native_due_minus_delayed": UC.ci(r["m0c0"] - r["m1c1"]),
                "record_effect": UC.ci((r["m0c0"] + r["m1c0"] - r["m0c1"] - r["m1c1"]) / 2),
                "memory_effect": UC.ci((r["m0c0"] + r["m0c1"] - r["m1c0"] - r["m1c1"]) / 2),
                "first_step_due_minus_delayed": UC.ci(k0["m0c0"] - k0["m1c1"])}

    res = {"due_episode_step": {}, "readers": {s: v[1] for s, v in readers.items()}, "p15": {}}
    for name, rows, slot, sign in SP.banks():
        res["due_episode_step"][name] = float(np.median([len(r[0][0]) + r[4] + 1 for r in rows]))
        res["p15"][name] = read_bank(rows, slot, sign, readers[slot][0])
        print(name, json.dumps(res["p15"][name]), flush=True)
    (HERE / f"schedule{TAG}.json").write_text(json.dumps(res, indent=1) + "\n")


def drift():
    DR = importlib.import_module("artifacts.experiments.20261116_loss_share.drift")
    DR.RUNS = {"F6.1": DR.RUNS["F6.1"], "P1.5": HERE / "run"}
    DR.HERE = HERE                                                      # writes HERE / "drift.json"
    DR.main()


if __name__ == "__main__":
    {"catalogue": catalogue, "schedule": schedule, "drift": drift}[sys.argv[1]]()
