"""Does the world use a place memory? PLAN.md (2026-11-23): P1.5 with a per-cell read of the last-seen coarse token at
the cell's world address (simulator addresses), denoiser-only G1 fine-tune by test 3's recipe, then the revisit test.

    MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_place_read.place_read smoke
    MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_place_read.place_read train
    MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_place_read.place_read eval 4000
"""
import dataclasses
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
import torch.nn as nn

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
S2 = importlib.import_module("artifacts.experiments.20261119_s2.s2")
B = importlib.import_module("artifacts.experiments.20261119_s2.s2b")
PM = importlib.import_module("artifacts.experiments.20261123_place_memory.place_memory")
DEV = S2.DEV
N, MAP, FEAT = 81, 63, 21
KR = torch.arange(MAP) // 9 - 3                                         # map cell k -> (row - 3, col - 4) from the player
KC = torch.arange(MAP) % 9 - 4
SHIFT = (3, 0)                                                          # the swap's wrong address, in world cells


def age_features(age, present):
    a, pr = age.float().clamp(min=0), present.float()
    return torch.stack([pr, pr * torch.log1p(a) / math.log(1000.0), pr * torch.exp(-a / 8), pr * torch.exp(-a / 32),
                        pr * torch.exp(-a / 128)], -1)


class Store:
    """Per stream and world cell (64 x 64): the last coarse token observed there and its episode frame."""

    def __init__(self, n, dev):
        self.tok = torch.zeros(n, 64, 64, 16, device=dev)
        self.time = torch.full((n, 64, 64), -1, dtype=torch.long, device=dev)

    def clone(self):
        s = Store.__new__(Store); s.tok, s.time = self.tok.clone(), self.time.clone(); return s

    def reset(self, m):
        self.tok[m] = 0; self.time[m] = -1

    def addr(self, pos, shift=(0, 0)):
        kr, kc = KR.to(pos.device), KC.to(pos.device)
        r, c = pos[:, 0:1] + kr + shift[0], pos[:, 1:2] + kc + shift[1]
        ok = (r >= 0) & (r < 64) & (c >= 0) & (c < 64)
        return r.clamp(0, 63), c.clamp(0, 63), ok

    def read(self, pos, t, shift=(0, 0)):
        r, c, ok = self.addr(pos, shift)
        ii = torch.arange(len(pos), device=pos.device)[:, None]
        tm = self.time[ii, r, c]
        present = ok & (tm >= 0)
        f = torch.cat([self.tok[ii, r, c] * present[..., None], age_features(t[:, None] - tm, present)], -1)
        out = torch.zeros(len(pos), N, FEAT, device=pos.device)
        out[:, :MAP] = f
        return out

    def write(self, pos, t, ptok, m):
        r, c, ok = self.addr(pos)
        ok = ok & m[:, None]
        ii = torch.arange(len(pos), device=pos.device)[:, None].expand_as(r)
        self.tok[ii[ok], r[ok], c[ok]] = ptok[:, :MAP].float()[ok]
        self.time[ii[ok], r[ok], c[ok]] = t[:, None].expand_as(r)[ok]


def install(den):
    """The place read: a per-cell MLP (last layer zero-initialised) of the 21 entry features, added to each periphery
    cell's token after its embedding, beside the record's read (test 3's additive path)."""
    dev = next(den.parameters()).device
    den.place_read = nn.Sequential(nn.Linear(FEAT, den.W), nn.GELU(), nn.Linear(den.W, den.W)).to(dev)
    nn.init.zeros_(den.place_read[2].weight); nn.init.zeros_(den.place_read[2].bias)
    den.place_input = torch.zeros(1, 1, N, FEAT, device=dev)
    embed = den.embed

    def with_place(x, memory, fix):
        tok, ang = embed(x, memory, fix)
        pin = den.place_input if den.place_input.shape[0] == tok.shape[0] else den.place_sub
        add = den.place_read(pin.float()).to(tok.dtype)
        return torch.cat([tok[..., :N, :] + add, tok[..., N:, :]], -2), ang
    den.embed = with_place
    loss = M.shortcut_loss

    def shortcut_loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
        d.place_sub = d.place_input[bootstrap]                          # the bootstrap target's row subset
        return loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels, noise)
    M.shortcut_loss = shortcut_loss


class PosStreams(D.Streams):
    """D.Streams.next exactly (the same draws), also returning each frame's episode index and time."""

    def next(self, steps):
        recs = []
        for i in range(len(self.cur)):
            new = []
            for _ in range(steps):
                (e, uid), t = self.cur[i], self.t[i]
                new.append((e, uid, t, *(int(x) for x in D.fixation_prior(self.rng, ()))))
                if t == self.eps[e].T:
                    self.cur[i], self.t[i] = self._pull(), 0
                else:
                    self.t[i] = t + 1
            pad = [None] * (self.prefix - len(self.hist[i]))
            recs.append(pad + self.hist[i] + new)
            self.hist[i] = (self.hist[i] + new)[-self.prefix:]
        c = D.gather(self.eps, recs, self.offsets)
        c["ep_index"] = np.array([[-1 if r is None else r[0] for r in row] for row in recs], np.int64)
        c["ts"] = np.array([[0 if r is None else r[2] for r in row] for row in recs], np.int64)
        return c


def label_positions(eps):
    """Simulator positions of the TRAIN split's episodes from the S0 labels, by (shard, slot)."""
    L = np.load(S.LABELS / "s0_train_labels.npz")
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(L["ep_shard"], L["ep_slot"]))}
    start = np.array([so[key[(e.shard, e.slot)]] for e in eps])
    assert all(int(L["ep_T"][key[(e.shard, e.slot)]]) == e.T for e in eps)
    return L["s_pos"].astype(np.int64), start


def chunk_features(store, ptok, pos, ts, first, valid, P):
    """Entries before each frame [n, L, 81, FEAT]; writes each valid frame; returns the store after frame L - P - 1
    (before the next chunk's prefix, which repeats this chunk's last P frames)."""
    n, L = first.shape
    feats = torch.zeros(n, L, N, FEAT, device=ptok.device)
    snap = store
    for j in range(L):
        v = valid[:, j]
        store.reset(v & first[:, j])
        feats[:, j] = store.read(pos[:, j], ts[:, j]) * v[:, None, None]
        store.write(pos[:, j], ts[:, j], ptok[:, j], v)
        if j == L - P - 1:
            snap = store.clone()
    return feats, snap


def train(updates=4000, saves=(2000, 4000)):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    RUN.mkdir(parents=True, exist_ok=True)
    ck = torch.load(S2.P15, map_location=DEV, weights_only=False)
    model, cfg = S2.AI.load(S2.P15)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    for p in model.parameters():
        p.requires_grad_(False)
    den = model.denoiser
    install(den)
    params = list(den.parameters())
    for p in params:
        p.requires_grad_(True)
    model.gaze_forecast = None                                          # not trained here; it needs the chooser's rows
    model.train()
    den.checkpoint = cfg["p1"]["activation_checkpointing"]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    split = D.load_split(cfg)
    streams = PosStreams(split["train"], d["streams"], P, seed=ck["world_seed"], targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    spos, start = label_positions(split["train"])
    carry, cache, record = ck["carry"], ck["cache"].to(DEV), ck["record"].to(DEV)
    store = Store(d["streams"], DEV)
    base, micro = ck["state"]["update"], d["accumulate_at_64"]
    log, t0, losses = [], time.time(), []
    for u in range(1, updates + 1):
        chunk = streams.next(64)
        fr = torch.as_tensor(chunk["frames"], device=DEV)
        with torch.no_grad(), TR.autocast(DEV, cfg):
            ptok = torch.stack([C.coarse(model.perception, fr[b])[0] for b in range(len(fr))]).float()
        ei, ts = chunk["ep_index"], chunk["ts"]
        pos = np.where(ei[..., None] >= 0, spos[start[np.maximum(ei, 0)] + ts], 0)
        feats, store = chunk_features(store, ptok, torch.as_tensor(pos, device=DEV), torch.as_tensor(ts, device=DEV),
                                      torch.as_tensor(chunk["first"], device=DEV), torch.as_tensor(chunk["valid"], device=DEV), P)
        boot = TR.bootstrap_rows(cfg, ck["world_seed"], base + u)
        for g in opt.param_groups:
            g["lr"] = 1e-4 * min(1.0, u / 100)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), micro):
            den.place_input = feats[rows]
            terms, c_new, cache_new, rec_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                            None, cfg, boot[rows], base + u, None, record[rows], None)
            (terms["g1"] / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows], record[rows] = cache_new, rec_new
            losses.append(float(terms["g1"]))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if u == 1:
            print("peak GiB", torch.cuda.max_memory_allocated() / 2 ** 30, flush=True)
        if u % 100 == 0 or u == updates:
            log.append({"update": u, "g1": float(np.mean(losses[-100 * micro:])), "read_weight_norm": float(den.place_read[2].weight.norm()),
                        "present_fraction": float(feats[..., 63 - 63:MAP, 16].mean()), "sec_per_update": (time.time() - t0) / u})
            print(json.dumps(log[-1]), flush=True)
            (RUN / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
        if u in saves:
            torch.save({"denoiser": den.state_dict(), "update": u}, RUN / f"denoiser_{u}.pt")


def episode_entries(Lpos, ptok, shift=(0, 0)):
    """For one episode: entries before each frame [T + 1, 81, FEAT] at the true address (or shifted)."""
    T1 = len(Lpos)
    st = Store(1, DEV)
    pos = torch.as_tensor(Lpos, device=DEV)
    out = torch.zeros(T1, N, FEAT, device=DEV)
    for t in range(T1):
        tt = torch.tensor([t], device=DEV)
        out[t] = st.read(pos[t:t + 1], tt, shift)[0]
        st.write(pos[t:t + 1], tt, ptok[t:t + 1].to(DEV), torch.ones(1, dtype=torch.bool, device=DEV))
    return out


@torch.no_grad()
def evaluate(update, smoke=False):
    model, cfg = S2.load(); S2.EV.guard()
    for p in model.parameters():
        p.requires_grad_(False)
    den = model.denoiser
    install(den)
    if update:
        den.load_state_dict(torch.load(RUN / f"denoiser_{update}.pt", map_location=DEV, weights_only=False)["denoiser"])
    model.eval()
    R = S2.load_readers(model, cfg)
    P = cfg["data"]["context_prefix"]
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz")); so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    eps = np.flatnonzero(L["ep_eps"] < 0.99)[:8 if smoke else 120]
    rng = np.random.default_rng(20261123)                              # place_memory's starts, the same draws
    rows = []
    for e in eps:
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(int)
        ts = [t for t in range(8, T) if (pos[t + 1] != pos[t]).any()]
        for t in sorted(rng.choice(ts, min(len(ts), 6 if smoke else 40), replace=False)) if ts else []:
            rows.append((int(e), int(t)))
    sub = {e: i for i, e in enumerate(eps)}
    enc = S.encode(model, S2.sublabels(L, eps), cfg)
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    ent = {e: episode_entries(L["s_pos"][so[e]:so[e] + int(L["ep_T"][e]) + 1].astype(np.int64), enc[sub[e]]["p"].float()) for e in eps}
    wrong = {e: episode_entries(L["s_pos"][so[e]:so[e] + int(L["ep_T"][e]) + 1].astype(np.int64), enc[sub[e]]["p"].float(), SHIFT) for e in eps}
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    MF, NB = 4, 16
    variants = ("correct", "empty", "wrong") if not update == 0 else ("correct",)
    drawn = {v: {} for v in variants}
    for bi in range(0, len(rows), NB):
        batch = rows[bi:bi + NB]
        rr = [(sub[e], t) for e, t in batch]
        at = lambda key, dt: torch.stack([enc[i][key][t + dt] for i, t in rr]).to(DEV)
        rep = lambda x: x.repeat_interleave(MF, 0)
        rin = S.window(model, enc, rr, P - 1)
        rs = []
        for i, t in rr:
            with TR.autocast(DEV, cfg):
                rs.append(S.history(model, enc[i]["p"][:t + 1].to(DEV).float()[None])[2])
        rs = torch.cat(rs)
        for v in variants:
            g = torch.Generator(device=DEV).manual_seed(20261123 + bi)
            cache = den.new_cache()
            with TR.autocast(DEV, cfg):
                for dt in range(1 - P, 1):
                    den.place_input = rep(torch.stack([ent[e][t + dt] for e, t in batch]))[:, None]
                    cache = M.context_pass(den, tuple(rep(at(k, dt)).float()[:, None] for k in ("p", "g", "v")),
                                           rep(at("prev_action", dt))[:, None],
                                           S.cond(model.belief.memory(rep(at("bp", dt)).float())[:, None], rep(rin[:, P - 1 + dt])[:, None]),
                                           rep(at("fix", dt))[:, None], cache, cfg, g)
                q = torch.stack([ent[e][t + 1] for e, t in batch])
                if v != "correct":
                    newc = torch.zeros(len(batch), N, dtype=torch.bool, device=DEV)
                    for b, (e, t) in enumerate(batch):                  # entering cells at t + 1: not in view at t
                        p0, p1 = L["s_pos"][so[e] + t].astype(int), L["s_pos"][so[e] + t + 1].astype(int)
                        for k in range(MAP):
                            wr, wc = p1[0] + KR[k].item(), p1[1] + KC[k].item()
                            newc[b, k] = bool(abs(wr - p0[0]) > 3 or abs(wc - p0[1]) > 4)
                    alt = torch.zeros_like(q) if v == "empty" else torch.stack([wrong[e][t + 1] for e, t in batch])
                    q = torch.where(newc[..., None], alt, q)
                den.place_input = rep(q)[:, None]
                parts = M.sample(den, dims, rep(at("action", 0))[:, None],
                                 S.cond(model.belief.memory(rep(at("bp", 1)).float())[:, None], rep(rs)[:, None]),
                                 rep(at("fix", 1))[:, None], cache, cfg, g)
            te = B.terrain(R, parts[0].float(), parts[2].float())[:, 0]   # [N, 7, 9]
            for b, (e, t) in enumerate(batch):
                drawn[v][(e, t)] = te[MF * b:MF * (b + 1)]
        if bi % (NB * 20) == 0:
            print("batch", bi // NB, flush=True)
    res = {"update": update, "variants": {}}
    for v in variants:
        z = cells(L, so, rows, drawn[v], rterr)
        np.savez_compressed(HERE / f"cells_{update}_{v}{'_smoke' if smoke else ''}.npz", **z)
        res["variants"][v] = PM.summarise(z, np.random.default_rng(0), reps=300 if smoke else 1000)
        print(v, json.dumps(res["variants"][v]["matched_minus_first"])[:600], flush=True)
    (HERE / f"eval_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")


def cells(L, so, rows, drawn, rterr):
    """place_memory's cell rule: map cells entering view at t + 1, real reader class, inward neighbour, gap group."""
    rec = {k: [] for k in ("episode", "group", "context", "cls", "agree")}
    RRn, CCn = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
    by_ep = {}
    for e, t in rows:
        by_ep.setdefault(e, []).append(t)
    for e, starts in by_ep.items():
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(int)
        last, k, starts = {}, 0, sorted(starts)
        for s in range(T + 1):
            for i in range(7):
                for j in range(9):
                    last[(pos[s][0] + RRn[i, j], pos[s][1] + CCn[i, j])] = s
            while k < len(starts) and starts[k] == s:
                t = starts[k]; k += 1
                now = {(pos[t][0] + RRn[i, j], pos[t][1] + CCn[i, j]) for i in range(7) for j in range(9)}
                dd = pos[t + 1] - pos[t]
                real = rterr[e][t + 1]
                for i in range(7):
                    for j in range(9):
                        cell = (pos[t + 1][0] + RRn[i, j], pos[t + 1][1] + CCn[i, j])
                        if cell in now or real[i, j] < 0:
                            continue
                        ii, jj = i - dd[0], j - dd[1]
                        if not (0 <= ii < 7 and 0 <= jj < 9) or real[ii, jj] < 0:
                            continue
                        gap = (t + 1 - last[cell]) if cell in last else None
                        group = "first" if gap is None else next(f"{lo}-{hi}" for lo, hi in PM.BUCKETS if lo <= gap <= hi)
                        rec["episode"].append(int(e)); rec["group"].append(group)
                        rec["context"].append("predictable" if real[i, j] == real[ii, jj] else "surprising")
                        rec["cls"].append(int(real[i, j])); rec["agree"].append(float((np.asarray(drawn[(e, t)])[:, i, j] == real[i, j]).mean()))
    return {k: np.asarray(v) for k, v in rec.items()}


def smoke():
    """A 30-update fine-tune into run_smoke/ and the zero-read equivalence: at update 0 the read is inert."""
    global RUN
    RUN = HERE / "run_smoke"
    if not (RUN / "denoiser_30.pt").exists():
        train(updates=30, saves=(30,))
    evaluate(0, smoke=True)
    evaluate(30, smoke=True)


if __name__ == "__main__":
    {"train": lambda: train(), "eval": lambda: evaluate(int(sys.argv[2])), "smoke": smoke}[sys.argv[1]]()
