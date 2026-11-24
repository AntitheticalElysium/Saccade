"""Decision 6's locked design, the world's side: stage B (the read, trained as stage 1 was) and the readings (PLAN.md).
Everything runs on batch.BatchMemory: training streams and panel episodes step together as lanes on the GPU.

The read: each map cell queries the memory at its place before the frame (batch.BatchMemory.read: the current
trail's nearest 100 entries, ranked by code similarity, at most 5 per place newest first, 25). Keys and values come
from an MLP of each candidate's whole entry, with its relative displacement expanded to the code's phases (cosine and
sine, so the read sees relative position). Cells with no world projection get only the null slot.

    MAMBA_DETERMINISTIC=1 LOC_DEVICE=cuda JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261124_learned_location.world smoke | train | readings 2000
"""
import dataclasses, importlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BT = importlib.import_module("artifacts.experiments.20261124_learned_location.batch")
LOC = BT.LOC
LM, PR, S2, D = LOC.LM, LOC.PR, LOC.S2, LOC.D
M, TR, S, PM, B_ = LM.M, LM.TR, LM.S, LM.PM, LM.B
HERE, DEV = LOC.HERE, LOC.DEV
RUN = LOC.RUN
MAP, NCELL, WIDTH, KC_, FEAT, REL = LM.MAP, LOC.NCELL, LOC.WIDTH, BT.KC_, BT.FEAT, BT.REL
FINE, FPRES = slice(30, 94), 94
SHIFT = torch.tensor([float(LM.SHIFT[0]), float(LM.SHIFT[1])], device=DEV)
LANES = 8                                               # panel episodes stepped together in the readings


def install(den, code_W):
    """The world's read (located_memory's): each map cell's input token queries its candidates and a null slot (4 heads
    of 16); keys and values from an MLP of the whole candidate; absent slots masked; output zero-initialised."""
    dev = next(den.parameters()).device
    den.register_buffer("code_W", code_W.detach().clone().to(dev), persistent=False)
    nin = FEAT - 2 + 2 * code_W.shape[0]
    den.mem_kv = nn.Sequential(nn.Linear(nin, 128), nn.GELU(), nn.Linear(128, 128)).to(dev)
    den.mem_q = nn.Linear(den.W, 64).to(dev)
    den.mem_null = nn.Parameter(torch.zeros(128, device=dev))
    den.mem_out = nn.Linear(64, den.W).to(dev)
    nn.init.zeros_(den.mem_out.weight); nn.init.zeros_(den.mem_out.bias)
    den.place_input = torch.zeros(1, 1, MAP, KC_, FEAT, device=dev)
    embed = den.embed

    def with_memory(x, memory, fix):
        tok, ang = embed(x, memory, fix)
        pin = den.place_input if den.place_input.shape[0] == tok.shape[0] else den.place_sub
        cell = tok[..., :MAP, :]
        pres = pin[..., 0:1] > 0
        ph = pin[..., REL].float() @ den.code_W.T
        z = torch.cat([pin[..., :REL.start], torch.cat([ph.cos(), ph.sin()], -1).to(pin.dtype) * pres, pin[..., REL.stop:]], -1)
        q = den.mem_q(cell).unflatten(-1, (4, 16))
        kv = den.mem_kv(z.to(cell.dtype))
        kv = torch.cat([kv, den.mem_null.to(kv.dtype).expand(*kv.shape[:-2], 1, 128)], -2)
        k, v = kv.unflatten(-1, (2, 4, 16)).unbind(-3)
        mask = torch.cat([pres[..., 0], torch.ones_like(pres[..., :1, 0])], -1)
        att = (torch.einsum("btchd,btcnhd->btchn", q, k) / 4.0).masked_fill(~mask[..., None, :], -1e4)
        o = torch.einsum("btchn,btcnhd->btchd", att.softmax(-1), v).flatten(-2)
        return torch.cat([cell + den.mem_out(o).to(cell.dtype), tok[..., MAP:, :]], -2), ang
    den.embed = with_memory
    loss = M.shortcut_loss

    def shortcut_loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
        d.place_sub = d.place_input[bootstrap]
        return loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels, noise)
    M.shortcut_loss = shortcut_loss


# ---------------------------------------------------------------- stage B: the read, trained on P1.5's streams

class StreamBatch:
    """The 16 training streams as lanes of one BatchMemory: a lane resets at its episode's start, takes that episode's
    outages (the training schedule, seeded by episode) and the motion sensor from S0 positions; the next chunk's
    prefix frames reuse the features already computed. Calibration counts accumulate on the GPU."""

    def __init__(self, parts, n, eps, spos, start):
        self.parts, self.eps, self.spos, self.start = parts, eps, spos, start
        self.C = max(e.T for e in eps) + 1
        self.mem = BT.BatchMemory(parts, n, self.C)
        self.n = n; self.ep = [-1] * n; self.last = [-1] * n
        self.out = torch.zeros(n, self.C, dtype=torch.bool, device=DEV)
        self.cache = torch.zeros(n, 16, MAP, KC_, FEAT, device=DEV, dtype=torch.float16)
        self.cache_t = torch.full((n, 16), -1, dtype=torch.long)
        self.cS, self.cN = {}, {}
        self.withheld = 0; self.steps = 0

    def flush_calibration(self, calib):
        for k, v in self.cS.items():
            calib.S[k] = calib.S.get(k, 0) + v.double().cpu(); calib.N[k] = calib.N.get(k, 0) + self.cN[k]
        self.cS, self.cN = {}, {}

    def feats(self, chunk, Pc):
        ei, ts, valid = chunk["ep_index"], chunk["ts"], chunk["valid"]
        n, L = ei.shape
        out = torch.zeros(n, L, MAP, KC_, FEAT, device=DEV, dtype=torch.float16)
        pa = torch.as_tensor(chunk["prev_action"], device=DEV)
        fixs = torch.as_tensor(chunk["fixation"], device=DEV)
        P, V, G = Pc.p.float(), Pc.v.float(), Pc.g.float()
        for j in range(L):
            new, tvec, dvec, sens = [], np.zeros(n, np.int64), np.zeros((n, 2), np.float32), np.ones(n, bool)
            for b in range(n):
                if not valid[b, j] or ei[b, j] < 0:
                    continue
                e, t = int(ei[b, j]), int(ts[b, j])
                if e != self.ep[b] or (t == 0 and self.last[b] >= 0):
                    self.mem.reset(torch.tensor([b], device=DEV))
                    self.ep[b], self.last[b] = e, -1; self.cache_t[b] = -1
                    m = LOC.outage_mask(self.eps[e].T, np.random.default_rng([20261124, e]))
                    self.out[b] = False; self.out[b, :len(m)] = torch.as_tensor(m, device=DEV)
                if t <= self.last[b]:
                    sl = t % 16
                    assert int(self.cache_t[b, sl]) == t
                    out[b, j] = self.cache[b, sl]; continue
                new.append(b); tvec[b] = t
                if t > 0:
                    dvec[b] = self.spos[self.start[e] + t] - self.spos[self.start[e] + t - 1]
                    sens[b] = not bool(self.out[b, t])
            if not new:
                continue
            nl = torch.tensor(new, device=DEV)
            tt = torch.as_tensor(tvec, device=DEV)
            act = torch.zeros(n, dtype=torch.bool, device=DEV); act[nl] = True
            d = torch.as_tensor(dvec, device=DEV); sensed = torch.as_tensor(sens, device=DEV)
            for b in new:                                                # the running calibration's counts
                if tvec[b] > 0 and np.abs(dvec[b]).sum() > 0:
                    key = (round(float(dvec[b][0]), 2), round(float(dvec[b][1]), 2))
                    cd = torch.cdist(self.mem.tok[b, tvec[b] - 1].float()[None], P[b, j][None], p=1)[0]
                    self.cS[key] = self.cS.get(key, 0) + cd; self.cN[key] = self.cN.get(key, 0) + 1
            with torch.no_grad():
                x, var = self.mem.predict(tt, pa[:, j], d, sensed)
                f, _ = self.mem.read(tt, x)
                self.mem.step(tt, P[:, j], V[:, j], G[:, j], fixs[:, j], x, var, sensed, act)
            out[nl, j] = f[nl].half()
            sl = tt[nl] % 16
            self.cache[nl, sl] = f[nl].half(); self.cache_t[nl.cpu(), sl.cpu()] = tt[nl].cpu()
            for b in new:
                self.last[b] = int(tvec[b])
            self.steps += len(new); self.withheld += int((~sensed[nl]).sum())
        return out


def train(updates=2000, saves=(1000, 2000)):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    RUN.mkdir(parents=True, exist_ok=True)
    parts = LOC.load()
    ck = torch.load(S2.P15, map_location=DEV, weights_only=False)
    model, cfg = S2.AI.load(S2.P15)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    for prm in model.parameters():
        prm.requires_grad_(False)
    den = model.denoiser
    install(den, parts["code"].W)
    params = [p for p in den.parameters() if p.requires_grad or True]
    for prm in params:
        prm.requires_grad_(True)
    model.gaze_forecast = None
    model.train()
    den.checkpoint = cfg["p1"]["activation_checkpointing"]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    split = D.load_split(cfg)
    streams = PR.PosStreams(split["train"], d["streams"], P, seed=ck["world_seed"], targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    spos, start = PR.label_positions(split["train"])
    carry, cache, record = ck["carry"], ck["cache"].to(DEV), ck["record"].to(DEV)
    sb = StreamBatch(parts, d["streams"], split["train"], spos, start)
    calib = parts["calib"]
    base, micro = ck["state"]["update"], d["accumulate_at_64"]
    log, t0, losses, tmem = [], time.time(), [], 0.0
    for u in range(1, updates + 1):
        chunk = streams.next(64)
        fr = torch.as_tensor(chunk["frames"], device=DEV)
        fx = torch.as_tensor(chunk["fixation"], device=DEV)
        with torch.no_grad(), TR.autocast(DEV, cfg):
            Pc = model.perception(fr, fx)
        torch.cuda.synchronize(); tm0 = time.time()
        feats = sb.feats(chunk, Pc)
        torch.cuda.synchronize(); tmem += time.time() - tm0
        if u % 100 == 0:                                                 # the running calibration (option 3)
            sb.flush_calibration(calib); calib.fit()
        boot = TR.bootstrap_rows(cfg, ck["world_seed"], base + u)
        for grp in opt.param_groups:
            grp["lr"] = 1e-4 * min(1.0, u / 100)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), micro):
            den.place_input = feats[rows].float()
            terms, c_new, cache_new, rec_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                            None, cfg, boot[rows], base + u, None, record[rows], None)
            (terms["g1"] / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows], record[rows] = cache_new, rec_new
            losses.append(float(terms["g1"].detach()))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if u == 1:
            print("peak GiB", torch.cuda.max_memory_allocated() / 2 ** 30, flush=True)
        if u % 100 == 0 or u == updates:
            log.append({"update": u, "g1": float(np.mean(losses[-100 * micro:])), "read_out_norm": float(den.mem_out.weight.norm()),
                        "corrections_applied": sb.mem.applied, "proposals": sb.mem.proposals, "steps": sb.steps,
                        "withheld_share": sb.withheld / max(sb.steps, 1), "calibration": calib.score(),
                        "memory_sec_per_update": tmem / u, "sec_per_update": (time.time() - t0) / u})
            print(json.dumps(log[-1]), flush=True)
            (RUN / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
        if u in saves:
            torch.save({"read": {k: v for k, v in den.state_dict().items()}, "update": u, "calib": calib.state()}, RUN / f"denoiser_{u}.pt")


# ---------------------------------------------------------------- readings

def load_world(update):
    model, cfg = S2.load(); S2.EV.guard()
    for prm in model.parameters():
        prm.requires_grad_(False)
    parts = LOC.load()
    install(model.denoiser, parts["code"].W)
    if update:
        st = torch.load(RUN / f"denoiser_{update}.pt", map_location=DEV, weights_only=False)
        model.denoiser.load_state_dict(st["read"], strict=False); parts["calib"].load(st["calib"])
    model.eval()
    return model, cfg, parts


def episode_dict(L, so, e, en):
    T = int(L["ep_T"][e])
    return {"p": en["p"], "v": en["v"], "g": en["g"], "fix": en["fix"], "prev_action": en["prev_action"],
            "pos": L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)}


def capture(parts, eps, mode="full", masks=None, want=None, shifted=None, cur=None, true_step=None, snap=None):
    """Panel episodes as lanes: at the requested frames, the read's features (correct), the current-only arm's (the
    same codes with repairs undone), the read 3 rows away with its best candidates (the swap's donors), the read placed
    by the true motion (reading 5); and memory snapshots after given frames (dream starts)."""
    B = len(eps)
    E = lambda x: x if x is not None else [set() for _ in range(B)]
    want, shifted, cur, true_step, snap = E(want), E(shifted), E(cur), E(true_step), E(snap)
    caps = [{"feat": {}, "wrong": {}, "donor": {}, "top": {}, "cur": {}, "true": {}, "snap": {}} for _ in range(B)]
    pos = [torch.as_tensor(e["pos"], dtype=torch.float32, device=DEV) for e in eps]

    def hook(mem, tt, x, active):
        t = int(tt[0])
        def lanes(sets):
            return [b for b in range(B) if bool(active[b]) and t in sets[b]]
        lw = lanes(want)
        if lw:
            f, top = mem.read(tt, x)
            for b in lw:
                caps[b]["feat"][t] = f[b].half().cpu(); caps[b]["top"][t] = tuple(z[b].cpu() for z in top) if top else None
        ls = lanes(shifted)
        if ls:
            f, top = mem.read(tt, x + SHIFT)
            for b in ls:
                caps[b]["wrong"][t] = f[b].half().cpu(); caps[b]["donor"][t] = tuple(z[b].cpu() for z in top) if top else None
        lc = lanes(cur)
        if lc:
            f, _ = mem.read(tt, x, X=mem.X - mem.R)
            for b in lc:
                caps[b]["cur"][t] = f[b].half().cpu()
        lt = lanes(true_step)
        if lt:
            xt = torch.stack([mem.X[b, t - 1] + (pos[b][t] - pos[b][t - 1]) if t <= len(pos[b]) - 1 else x[b] for b in range(B)])
            f, _ = mem.read(tt, xt)
            for b in lt:
                caps[b]["true"][t] = f[b].half().cpu()

    def post(mem, tt, active):
        t = int(tt[0])
        for b in range(B):
            if bool(active[b]) and t in snap[b]:
                caps[b]["snap"][t] = {"X": mem.X[b].clone(), "Xcur": (mem.X[b] - mem.R[b]).clone(), "trail": mem.trail[b].clone(), "cur": int(mem.cur[b])}
    mem, xpre = BT.run_lanes(parts, eps, mode, masks, hook, keep_log=True, post_hook=post)
    return mem, caps


def cell_keys(L, so, rows, rterr):
    """(episode, drawn frame, map cell) of each record PR.cells produces, in its order."""
    keys = []
    RRn, CCn = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
    by_ep = {}
    for e, t in rows:
        by_ep.setdefault(e, []).append(t)
    for e, starts in by_ep.items():
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(int)
        for t in sorted(starts):
            now = {(pos[t][0] + RRn[i, j], pos[t][1] + CCn[i, j]) for i in range(7) for j in range(9)}
            dd = pos[t + 1] - pos[t]; real = rterr[e][t + 1]
            for i in range(7):
                for j in range(9):
                    cell = (pos[t + 1][0] + RRn[i, j], pos[t + 1][1] + CCn[i, j])
                    if cell in now or real[i, j] < 0:
                        continue
                    ii, jj = i - dd[0], j - dd[1]
                    if not (0 <= ii < 7 and 0 <= jj < 9) or real[ii, jj] < 0:
                        continue
                    keys.append((e, t + 1, i * 9 + j))
    return keys


def true_cell(L, so, e, f, c):
    pos = L["s_pos"][so[e] + f].astype(int)
    return (int(pos[0] + LM.KR[c]), int(pos[1] + LM.KC[c]))


def outage_events(L, so, e, T):
    """Reading 3's events (as stage A's gates chose them): outages of 8, 16 or 32 steps from movement steps preceding a
    return to old memory within 32 steps; then the first later view (after the return, within 64 steps) whose entering
    cells include cells first seen during the outage: the cells the repair matters for."""
    pos = L["s_pos"][so[e]:so[e] + T + 1].astype(int)
    first_seen = {}
    for t in range(T + 1):
        for k in range(MAP):
            first_seen.setdefault((pos[t][0] + int(LM.KR[k]), pos[t][1] + int(LM.KC[k])), t)
    W = lambda t, k: (pos[t][0] + int(LM.KR[k]), pos[t][1] + int(LM.KC[k]))
    entering = lambda t, k: abs(W(t, k)[0] - pos[t - 1][0]) > 3 or abs(W(t, k)[1] - pos[t - 1][1]) > 4
    events, t0, Ls = [], 9, [8, 16, 32]
    while t0 < T - 33:
        if (pos[t0] != pos[t0 - 1]).any():
            for s in range(t0 + 1, t0 + 33):
                if any(first_seen[W(s, k)] < t0 - 1 for k in range(MAP) if entering(s, k)):
                    Lo = Ls[len(events) % 3]
                    u, tgt = None, []
                    for uu in range(s + 1, min(T, s + 64) + 1):
                        tgt = [k for k in range(MAP) if entering(uu, k) and t0 <= first_seen[W(uu, k)] < t0 + Lo]
                        if tgt:
                            u = uu; break
                    if u is not None:
                        events.append({"t0": t0, "L": Lo, "return": s, "u": u, "targets": tgt})
                    t0 = (u if u is not None else s) + 17
                    break
        t0 += 1
    return events


def reading1(update, smoke=False):
    """Reading 1 (revisits with the agent's codes, the sense present), the fine swap holding metadata, reading 6 (the
    identity swap with the donors actually injected), and the per-read check."""
    model, cfg, parts = load_world(update)
    R = S2.load_readers(model, cfg)
    L, so, eps, rows, sub, enc = LM.panel(model, cfg, smoke)
    P = cfg["data"]["context_prefix"]
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    new = {(e, t): LM.entering(L, so, e, t) for e, t in rows}
    cur = {}

    def feat_of(v, e, t, drawn):
        f = cur[e]["feat"][t].to(DEV).float()
        if v == "correct" or not drawn:
            return f
        m = new[(e, t - 1)][:, None, None]
        if v == "empty":
            return torch.where(m, torch.zeros_like(f), f)
        w = cur[e]["wrong"][t].to(DEV).float()
        if v == "wrong":
            return torch.where(m, w, f)
        alt = f.clone()                                                  # fine swap: only the fine token moves
        both = (f[..., FPRES] > 0) & (w[..., FPRES] > 0)
        alt[..., FINE] = torch.where(both[..., None], w[..., FINE], f[..., FINE])
        return torch.where(m, alt, f)
    variants = ("correct", "empty", "wrong", "fine_swap") if update else ("correct",)
    drawn, donors, tops = {v: {} for v in variants}, {}, {}
    eps_r = [e for e in eps if any(r[0] == e for r in rows)]
    for k0 in range(0, len(eps_r), LANES):
        grp = eps_r[k0:k0 + LANES]
        ed = [episode_dict(L, so, e, enc[sub[e]]) for e in grp]
        want, shifted = [], []
        for e in grp:
            T = int(L["ep_T"][e]); ts_ = [t for ee, t in rows if ee == e]
            want.append({t + dt for t in ts_ for dt in range(-P, 2) if 0 <= t + dt <= T}); shifted.append({t + 1 for t in ts_})
        _, caps = capture(parts, ed, "full", None, want=want, shifted=shifted)
        cur.clear()
        for e, c in zip(grp, caps):
            cur[e] = c
            for t, lg in c["donor"].items():
                donors[(e, t)] = lg
            for t, lg in c["top"].items():
                tops[(e, t)] = lg
        rows_g = [r for r in rows if r[0] in set(grp)]
        dr = LM.one_step(model, cfg, R, enc, sub, rows_g, feat_of, variants, seed=20261123 + 1009 * k0)
        for v in variants:
            drawn[v].update(dr[v])
        print("episodes", k0 + len(grp), "of", len(eps_r), flush=True)
    res = {"update": update, "variants": {}}
    zs = {}
    for v in variants:
        z = PR.cells(L, so, rows, drawn[v], rterr); zs[v] = z
        res["variants"][v] = PM.summarise(z, np.random.default_rng(0), reps=300 if smoke else 1000)
        print(v, json.dumps(res["variants"][v]["matched_minus_first"])[:400], flush=True)
    keys = cell_keys(L, so, rows, rterr)
    c = zs[variants[0]]
    assert len(keys) == len(c["agree"])
    if update:
        wcls = np.full(len(keys), -1); right_top = np.zeros(len(keys), bool)
        for i, (e, t, kk) in enumerate(keys):
            dn = donors.get((e, t))
            if dn is not None and bool(dn[2][kk]):
                fr, ce = int(dn[0][kk]), int(dn[1][kk])
                if ce < MAP:
                    wcls[i] = int(rterr[e][fr][ce // 9, ce % 9])
            tp = tops.get((e, t))
            if tp is not None and bool(tp[2][kk]) and int(tp[1][kk]) < MAP:
                right_top[i] = true_cell(L, so, e, int(tp[0][kk]), int(tp[1][kk])) == true_cell(L, so, e, t, kk)
        base = (c["context"] == "surprising") & (c["group"] != "first") & (wcls >= 0) & (wcls != c["cls"])
        takes = {v: float((zs[v]["drawn"][base] == wcls[base][:, None]).mean()) if base.any() else None for v in ("correct", "empty", "wrong")}
        res["identity"] = {"cells": int(base.sum()), "takes_donor_class": takes,
                           "wrong_minus_empty": (takes["wrong"] - takes["empty"]) if base.any() else None}
        rev = (c["context"] == "surprising") & (c["group"] != "first")
        gain = zs["correct"]["agree"] - zs["empty"]["agree"]
        res["per_read"] = {"best_candidate_is_the_cell_share": float(right_top[rev].mean()) if rev.any() else None,
                           "gain_when_right": float(gain[rev & right_top].mean()) if (rev & right_top).any() else None,
                           "gain_when_wrong": float(gain[rev & ~right_top].mean()) if (rev & ~right_top).any() else None,
                           "cells_wrong": int((rev & ~right_top).sum())}
    (HERE / f"eval_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in res if k != "variants"}, default=float), flush=True)


def reading3(update, smoke=False):
    """Reading 3 (the causal chain: none, current-only, repair, clean sense; empty memory as the floor), 3b (wrong
    corrections among accepted ones) and 5 (the one-step motion cost)."""
    model, cfg, parts = load_world(update)
    R = S2.load_readers(model, cfg)
    L, so, eps, rows, sub, enc = LM.panel(model, cfg, smoke)
    P = cfg["data"]["context_prefix"]
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    arms = ("none", "current", "repair", "clean", "empty")
    agree = {a: [] for a in arms}; per_event = []; m5 = {"predicted": [], "true": []}
    cur = {}

    def feat_of(v, e, t, drawn):
        return cur[e][v][t].to(DEV).float()
    evs_all = {e: outage_events(L, so, e, int(L["ep_T"][e])) for e in eps}
    eps_e = [e for e in eps if evs_all[e]]
    for k0 in range(0, len(eps_e), LANES):
        grp = eps_e[k0:k0 + LANES]
        ed = [episode_dict(L, so, e, enc[sub[e]]) for e in grp]
        masks, want, t5s, want5 = [], [], [], []
        for e in grp:
            T = int(L["ep_T"][e]); m = np.zeros(T + 1, bool)
            for ev in evs_all[e]:
                m[ev["t0"]:ev["t0"] + ev["L"]] = True
            masks.append(m)
            want.append({ev["u"] + dt for ev in evs_all[e] for dt in range(-P, 1)})
            t5 = {min(ev["t0"] + 4, ev["t0"] + ev["L"] - 1) for ev in evs_all[e]}
            t5s.append(t5); want5.append({t + dt for t in t5 for dt in range(-P, 1)})
        mem_f, cf = capture(parts, ed, "full", masks, want=[a | b for a, b in zip(want, want5)], cur=want, true_step=t5s)
        _, cn = capture(parts, ed, "none", masks, want=want)
        _, cc = capture(parts, ed, "full", None, want=want)
        for b, e in enumerate(grp):
            cur[e] = {"repair": cf[b]["feat"], "current": {**cf[b]["feat"], **cf[b]["cur"]}, "none": cn[b]["feat"],
                      "clean": cc[b]["feat"], "empty": {t: torch.zeros_like(f) for t, f in cf[b]["feat"].items()},
                      "predicted": cf[b]["feat"], "true": {**cf[b]["feat"], **cf[b]["true"]}}
        rows_g = [(e, ev["u"] - 1) for e in grp for ev in evs_all[e]]
        dr = LM.one_step(model, cfg, R, enc, sub, rows_g, feat_of, arms, seed=20261126 + 1009 * k0)
        keys = cell_keys(L, so, rows_g, rterr)
        for a in arms:
            z = PR.cells(L, so, rows_g, dr[a], rterr)
            for i, (ee, u, kk) in enumerate(keys):
                ev = next(x for x in evs_all[ee] if x["u"] == u)
                if kk in ev["targets"]:
                    agree[a].append((int(ee), float(z["agree"][i]), z["context"][i]))
        rows5 = [(e, t - 1) for b, e in enumerate(grp) for t in sorted(t5s[b])]
        dr5 = LM.one_step(model, cfg, R, enc, sub, rows5, feat_of, ("predicted", "true"), seed=20261127 + 1009 * k0)
        for a in ("predicted", "true"):
            z = PR.cells(L, so, rows5, dr5[a], rterr)
            m5[a] += z["agree"][z["group"] != "first"].tolist()
        per_event += [dict(ev, e=int(e)) for e in grp for ev in evs_all[e]]
        print("episodes", k0 + len(grp), "of", len(eps_e), flush=True)
    res = {"update": update, "events": len(per_event)}
    for a in arms:
        v = np.array([x[1] for x in agree[a]]); sur = np.array([x[2] == "surprising" for x in agree[a]], bool)
        res[a] = {"cells": int(len(v)), "agree": float(v.mean()) if len(v) else None,
                  "agree_surprising": float(v[sur].mean()) if sur.any() else None}
    if res["repair"]["agree"] is not None:
        res["repair_minus_current"] = res["repair"]["agree"] - res["current"]["agree"]
        gap = res["clean"]["agree"] - res["none"]["agree"]
        res["repair_share_of_clean_gain"] = (res["repair"]["agree"] - res["none"]["agree"]) / gap if gap > 0 else None
        eps_ = np.unique([x[0] for x in agree["repair"]]); rng = np.random.default_rng(0)
        byep = {a: {e: [x[1] for x in agree[a] if x[0] == e] for e in eps_} for a in ("repair", "current")}
        bs = []
        for _ in range(300 if smoke else 1000):
            pick = rng.choice(eps_, len(eps_))
            bs.append(np.concatenate([byep["repair"][e] for e in pick]).mean() - np.concatenate([byep["current"][e] for e in pick]).mean())
        res["repair_minus_current_ci95"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    res["reading5"] = {a: {"cells": len(m5[a]), "agree_revisits": float(np.mean(m5[a])) if m5[a] else None} for a in m5}
    (HERE / f"reading3_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps({"summary": res, "events": per_event}, indent=1, default=float) + "\n")
    print(json.dumps(res, default=float), flush=True)


class DreamMemory(BT.BatchMemory):
    """Dream rows as lanes: each a copy of one episode's memory as of the start frame (the snapshot), with room for H
    dreamed frames, written to an overlay that only adds new places (decision 6's dream flow)."""

    def __init__(self, parts, src, b, snap, t, H, rows, repaired):
        super().__init__(parts, rows, t + 1 + H, mode="none", gate=False)
        n = t + 1
        X = snap["X"] if repaired else snap["Xcur"]
        self.X[:, :n] = X[:n]
        for name in ("off", "wm", "tok", "light", "g", "win", "unc", "var"):
            getattr(self, name)[:, :n] = getattr(src, name)[b, :n]
        self.trail[:, :n] = snap["trail"][:n]; self.cur[:] = snap["cur"]; self.trail[:, n:] = snap["cur"]
        self.n[:] = n

    def write_dream(self, pose, p, v, g, fix):
        calib = self.p["calib"]; i = int(self.n[0]); bi = torch.arange(self.B, device=DEV)
        world = calib.world.to(DEV); off = calib.offset.to(DEV)
        places = pose[:, None] + off[None]                               # [rows, 81, 2]
        near = torch.zeros(self.B, NCELL, dtype=torch.bool, device=DEV)
        for a in range(0, i, BT.CH):
            b_ = min(i, a + BT.CH)
            Pe = (self.X[:, a:b_, None] + self.off[:, a:b_]).reshape(self.B, -1, 2)
            wv = self.wm[:, a:b_].reshape(self.B, -1)
            hit = ((Pe[:, None] - places[:, :, None]).abs().max(-1).values < WIDTH) & wv[:, None]
            near |= hit.any(-1)
        self.X[:, i] = pose; self.off[:, i] = off; self.wm[:, i] = world[None] & ~near
        self.tok[:, i] = p.half(); self.light[:, i] = v; self.g[:, i] = g.half(); self.win[:, i] = M.window(fix)
        self.unc[:, i] = 0; self.dream[:, i] = True; self.trail[:, i] = self.cur
        self.n += 1


def reading4(update, smoke=False, H=16, MF=4):
    """Reading 4: 16-step self-fed dreams under the recorded commands (s2b.rollout's free mode), motion sampled from the
    motion model (one coherent motion per step). Free-running at all starts: empty memory against memory. Paired, at
    starts after an outage and its return: repaired against unrepaired memory with the same sampled motion (taken from
    the repaired rows), so only memory differs (Codex). Scored on world cells entering view (by the true pose) that
    the real prefix verified and that the simulator says are unchanged."""
    model, cfg, parts = load_world(update)
    R = S2.load_readers(model, cfg)
    L, so, eps, rows, sub, enc = LM.panel(model, cfg, smoke)
    den, P = model.denoiser, cfg["data"]["context_prefix"]
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    rng = np.random.default_rng(20261125)
    acc, motion = {}, []
    plan = {}
    for e in eps:
        T = int(L["ep_T"][e]); starts = list(range(P + 8, T - H))
        if not starts:
            continue
        free = sorted(rng.choice(starts, min(1 if smoke else 2, len(starts)), replace=False).tolist())
        evs = outage_events(L, so, e, T)
        paired = [t for t in ([min(ev["return"] + 4, T - H - 1) for ev in evs[:1]] if evs else []) if t >= P + 8]
        m = np.zeros(T + 1, bool)
        for ev in evs:
            m[ev["t0"]:ev["t0"] + ev["L"]] = True
        plan[e] = (free, paired, m)
    eps_d = list(plan)
    for k0 in range(0, len(eps_d), LANES):
        grp = eps_d[k0:k0 + LANES]
        ed = [episode_dict(L, so, e, enc[sub[e]]) for e in grp]
        wf = [{t + dt for t in plan[e][0] for dt in range(1 - P, 1)} for e in grp]
        wp = [{t + dt for t in plan[e][1] for dt in range(1 - P, 1)} for e in grp]
        mem_c, cc = capture(parts, ed, "full", None, want=wf, snap=[set(plan[e][0]) for e in grp])
        mem_o, co = capture(parts, ed, "full", [plan[e][2] for e in grp], want=wp, cur=wp, snap=[set(plan[e][1]) for e in grp])
        for b, e in enumerate(grp):
            en, i = enc[sub[e]], sub[e]
            T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)
            jobs = [(t, ("empty", "memory"), {"memory": (mem_c, cc[b]["snap"][t], True, cc[b]["feat"])}) for t in plan[e][0]]
            jobs += [(t, ("repaired", "unrepaired"), {"repaired": (mem_o, co[b]["snap"][t], True, co[b]["feat"]),
                                                       "unrepaired": (mem_o, co[b]["snap"][t], False, {**co[b]["feat"], **co[b]["cur"]})})
                     for t in plan[e][1]]
            for t, arms, mm in jobs:
                Bn = len(arms) * MF
                rep = lambda x: x.expand(Bn, *x.shape[1:]).clone() if x.shape[0] == 1 else x.repeat_interleave(Bn, 0)
                pp = M.Percept(*(en[k][:t + 1].to(DEV).float()[None] for k in ("p", "g", "v")), en["fix"][:t + 1].to(DEV)[None])
                with torch.no_grad(), TR.autocast(DEV, cfg):
                    reads, _, last = S.history(model, pp.p)
                    y, c = model.belief(model.belief.tokens(S.embed(model, pp, reads), en["reward"][:t + 1].to(DEV)[None],
                                        en["terminated"][:t + 1].to(DEV)[None], en["action"][:t + 1].to(DEV)[None]),
                                        model.belief.initial(1, DEV))
                carry = tuple(type(c[l])(*(rep(getattr(c[l], f.name)) for f in dataclasses.fields(c[l]))) for l in range(len(c)))
                bp, rs = rep(y[:, -1].float()), rep(last); rm = rs
                rin = S.window(model, enc, [(i, t)], P - 1)
                g = torch.Generator(device=DEV).manual_seed(20261125 + 7919 * int(e) + t)
                cache = den.new_cache()
                at = lambda key, dt: en[key][t + dt][None].to(DEV)
                ctx = lambda dt: torch.stack([(torch.zeros(MAP, KC_, FEAT, device=DEV) if a == "empty" else mm[a][3][t + dt].to(DEV).float())
                                              for a in arms for _ in range(MF)])
                with torch.no_grad(), TR.autocast(DEV, cfg):
                    for dt in range(1 - P, 1):
                        den.place_input = ctx(dt)[:, None]
                        cache = M.context_pass(den, tuple(rep(at(k, dt)).float()[:, None] for k in ("p", "g", "v")),
                                               rep(at("prev_action", dt))[:, None],
                                               S.cond(model.belief.memory(rep(at("bp", dt)).float())[:, None], rep(rin[:, P - 1 + dt])[:, None]),
                                               rep(at("fix", dt))[:, None], cache, cfg, g)
                mems = {a: DreamMemory(parts, mm[a][0], b, mm[a][1], t, H, MF, mm[a][2]) for a in arms if a != "empty"}
                pose = torch.stack([(torch.zeros(2, device=DEV) if a == "empty" else mm[a][1]["X" if mm[a][2] else "Xcur"][t].clone())
                                    for a in arms for _ in range(MF)])
                pose0 = pose.clone()
                last_p, last_v = en["p"][t][None].to(DEV).float().expand(Bn, -1, -1), en["v"][t][None].to(DEV).float().expand(Bn, -1)
                seen = {}
                for s_ in range(t + 1):
                    for kk in range(MAP):
                        cl = int(rterr[e][s_][kk // 9, kk % 9])
                        if cl >= 0:
                            seen[(int(pos[s_][0] + LM.KR[kk]), int(pos[s_][1] + LM.KC[kk]))] = (s_, cl)
                gm = torch.Generator(device=DEV).manual_seed(20261128 + 7919 * int(e) + t)
                for j in range(1, H + 1):
                    a_ = int(en["action"][t + j - 1])
                    pa = torch.full((Bn,), a_, device=DEV, dtype=torch.long)
                    fixj = rep(at("fix", j))
                    with torch.no_grad():
                        step = parts["motion"].sample(last_p, last_v, pa, gm)
                        if arms[0] == "repaired":
                            step = step[:MF].repeat(len(arms), 1)
                        pose = pose + step
                        fj = torch.zeros(Bn, MAP, KC_, FEAT, device=DEV)
                        for ai, a in enumerate(arms):
                            if a in mems:
                                dm = mems[a]
                                f, _ = dm.read(dm.n.clone(), pose[ai * MF:(ai + 1) * MF])
                                fj[ai * MF:(ai + 1) * MF] = f
                    with torch.no_grad(), TR.autocast(DEV, cfg):
                        memc = S.cond(model.belief.memory(bp)[:, None], rs[:, None])
                        den.place_input = fj[:, None]
                        out = M.sample(den, dims, pa[:, None], memc, fixj[:, None], cache, cfg, g)
                        out = tuple(x[:, 0].float() for x in out)
                        reads0, _, _ = S.history(model, out[0], state=rm)
                        h = model.outcome.features(bp, S.embed(model, M.Percept(*out, fixj), reads0)).float()
                        r = model.outcome.sample_reward(h, g)
                        done = torch.rand(Bn, device=DEV, generator=g) > torch.sigmoid(model.outcome.cont_logit(h, r).float())
                        reads1, _, next_rm = S.history(model, out[0], state=rm)
                        emb = S.embed(model, M.Percept(*out, fixj), reads1)
                        nxt = torch.full((Bn,), int(en["action"][t + j]) if j < H else 0, device=DEV, dtype=torch.long)
                        yy, carry = model.belief(model.belief.tokens(emb[:, None], r[:, None], done[:, None], nxt[:, None]), carry)
                        cache = M.context_pass(den, tuple(x[:, None] for x in out), pa[:, None], memc, fixj[:, None], cache, cfg, g)
                    bp, rs, rm = yy[:, 1].float(), next_rm, next_rm
                    with torch.no_grad():
                        for ai, a in enumerate(arms):
                            if a in mems:
                                sl = slice(ai * MF, (ai + 1) * MF)
                                mems[a].write_dream(pose[sl], out[0][sl], out[2][sl], out[1][sl], fixj[sl])
                    last_p, last_v = out[0], out[2]
                    te = B_.terrain(R, out[0][:, None], out[2][:, None])[:, 0]
                    right = (np.abs((pose - pose0).cpu().numpy() - (pos[t + j] - pos[t])) < WIDTH).all(1)
                    motion.append(float(right[MF:].mean()) if arms[0] == "empty" else float(right.mean()))
                    prev = {(int(pos[t + j - 1][0] + LM.KR[kk]), int(pos[t + j - 1][1] + LM.KC[kk])) for kk in range(MAP)}
                    for kk in range(MAP):
                        w = (int(pos[t + j][0] + LM.KR[kk]), int(pos[t + j][1] + LM.KC[kk]))
                        if w in prev or w not in seen:
                            continue
                        s0, cl = seen[w]
                        if int(rterr[e][t + j][kk // 9, kk % 9]) != cl:
                            continue
                        band = "le16" if t + j - s0 <= 16 else "17-64" if t + j - s0 <= 64 else "65+"
                        for bb in range(Bn):
                            key = (arms[bb // MF], band, "pose_right" if right[bb] else "pose_wrong")
                            a2, n2 = acc.get(key, (0, 0))
                            acc[key] = (a2 + int(te[bb, kk // 9, kk % 9] == cl), n2 + 1)
        print("episodes", k0 + len(grp), "of", len(eps_d), flush=True)
    res = {"update": update, "dream_motion_right": float(np.mean(motion)) if motion else None,
           "agreement": {f"{a}/{b}/{c}": {"agree": v[0] / v[1], "cells": v[1]} for (a, b, c), v in sorted(acc.items())}}
    (HERE / f"dream_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)


def smoke():
    """A 30-update fine-tune into run_smoke/, then every reading on 8 episodes."""
    global RUN
    import shutil
    RUN = HERE / "run_smoke"; RUN.mkdir(exist_ok=True)
    shutil.copy(HERE / "run" / "location.pt", RUN / "location.pt")
    LOC.RUN = RUN
    train(updates=30, saves=(30,))
    reading1(30, smoke=True); reading3(30, smoke=True); reading4(30, smoke=True)


if __name__ == "__main__":
    a = sys.argv
    {"train": lambda: train(), "reading1": lambda: reading1(int(a[2]), len(a) > 3),
     "reading3": lambda: reading3(int(a[2]), len(a) > 3), "reading4": lambda: reading4(int(a[2]), len(a) > 3),
     "smoke": smoke, "readings": lambda: (reading1(int(a[2])), reading3(int(a[2])), reading4(int(a[2])))}[a[1]]()
