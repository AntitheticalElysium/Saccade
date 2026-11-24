"""Decision 6's locked design, the world's side: stage B (the read, trained as stage 1 was) and the readings (PLAN.md).

The read: each map cell queries the memory at its place before the frame. Candidates are the entries of the current
trail nearest that place (100, by displacement), ranked by code similarity, at most 5 per place, newest first, 25 in
all (the diversified shortlist), plus a learned null slot. Keys and values come from an MLP of each candidate's whole
entry and its relative-code features. Cells the calibration gives no world projection get no candidates.

    MAMBA_DETERMINISTIC=1 LOC_DEVICE=cuda JAX_PLATFORMS=cpu .venv/bin/python -m \
        artifacts.experiments.20261124_learned_location.world train | readings 2000 | smoke
"""
import importlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOC = importlib.import_module("artifacts.experiments.20261124_learned_location.location")
LM, PR, S2, D = LOC.LM, LOC.PR, LOC.S2, LOC.D
M, TR, S, PM, B_ = LM.M, LM.TR, LM.S, LM.PM, LM.B
HERE, RUN, DEV = LOC.HERE, LOC.RUN, LOC.DEV
MAP, NCELL, WIDTH = LM.MAP, LOC.NCELL, LOC.WIDTH
KC_ = 25                                   # candidates per cell (the agreed 5 x 5 reach)
NEAR = 100                                 # nearest entries considered before ranking
PER_PLACE = 5
FEAT = 5 + 16 + 1 + 8 + 64 + 1 + 9 + 64 + 1 + 1 + 1   # age, token, write conf, light, fine, fine flag, crop, rel, sim, unc, dream


class MemEstimator(LOC.Estimator):
    """The stage-A estimator, also keeping what the read needs per frame: tokens, lighting, the glimpse and its window,
    and the position uncertainty at write time."""

    def __init__(self, parts, T, mode="full", gate=True):
        super().__init__(parts, T, mode, gate)
        self.tok = torch.zeros(T + 1, NCELL, 16, device=DEV, dtype=torch.float16)
        self.light = torch.zeros(T + 1, 8, device=DEV)
        self.g = torch.zeros(T + 1, 9, 64, device=DEV, dtype=torch.float16)
        self.win = torch.full((T + 1, 9), -1, dtype=torch.long, device=DEV)
        self.unc = torch.zeros(T + 1, device=DEV)
        self.repairs = []                                                # (t, anchor, per-frame additions [n, 2])
        self.dream = torch.zeros(T + 1, dtype=torch.bool, device=DEV)

    def write_extra(self, t, p, v, g, fix):
        self.tok[t] = p.to(DEV).half(); self.light[t] = v.to(DEV); self.g[t] = g.to(DEV).half()
        self.win[t] = M.window(fix.to(DEV)); self.unc[t] = math.sqrt(self.var_since)

    def apply(self, t, c, trail):
        if self.mode == "full" and t - self.anchor > 1:
            v = self.var[self.anchor + 1:t + 1]
            share = torch.cumsum(v, 0) / v.sum().clamp(min=1e-12)
            self.repairs.append((t, self.anchor, (share[:-1, None] * c).clone()))
        return super().apply(t, c, trail)


def read_feats(est, t, xq, parts, X=None, sel_log=None):
    """The read's input for frame t [63, KC_, FEAT]: candidates among entries written before t (current trail), at
    places nearest xq + each map cell's offset."""
    out = torch.zeros(MAP, KC_, FEAT, device=DEV)
    if t == 0:
        return out
    X = est.X if X is None else X
    calib, code = parts["calib"], parts["code"]
    wm = est.wm[:t] & (est.trail[:t, None] == est.cur)
    fi, ci = torch.nonzero(wm, as_tuple=True)
    if not len(fi):
        return out
    Pe = X[fi] + est.off[fi, ci]
    off = calib.offset.to(DEV)[:MAP]; world = calib.world.to(DEV)[:MAP]
    q = xq[None] + off
    k1 = min(NEAR, len(fi))
    near = (Pe[None] - q[:, None]).abs().max(-1).values.topk(k1, largest=False).indices      # [63, k1]
    P_c, fr, ce = Pe[near], fi[near], ci[near]
    sim = code.sim(q[:, None], P_c)
    b = torch.round(P_c / WIDTH).long(); bk = b[..., 0] * 100003 + b[..., 1]
    rank = ((bk[:, :, None] == bk[:, None, :]) & (fr[:, None, :] > fr[:, :, None])).sum(-1)   # newer entries, same place
    score = torch.where(rank < PER_PLACE, sim + 1e-7 * fr.float(), torch.full_like(sim, -1e9))
    k2 = min(KC_, k1)
    sel = score.topk(k2, dim=1).indices
    ok = score.gather(1, sel) > -1e8
    f_, c_, P_s, s_ = fr.gather(1, sel), ce.gather(1, sel), P_c.gather(1, sel[..., None].expand(-1, -1, 2)), sim.gather(1, sel)
    age = (t - f_).float()
    pres = ok & world[:, None]
    agef = PR.age_features(age, pres)                                                      # [63, k2, 5]
    tok = est.tok[f_, c_].float()
    wconf = (1 / (1 + est.unc[f_]))[..., None]
    light = est.light[f_]
    w9 = est.win[f_]                                                                       # [63, k2, 9]
    hitj = (w9 == c_[..., None])
    fpres = hitj.any(-1) & pres
    j = hitj.float().argmax(-1)
    fine = est.g[f_, j].float() * fpres[..., None]
    crop = torch.nn.functional.one_hot(j, 9).float() * fpres[..., None]
    rel = code.rel(q[:, None], P_s)
    unc = torch.full((MAP, k2, 1), float(est.unc[t - 1]) if t > 0 else 0.0, device=DEV)
    dre = est.dream[f_].float()[..., None]
    f = torch.cat([agef, tok, wconf, light, fine, fpres[..., None].float(), crop, rel, s_[..., None], unc, dre], -1)
    out[:, :k2] = f * pres[..., None]
    if sel_log is not None:                                              # each cell's best candidate (frame, cell, ok)
        sel_log.append((f_[:, 0].cpu(), c_[:, 0].cpu(), pres[:, 0].cpu()))
    return out


def install(den):
    """The world's read: each map cell's input token queries its candidates and a null slot (4 heads of 16); keys and
    values from an MLP of the whole candidate; output zero-initialised, added to the cell's token (located_memory's)."""
    dev = next(den.parameters()).device
    den.mem_kv = nn.Sequential(nn.Linear(FEAT, 128), nn.GELU(), nn.Linear(128, 128)).to(dev)
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
        q = den.mem_q(cell).unflatten(-1, (4, 16))
        kv = den.mem_kv(pin.to(cell.dtype))
        kv = torch.cat([kv, den.mem_null.to(kv.dtype).expand(*kv.shape[:-2], 1, 128)], -2)
        k, v = kv.unflatten(-1, (2, 4, 16)).unbind(-3)
        pres = torch.cat([pin[..., 0] > 0, torch.ones_like(pin[..., :1, 0], dtype=torch.bool)], -1)   # absent slots masked
        att = torch.einsum("btchd,btcnhd->btchn", q, k) / 4.0
        att = att.masked_fill(~pres[..., None, :], -1e4)
        o = torch.einsum("btchn,btcnhd->btchd", att.softmax(-1), v).flatten(-2)
        return torch.cat([cell + den.mem_out(o).to(cell.dtype), tok[..., MAP:, :]], -2), ang
    den.embed = with_memory
    loss = M.shortcut_loss

    def shortcut_loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
        d.place_sub = d.place_input[bootstrap]
        return loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels, noise)
    M.shortcut_loss = shortcut_loss


def mem_params(den):
    return [p for m in (den.mem_kv, den.mem_q, den.mem_out) for p in m.parameters()] + [den.mem_null]


# ---------------------------------------------------------------- stage B: the read, trained on P1.5's streams

class StreamLocation:
    """Per stream: the estimator of its current episode, with that episode's outages (the training schedule, seeded by
    episode), the motion sensor from S0 positions, and the read features of the last frames (the next chunk's prefix
    repeats them)."""

    def __init__(self, parts, n, eps, spos, start):
        self.parts, self.eps, self.spos, self.start = parts, eps, spos, start
        self.est, self.ep, self.last, self.cache, self.mask = [None] * n, [-1] * n, [-1] * n, [dict() for _ in range(n)], [None] * n
        self.applied = 0; self.withheld = 0; self.steps = 0

    def feats(self, chunk, Pc, calib_obs):
        ei, ts, valid = chunk["ep_index"], chunk["ts"], chunk["valid"]
        n, L = ei.shape
        out = torch.zeros(n, L, MAP, KC_, FEAT, device=DEV, dtype=torch.float16)
        pa = torch.as_tensor(chunk["prev_action"])
        for b in range(n):
            for j in range(L):
                if not valid[b, j] or ei[b, j] < 0:
                    continue
                e, t = int(ei[b, j]), int(ts[b, j])
                if e != self.ep[b] or t == 0 and self.last[b] >= 0:
                    T = self.eps[e].T
                    self.est[b] = MemEstimator(self.parts, T); self.ep[b], self.last[b], self.cache[b] = e, -1, {}
                    self.mask[b] = LOC.outage_mask(T, np.random.default_rng([20261124, e]))
                if t <= self.last[b]:
                    out[b, j] = self.cache[b][t]; continue
                est = self.est[b]
                p, v, g, fx = Pc.p[b, j].float(), Pc.v[b, j].float(), Pc.g[b, j].float(), torch.as_tensor(chunk["fixation"][b, j])
                pos = self.spos[self.start[e] + t]
                with torch.no_grad():
                    if t == 0:
                        x, var = torch.zeros(2, device=DEV), LOC.SENSED_VAR
                    else:
                        d = torch.as_tensor(pos - self.spos[self.start[e] + t - 1], dtype=torch.float32)
                        if float(d.abs().sum()) > 0:
                            calib_obs(est.tok[t - 1].float(), p, d)
                        w = bool(self.mask[b][t]); self.withheld += int(w)
                        x, var = est.predict(t, est.tok[t - 1].float(), est.light[t - 1], pa[b, j], None if w else d)
                    f = read_feats(est, t, x, self.parts)
                    nlog = len(est.log)
                    est.step(t, p, v, x, var, not bool(self.mask[b][t]) if t > 0 else True)
                    est.write_extra(t, p, v, g, fx)
                    self.applied += int(len(est.log) > nlog and est.log[-1][4]); self.steps += 1
                out[b, j] = f.half(); self.cache[b][t] = out[b, j]
                for old in [k for k in self.cache[b] if k < t - 16]:
                    del self.cache[b][old]
                self.last[b] = t
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
    install(den)
    params = list(den.parameters())
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
    sl = StreamLocation(parts, d["streams"], split["train"], spos, start)
    calib = parts["calib"]
    base, micro = ck["state"]["update"], d["accumulate_at_64"]
    log, t0, losses = [], time.time(), []
    for u in range(1, updates + 1):
        chunk = streams.next(64)
        fr = torch.as_tensor(chunk["frames"], device=DEV)
        fx = torch.as_tensor(chunk["fixation"], device=DEV)
        with torch.no_grad(), TR.autocast(DEV, cfg):
            Pc = model.perception(fr, fx)
        feats = sl.feats(chunk, Pc, calib.observe)
        if u % 100 == 0:                                                 # the running calibration (option 3)
            calib.fit()
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
                        "corrections_applied": sl.applied, "steps": sl.steps, "withheld_share": sl.withheld / max(sl.steps, 1),
                        "calibration": calib.score(), "sec_per_update": (time.time() - t0) / u})
            print(json.dumps(log[-1]), flush=True)
            (RUN / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
        if u in saves:
            torch.save({"read": {k: v for k, v in den.state_dict().items()}, "update": u, "calib": calib.state()}, RUN / f"denoiser_{u}.pt")


# ---------------------------------------------------------------- readings

FINE = slice(30, 94); FPRES = 94                                           # the fine token and its flag in FEAT
SHIFT = torch.tensor([float(LM.SHIFT[0]), float(LM.SHIFT[1])])             # the swap's wrong place (3 rows)


def load_world(update):
    model, cfg = S2.load(); S2.EV.guard()
    for prm in model.parameters():
        prm.requires_grad_(False)
    install(model.denoiser)
    parts = LOC.load()
    if update:
        st = torch.load(RUN / f"denoiser_{update}.pt", map_location=DEV, weights_only=False)
        model.denoiser.load_state_dict(st["read"]); parts["calib"].load(st["calib"])   # the calibration as stage B left it
    model.eval()
    return model, cfg, parts


def sensed_np(pos):
    return torch.cat([torch.zeros(1, 2), torch.as_tensor(pos[1:] - pos[:-1], dtype=torch.float32)])


def panel_run(parts, en, pos, mode="full", mask=None, want=(), shifted=(), cur=(), true_step=(), snap=()):
    """One panel episode through the estimator. At frames in `want`: the read's features; `cur`: the current-only arm's
    (the same codes, earlier repairs undone); `shifted`: the read at the place 3 rows away, with its best candidates
    logged (the swap's donors); `true_step`: the read placed by the true motion instead (reading 5). Also each frame's
    best candidate under the correct read, for the per-read check."""
    T = len(en["p"]) - 1
    est = MemEstimator(parts, T, mode, True)
    d = sensed_np(pos)
    out = {"feat": {}, "cur": {}, "wrong": {}, "donor": {}, "top": {}, "true": {}, "snap": {}}
    pa = en["prev_action"]
    with torch.no_grad():
        for t in range(T + 1):
            if t == 0:
                x, var = torch.zeros(2, device=DEV), LOC.SENSED_VAR
            else:
                w = mask is not None and bool(mask[t])
                x, var = est.predict(t, est.tok[t - 1].float(), est.light[t - 1], torch.as_tensor(int(pa[t])), None if w else d[t])
            if t in want:
                lg = []
                out["feat"][t] = read_feats(est, t, x, parts, sel_log=lg).half().cpu(); out["top"][t] = lg[0]
            if t in cur:
                Xc = est.X.clone()
                for tr, an, adds in est.repairs:
                    Xc[an + 1:tr] -= adds
                out["cur"][t] = read_feats(est, t, x, parts, X=Xc).half().cpu()
            if t in shifted:
                lg = []
                out["wrong"][t] = read_feats(est, t, x + SHIFT.to(DEV), parts, sel_log=lg).half().cpu(); out["donor"][t] = lg[0]
            if t in true_step:
                out["true"][t] = read_feats(est, t, est.X[t - 1] + d[t].to(DEV), parts).half().cpu()
            est.step(t, en["p"][t].float(), en["v"][t].float(), x, var, not (mask is not None and bool(mask[t])))
            est.write_extra(t, en["p"][t].float(), en["v"][t].float(), en["g"][t].float(), en["fix"][t])
            if t in snap:                                                # the memory as of frame t (dreams start here)
                Xc = est.X.clone()
                for tr, an, adds in est.repairs:
                    Xc[an + 1:tr] -= adds
                out["snap"][t] = {"X": est.X.clone(), "Xcur": Xc, "trail": est.trail.clone(), "cur": est.cur}
    return est, out


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
        f = cur["feat"][t].to(DEV).float()
        if v == "correct" or not drawn:
            return f
        m = new[(e, t - 1)][:, None, None]
        if v == "empty":
            return torch.where(m, torch.zeros_like(f), f)
        w = cur["wrong"][t].to(DEV).float()
        if v == "wrong":
            return torch.where(m, w, f)
        alt = f.clone()                                                  # fine swap: only the fine token moves
        both = (f[..., FPRES] > 0) & (w[..., FPRES] > 0)
        alt[..., FINE] = torch.where(both[..., None], w[..., FINE], f[..., FINE])
        return torch.where(m, alt, f)
    variants = ("correct", "empty", "wrong", "fine_swap") if update else ("correct",)
    drawn, donors, tops = {v: {} for v in variants}, {}, {}
    for k, e in enumerate(eps):
        rows_e = [r for r in rows if r[0] == e]
        if not rows_e:
            continue
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)
        want = {t + dt for _, t in rows_e for dt in range(1 - P, 2) if 0 <= t + dt <= T}
        _, out = panel_run(parts, enc[sub[e]], pos, want=want, shifted={t + 1 for _, t in rows_e})
        cur.clear(); cur.update(out)
        for t, lg in out["donor"].items():
            donors[(e, t)] = lg
        for t, lg in out["top"].items():
            tops[(e, t)] = lg
        dr = LM.one_step(model, cfg, R, enc, sub, rows_e, feat_of, variants, seed=20261123 + 1009 * k)
        for v in variants:
            drawn[v].update(dr[v])
        print("episode", int(e), flush=True)
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
        # reading 6: the class of the donor actually injected (the wrong read's best candidate)
        wcls = np.full(len(keys), -1)
        right_top = np.zeros(len(keys), bool)
        for i, (e, t, kk) in enumerate(keys):
            f_, c_, ok = donors[(e, t)]
            if bool(ok[kk]):
                fr, ce = int(f_[kk]), int(c_[kk])
                if ce < MAP:
                    wcls[i] = int(rterr[e][fr][ce // 9, ce % 9])
            f2, c2, ok2 = tops[(e, t)]
            right_top[i] = bool(ok2[kk]) and int(c2[kk]) < MAP and true_cell(L, so, e, int(f2[kk]), int(c2[kk])) == true_cell(L, so, e, t, kk)
        base = (c["context"] == "surprising") & (c["group"] != "first") & (wcls >= 0) & (wcls != c["cls"])
        takes = {v: float((zs[v]["drawn"][base] == wcls[base][:, None]).mean()) for v in ("correct", "empty", "wrong")}
        res["identity"] = {"cells": int(base.sum()), "takes_donor_class": takes, "wrong_minus_empty": takes["wrong"] - takes["empty"]}
        rev = (c["context"] == "surprising") & (c["group"] != "first")
        gain = zs["correct"]["agree"] - zs["empty"]["agree"]
        res["per_read"] = {"best_candidate_is_the_cell": {"share": float(right_top[rev].mean()) if rev.any() else None,
                                                          "gain_when_right": float(gain[rev & right_top].mean()) if (rev & right_top).any() else None,
                                                          "gain_when_wrong": float(gain[rev & ~right_top].mean()) if (rev & ~right_top).any() else None}}
    (HERE / f"eval_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in res if k != "variants"}, default=float), flush=True)


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


def reading3(update, smoke=False):
    """Reading 3 (the causal chain), 3b (wrong corrections) and 5 (the one-step motion cost)."""
    model, cfg, parts = load_world(update)
    R = S2.load_readers(model, cfg)
    L, so, eps, rows, sub, enc = LM.panel(model, cfg, smoke)
    P = cfg["data"]["context_prefix"]
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    arms = ("none", "current", "repair", "clean", "empty")
    agree = {a: [] for a in arms}; per_event = []
    m5 = {"predicted": [], "true": []}
    cur = {}

    def feat_of(v, e, t, drawn):
        return cur[v][t].to(DEV).float()
    for k, e in enumerate(eps):
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)
        evs = outage_events(L, so, e, T)
        if not evs:
            continue
        mask = np.zeros(T + 1, bool)
        for ev in evs:
            mask[ev["t0"]:ev["t0"] + ev["L"]] = True
        want = {ev["u"] + dt for ev in evs for dt in range(-P, 1)}         # one_step reads frames u - P .. u
        t5 = {min(ev["t0"] + 4, ev["t0"] + ev["L"] - 1) for ev in evs}
        want5 = {t + dt for t in t5 for dt in range(-P, 1)}
        est_f, out_f = panel_run(parts, enc[sub[e]], pos, "full", mask, want=want | want5, cur=want, true_step=t5)
        _, out_n = panel_run(parts, enc[sub[e]], pos, "none", mask, want=want)
        _, out_c = panel_run(parts, enc[sub[e]], pos, "full", None, want=want)
        cur.clear()
        cur["repair"], cur["current"], cur["none"], cur["clean"] = out_f["feat"], {**out_f["feat"], **out_f["cur"]}, out_n["feat"], out_c["feat"]
        cur["empty"] = {t: torch.zeros_like(f) for t, f in out_f["feat"].items()}
        rows_e = [(e, ev["u"] - 1) for ev in evs]
        dr = LM.one_step(model, cfg, R, enc, sub, rows_e, feat_of, arms, seed=20261126 + 1009 * k)
        keys = cell_keys(L, so, rows_e, rterr)
        for a in arms:
            z = PR.cells(L, so, rows_e, dr[a], rterr)
            for i, (ee, u, kk) in enumerate(keys):
                ev = next(x for x in evs if x["u"] == u)
                if kk in ev["targets"]:
                    agree[a].append((int(ee), float(z["agree"][i]), z["context"][i]))
        # reading 5: one step into an outage, the read placed by the predicted against the true motion
        cur2 = {"predicted": out_f["feat"], "true": {**out_f["feat"], **out_f["true"]}}
        cur.clear(); cur.update(cur2)
        rows5 = [(e, t - 1) for t in sorted(t5)]
        dr5 = LM.one_step(model, cfg, R, enc, sub, rows5, feat_of, ("predicted", "true"), seed=20261127 + 1009 * k)
        for a in ("predicted", "true"):
            z = PR.cells(L, so, rows5, dr5[a], rterr)
            sel = z["group"] != "first"
            m5[a] += z["agree"][sel].tolist()
        per_event += [dict(ev, e=int(e)) for ev in evs]
        print("episode", int(e), "events", len(evs), flush=True)
    res = {"update": update, "events": len(per_event)}
    for a in arms:
        v = np.array([x[1] for x in agree[a]]); sur = np.array([x[2] == "surprising" for x in agree[a]])
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
            r_ = np.concatenate([byep["repair"][e] for e in pick]); c_ = np.concatenate([byep["current"][e] for e in pick])
            bs.append(r_.mean() - c_.mean())
        res["repair_minus_current_ci95"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    res["reading5"] = {a: {"cells": len(m5[a]), "agree_revisits": float(np.mean(m5[a])) if m5[a] else None} for a in m5}
    (HERE / f"reading3_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps({"summary": res, "events": per_event}, indent=1, default=float) + "\n")
    print(json.dumps(res, default=float), flush=True)


class DreamMem:
    """A copy of an episode's memory as of frame t, with room for H dreamed frames: an overlay that only adds new
    places, never overwriting a real entry (decision 6's dream flow)."""

    def __init__(self, est, t, H, snap, repaired=True):
        n = t + 1
        self.X = torch.cat([(snap["X"] if repaired else snap["Xcur"])[:n], torch.zeros(H, 2, device=DEV)])
        for name in ("off", "wm", "tok", "light", "g", "win", "unc", "dream", "var"):
            a = getattr(est, name)
            setattr(self, name, torch.cat([a[:n], torch.zeros((H,) + a.shape[1:], dtype=a.dtype, device=DEV)]))
        self.cur = snap["cur"]
        self.trail = torch.cat([snap["trail"][:n], torch.full((H,), self.cur, dtype=torch.long, device=DEV)])
        self.n = n

    def write(self, pose, p, v, g, fix, parts):
        calib = parts["calib"]; i = self.n
        world = calib.world.to(DEV); off = calib.offset.to(DEV)
        places = pose[None] + off
        wm = self.wm[:i]
        fi, ci = torch.nonzero(wm, as_tuple=True)
        new = world.clone()
        if len(fi):
            Pe = self.X[fi] + self.off[fi, ci]
            new &= ~((Pe[None] - places[:, None]).abs().max(-1).values < WIDTH).any(1)
        self.X[i] = pose; self.off[i] = off; self.wm[i] = new; self.tok[i] = p.half(); self.light[i] = v
        self.g[i] = g.half(); self.win[i] = M.window(fix); self.unc[i] = 0; self.dream[i] = True
        self.n += 1


def reading4(update, smoke=False, H=16, MF=4):
    """Reading 4: 16-step self-fed dreams under the recorded commands (s2b.rollout's free mode), motion sampled from the
    motion model (one coherent motion per step). Free-running arms at all starts: empty memory, memory. Paired arms at
    starts after an outage and its return: repaired against unrepaired memory, with the same sampled motion (taken
    from the repaired arm) so only memory differs (Codex). Scored on world cells entering view (by the true pose) that
    the real prefix verified and that the simulator says are unchanged."""
    model, cfg, parts = load_world(update)
    R = S2.load_readers(model, cfg)
    L, so, eps, rows, sub, enc = LM.panel(model, cfg, smoke)
    den, P = model.denoiser, cfg["data"]["context_prefix"]
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    rng = np.random.default_rng(20261125)
    acc, motion = {}, []
    for e in eps:
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)
        en, i = enc[sub[e]], sub[e]
        starts = list(range(P + 8, T - H))
        if not starts:
            continue
        free = sorted(rng.choice(starts, min(1 if smoke else 2, len(starts)), replace=False).tolist())
        evs = outage_events(L, so, e, T)
        paired = [min(ev["return"] + 4, T - H - 1) for ev in evs[:1]] if evs else []
        paired = [t for t in paired if t >= P + 8]
        mask = np.zeros(T + 1, bool)
        for ev in evs:
            mask[ev["t0"]:ev["t0"] + ev["L"]] = True
        want = {t + dt for t in free for dt in range(1 - P, 1)}
        est_c, out_c = panel_run(parts, en, pos, "full", None, want=want, snap=set(free))
        jobs = [(t, ("empty", "memory"), {"memory": (est_c, out_c["snap"][t], True, out_c["feat"])}) for t in free]
        if paired:
            want_p = {t + dt for t in paired for dt in range(1 - P, 1)}
            est_o, out_o = panel_run(parts, en, pos, "full", mask, want=want_p, cur=want_p, snap=set(paired))
            jobs += [(t, ("repaired", "unrepaired"), {"repaired": (est_o, out_o["snap"][t], True, out_o["feat"]),
                                                       "unrepaired": (est_o, out_o["snap"][t], False, {**out_o["feat"], **out_o["cur"]})})
                     for t in paired]
        for t, arms, mem in jobs:
            Bn = len(arms) * MF
            arm_of = torch.arange(Bn, device=DEV) // MF
            rep = lambda x: x.expand(Bn, *x.shape[1:]).clone() if x.shape[0] == 1 else x.repeat_interleave(Bn, 0)
            pp = M.Percept(*(en[k][:t + 1].to(DEV).float()[None] for k in ("p", "g", "v")), en["fix"][:t + 1].to(DEV)[None])
            with torch.no_grad(), TR.autocast(DEV, cfg):
                reads, _, last = S.history(model, pp.p)
                y, c = model.belief(model.belief.tokens(S.embed(model, pp, reads), en["reward"][:t + 1].to(DEV)[None],
                                    en["terminated"][:t + 1].to(DEV)[None], en["action"][:t + 1].to(DEV)[None]),
                                    model.belief.initial(1, DEV))
            carry = tuple(type(c[l])(*(rep(getattr(c[l], f.name)) for f in __import__("dataclasses").fields(c[l]))) for l in range(len(c)))
            bp, rs = rep(y[:, -1].float()), rep(last); rm = rs
            rin = S.window(model, enc, [(i, t)], P - 1)
            g = torch.Generator(device=DEV).manual_seed(20261125 + 7919 * int(e) + t)
            cache = den.new_cache()
            at = lambda key, dt: en[key][t + dt][None].to(DEV)
            ctx = lambda dt: torch.stack([(torch.zeros(MAP, KC_, FEAT, device=DEV) if a == "empty" else mem[a][3][t + dt].to(DEV).float())
                                          for a in arms for _ in range(MF)])
            with torch.no_grad(), TR.autocast(DEV, cfg):
                for dt in range(1 - P, 1):
                    den.place_input = ctx(dt)[:, None]
                    cache = M.context_pass(den, tuple(rep(at(k, dt)).float()[:, None] for k in ("p", "g", "v")),
                                           rep(at("prev_action", dt))[:, None],
                                           S.cond(model.belief.memory(rep(at("bp", dt)).float())[:, None], rep(rin[:, P - 1 + dt])[:, None]),
                                           rep(at("fix", dt))[:, None], cache, cfg, g)
            dms = [None if a == "empty" else DreamMem(mem[a][0], t, H, mem[a][1], mem[a][2]) for a in arms for _ in range(MF)]
            pose0 = torch.stack([torch.zeros(2, device=DEV) if dm is None else dm.X[t].clone() for dm in dms])
            pose = pose0.clone()
            last_p, last_v = en["p"][t][None].to(DEV).float().expand(Bn, -1, -1), en["v"][t][None].to(DEV).float().expand(Bn, -1)
            seen = {}
            for s_ in range(t + 1):
                for k in range(MAP):
                    cl = int(rterr[e][s_][k // 9, k % 9])
                    if cl >= 0:
                        seen[(int(pos[s_][0] + LM.KR[k]), int(pos[s_][1] + LM.KC[k]))] = (s_, cl)
            gm = torch.Generator(device=DEV).manual_seed(20261128 + 7919 * int(e) + t)
            for j in range(1, H + 1):
                a = int(en["action"][t + j - 1])
                pa = torch.full((Bn,), a, device=DEV, dtype=torch.long)
                fixj = rep(at("fix", j))
                with torch.no_grad():
                    step = parts["motion"].sample(last_p, last_v, pa, gm)
                    if arms[0] == "repaired":                                # the same motion for both memory arms
                        step = step[:MF].repeat(len(arms), 1)
                    pose = pose + step
                    fj = torch.stack([torch.zeros(MAP, KC_, FEAT, device=DEV) if dms[b] is None else
                                      read_feats(dms[b], dms[b].n, pose[b], parts) for b in range(Bn)])
                with torch.no_grad(), TR.autocast(DEV, cfg):
                    memc = S.cond(model.belief.memory(bp)[:, None], rs[:, None])
                    den.place_input = fj[:, None]
                    parts_ = M.sample(den, dims, pa[:, None], memc, fixj[:, None], cache, cfg, g)
                    parts_ = tuple(x[:, 0].float() for x in parts_)
                    reads0, _, _ = S.history(model, parts_[0], state=rm)
                    h = model.outcome.features(bp, S.embed(model, M.Percept(*parts_, fixj), reads0)).float()
                    r = model.outcome.sample_reward(h, g)
                    done = torch.rand(Bn, device=DEV, generator=g) > torch.sigmoid(model.outcome.cont_logit(h, r).float())
                    reads1, _, next_rm = S.history(model, parts_[0], state=rm)
                    emb = S.embed(model, M.Percept(*parts_, fixj), reads1)
                    nxt = torch.full((Bn,), int(en["action"][t + j]) if j < H else 0, device=DEV, dtype=torch.long)
                    yy, carry = model.belief(model.belief.tokens(emb[:, None], r[:, None], done[:, None], nxt[:, None]), carry)
                    cache = M.context_pass(den, tuple(x[:, None] for x in parts_), pa[:, None], memc, fixj[:, None], cache, cfg, g)
                bp, rs, rm = yy[:, 1].float(), next_rm, next_rm
                with torch.no_grad():
                    for b in range(Bn):
                        if dms[b] is not None:
                            dms[b].write(pose[b], parts_[0][b], parts_[2][b], parts_[1][b], fixj[b], parts)
                last_p, last_v = parts_[0], parts_[2]
                te = B_.terrain(R, parts_[0][:, None], parts_[2][:, None])[:, 0]
                right = (np.abs((pose - pose0).cpu().numpy() - (pos[t + j] - pos[t])) < WIDTH).all(1)
                motion.append(float(right[MF:].mean()) if arms[0] == "empty" else float(right.mean()))
                prev = {(int(pos[t + j - 1][0] + LM.KR[k]), int(pos[t + j - 1][1] + LM.KC[k])) for k in range(MAP)}
                for k in range(MAP):
                    w = (int(pos[t + j][0] + LM.KR[k]), int(pos[t + j][1] + LM.KC[k]))
                    if w in prev or w not in seen:
                        continue
                    s0, cl = seen[w]
                    if int(rterr[e][t + j][k // 9, k % 9]) != cl:
                        continue
                    band = "le16" if t + j - s0 <= 16 else "17-64" if t + j - s0 <= 64 else "65+"
                    for b in range(Bn):
                        stratum = "pose_right" if right[b] else "pose_wrong"
                        key = (arms[b // MF], band, stratum)
                        a_, n_ = acc.get(key, (0, 0))
                        acc[key] = (a_ + int(te[b, k // 9, k % 9] == cl), n_ + 1)
        print("episode", int(e), "done", flush=True)
    res = {"update": update, "dream_motion_right": float(np.mean(motion)) if motion else None,
           "agreement": {f"{a}/{b}/{c}": {"agree": v[0] / v[1], "cells": v[1]} for (a, b, c), v in sorted(acc.items())}}
    (HERE / f"dream_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)


def smoke():
    """A 30-update fine-tune into run_smoke/ (the read only), then every reading on 8 episodes."""
    global RUN
    RUN = HERE / "run_smoke"; RUN.mkdir(exist_ok=True)
    import shutil
    shutil.copy(HERE / "run" / "location.pt", RUN / "location.pt")
    LOC.RUN = RUN
    train(updates=30, saves=(30,))
    reading1(30, smoke=True); reading3(30, smoke=True); reading4(30, smoke=True)


if __name__ == "__main__":
    a = sys.argv
    {"train": lambda: train(), "reading1": lambda: reading1(int(a[2]), len(a) > 3),
     "reading3": lambda: reading3(int(a[2]), len(a) > 3), "reading4": lambda: reading4(int(a[2]), len(a) > 3),
     "smoke": smoke, "readings": lambda: (reading1(int(a[2])), reading3(int(a[2])), reading4(int(a[2])))}[a[1]]()
