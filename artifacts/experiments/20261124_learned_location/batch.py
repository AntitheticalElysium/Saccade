"""The location stage's estimator and the world's read, batched over lanes (streams in training, episodes in the
readings) on the GPU. The algorithm is location.Estimator's, step for step; only the computation is batched. Exact
ties are broken toward the newer entry. selftest() compares it with location.Estimator on panel episodes.

    LOC_DEVICE=cuda JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261124_learned_location.batch selftest
"""
import importlib, json, math, sys, time

import numpy as np
import torch
import torch.nn.functional as F

LOC = importlib.import_module("artifacts.experiments.20261124_learned_location.location")
LM, M = LOC.LM, LOC.LM.M
DEV, NCELL, WIDTH, EXCL, TOPR, ACC = LOC.DEV, LOC.NCELL, LOC.WIDTH, LOC.EXCL, LOC.TOPR, LOC.ACC
MAP = LM.MAP
KC_, NEAR, PER_PLACE, KREC = 25, 100, 5, 32
FEAT = 5 + 16 + 1 + 8 + 64 + 1 + 9 + 2 + 1 + 1 + 1      # age, token, write conf, light, fine, flag, crop, rel d, sim, unc, dream
REL = slice(104, 106)                                   # the relative displacement (the read expands it to phases)


CH = 512                                                # frames per chunk of the similarity searches (memory-bounded)


def chunked_topk(score, Fm, k, largest):
    """Top k over all entries of frames [0, Fm), computed chunk by chunk with a running merge. score(a, b) -> values
    [B, Q, (b - a) * 81] for frames [a, b). Returns values and global entry indices (frame * 81 + cell)."""
    bv = bi = None
    for a in range(0, Fm, CH):
        b = min(Fm, a + CH)
        v = score(a, b)
        kk = min(k, v.shape[-1])
        cv, ci = v.topk(kk, dim=-1, largest=largest)
        ci = ci + a * NCELL
        if bv is None:
            bv, bi = cv, ci
        else:
            mv, mi = torch.cat([bv, cv], -1), torch.cat([bi, ci], -1)
            kk = min(k, mv.shape[-1])
            bv, j = mv.topk(kk, dim=-1, largest=largest)
            bi = torch.gather(mi, -1, j)
    return bv, bi


class BatchMemory:
    """B lanes, capacity C frames each. mode: 'full', 'current' or 'none' (as location.Estimator)."""

    def __init__(self, parts, B, C, mode="full", gate=True):
        self.p, self.B, self.C, self.mode, self.use_gate = parts, B, C, mode, gate
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=DEV, dtype=dt)
        self.X, self.var, self.R = z(B, C, 2), z(B, C), z(B, C, 2)       # R: repairs added to earlier frames
        self.trail = z(B, C, dt=torch.long)
        self.keys = z(B, C, NCELL, 32); self.off = z(B, C, NCELL, 2); self.wm = z(B, C, NCELL, dt=torch.bool)
        self.tok = z(B, C, NCELL, 16, dt=torch.float16); self.light = z(B, C, 8)
        self.g = z(B, C, 9, 64, dt=torch.float16); self.win = torch.full((B, C, 9), -1, dtype=torch.long, device=DEV)
        self.unc = z(B, C); self.dream = z(B, C, dt=torch.bool)
        self.anchor = z(B, dt=torch.long); self.var_since = z(B); self.cur = z(B, dt=torch.long)
        self.lost = z(B, dt=torch.bool); self.n_trails = torch.ones(B, dtype=torch.long, device=DEV)
        self.rec_t = torch.full((B, ACC), -10 ** 9, dtype=torch.long, device=DEV); self.rec_d = z(B, ACC, 2)
        self.n = z(B, dt=torch.long)                                     # frames written
        self.frames = torch.arange(C, device=DEV)
        self.applied = 0; self.proposals = 0
        self.log = [[] for _ in range(B)]                                # per lane: (t, feats, delta, gamma, ok)
        self.keep_log = False

    def reset(self, lanes):
        if not len(lanes):
            return
        for a in (self.X, self.var, self.R, self.keys, self.off, self.light, self.unc):
            a[lanes] = 0
        for a in (self.trail, self.wm, self.dream, self.anchor, self.cur, self.lost, self.n):
            a[lanes] = 0
        self.tok[lanes] = 0; self.g[lanes] = 0; self.win[lanes] = -1
        self.var_since[lanes] = 0; self.n_trails[lanes] = 1; self.rec_t[lanes] = -10 ** 9; self.rec_d[lanes] = 0
        for b in lanes.tolist():
            self.log[b] = []

    # ------------------------------------------------------------------ before the frame

    def predict(self, t, a, d, sensed):
        """t [B] frame index (> 0), a [B] command, d [B, 2] sensed displacement, sensed [B] bool -> x [B, 2], var [B]."""
        bi = torch.arange(self.B, device=DEV)
        prev = self.X[bi, (t - 1).clamp(min=0)]
        mode, var = self.p["motion"].mode(self.tok[bi, (t - 1).clamp(min=0)].float(), self.light[bi, (t - 1).clamp(min=0)], a)
        x = prev + torch.where(sensed[:, None], d, mode)
        v = torch.where(sensed, torch.full_like(var, LOC.SENSED_VAR), var)
        return torch.where((t == 0)[:, None], torch.zeros_like(x), x), torch.where(t == 0, torch.full_like(v, LOC.SENSED_VAR), v)

    def read(self, t, xq, X=None, extra=None):
        """The read's candidates for frame t [B] at places xq [B, 2] + each map cell's offset: the current trail's
        entries written before t, the nearest 100 (exact ties: newer first), ranked by code similarity, at most 5 per
        place newest first, 25. Returns the features [B, 63, 25, FEAT] and each cell's best candidate (frame, cell, ok)."""
        X = self.X if X is None else X
        calib, code = self.p["calib"], self.p["code"]
        off = calib.offset.to(DEV)[:MAP]; world = calib.world.to(DEV)[:MAP]
        Fm = int(t.max())
        out = torch.zeros(self.B, MAP, KC_, FEAT, device=DEV)
        if Fm == 0:
            return out, None
        q = xq[:, None] + off[None]                                                          # [B, 63, 2]

        def score(a, b):
            valid = self.wm[:, a:b] & (self.trail[:, a:b, None] == self.cur[:, None, None]) & (self.frames[None, a:b, None] < t[:, None, None])
            Pe = (X[:, a:b, None] + self.off[:, a:b]).reshape(self.B, -1, 2)
            fr_all = self.frames[a:b, None].expand(b - a, NCELL).reshape(-1)
            dist = (Pe[:, None] - q[:, :, None]).abs().max(-1).values + 1e-4 * (1 - fr_all.float() / self.C)
            return dist.masked_fill(~valid.reshape(self.B, -1)[:, None], float("inf"))
        dn, near = chunked_topk(score, Fm, NEAR, largest=False)
        k1 = dn.shape[-1]
        okn = torch.isfinite(dn)
        fr, ce = near // NCELL, near % NCELL
        bi0 = torch.arange(self.B, device=DEV)[:, None, None]
        P_c = X[bi0, fr] + self.off[bi0, fr, ce]
        sim = code.sim(q[:, :, None], P_c)
        b_ = torch.round(P_c / WIDTH).long(); bk = b_[..., 0] * 100003 + b_[..., 1]
        rank = ((bk[..., :, None] == bk[..., None, :]) & (fr[..., None, :] > fr[..., :, None]) & okn[..., None, :]).sum(-1)
        score = torch.where((rank < PER_PLACE) & okn, sim + 1e-7 * fr.float(), torch.full_like(sim, -1e9))
        k2 = min(KC_, k1)
        sc, sel = score.topk(k2, dim=-1)
        ok = (sc > -1e8) & world[None, :, None]
        g2 = lambda a: torch.gather(a, 2, sel)
        f_, c_, s_ = g2(fr), g2(ce), g2(sim)
        P_s = torch.gather(P_c, 2, sel[..., None].expand(-1, -1, -1, 2))
        bi = torch.arange(self.B, device=DEV)[:, None, None]
        age = (t[:, None, None] - f_).float()
        agef = LM.PR.age_features(age, ok)
        tok = self.tok[bi, f_, c_].float()
        wconf = (1 / (1 + self.unc[bi, f_]))[..., None]
        light = self.light[bi, f_]
        w9 = self.win[bi, f_]
        hitj = w9 == c_[..., None]
        fpres = hitj.any(-1) & ok
        j = hitj.float().argmax(-1)
        fine = self.g[bi, f_, j].float() * fpres[..., None]
        crop = F.one_hot(j, 9).float() * fpres[..., None]
        reld = P_s - q[:, :, None]
        unc = self.unc[torch.arange(self.B, device=DEV), (t - 1).clamp(min=0)][:, None, None, None].expand(-1, MAP, k2, 1)
        dre = self.dream[bi, f_].float()[..., None]
        f = torch.cat([agef, tok, wconf, light, fine, fpres[..., None].float(), crop, reld, s_[..., None], unc, dre], -1)
        out[:, :, :k2] = f * ok[..., None]
        return out, (f_[..., 0], c_[..., 0], ok[..., 0])

    # ------------------------------------------------------------------ the frame

    def step(self, t, p, v, g, fix, x, var, sensed_now, active):
        """Frame t [B] observed (lanes in `active`): place it at x, recognise, maybe correct (and repair), write."""
        B, bi = self.B, torch.arange(self.B, device=DEV)
        act = active
        self.X[bi[act], t[act]] = x[act]; self.var[bi[act], t[act]] = var[act]
        self.var_since = torch.where(act, self.var_since + var, self.var_since)
        newly = act & ~self.lost & (self.var_since > WIDTH ** 2)
        self.lost |= newly
        self.cur = torch.where(newly, self.n_trails, self.cur); self.n_trails = self.n_trails + newly.long()
        self.trail[bi[act], t[act]] = self.cur[act]
        if self.mode != "none":
            pr = self.propose(t, p, v, act & (t > 0))
            if pr is not None:
                has, delta, f, trl = pr
                f[:, 4] = sensed_now.float()
                big = delta.abs().max(-1).values > WIDTH
                slot = t % ACC
                upd = has & big
                old = self.rec_t < (t - ACC + 1)[:, None]
                self.rec_t = torch.where(upd[:, None] & old, torch.full_like(self.rec_t, -10 ** 9), self.rec_t)
                self.rec_t[bi[upd], slot[upd]] = t[upd]; self.rec_d[bi[upd], slot[upd]] = delta[upd]
                live = self.rec_t >= (t - ACC + 1)[:, None]
                cons_big = (live & ((self.rec_d - delta[:, None]).abs().max(-1).values < WIDTH)).sum(-1)
                cons = torch.where(big, cons_big, torch.full_like(cons_big, ACC))
                f[:, 6] = torch.where(big, cons.float() / ACC, torch.ones_like(f[:, 6]))
                gam = self.p["gate"](f, v) if self.use_gate else torch.zeros(B, device=DEV)
                ok = has & (cons >= ACC) & (gam > 0.5) & (delta.abs().max(-1).values > 1e-3) & torch.tensor(self.use_gate, device=DEV)
                self.proposals += int(has.sum())
                if self.keep_log:
                    for b in torch.nonzero(has)[:, 0].tolist():
                        self.log[b].append((int(t[b]), f[b].cpu(), delta[b].cpu(), float(gam[b]), bool(ok[b])))
                if ok.any():
                    self.apply(t, gam[:, None] * delta, trl, ok)
        cells = torch.arange(NCELL, device=DEV)
        ia = bi[act]; ta = t[act]
        if len(ia):
            k = self.p["key"](p[ia].reshape(-1, 16), v[ia][:, None].expand(-1, NCELL, -1).reshape(-1, 8),
                              cells[None].expand(len(ia), -1).reshape(-1)).view(len(ia), NCELL, 32)
            self.keys[ia, ta] = k
            self.off[ia, ta] = self.p["calib"].offset.to(DEV); self.wm[ia, ta] = self.p["calib"].world.to(DEV)
            self.tok[ia, ta] = p[ia].half(); self.light[ia, ta] = v[ia]; self.g[ia, ta] = g[ia].half()
            self.win[ia, ta] = M.window(fix[ia]); self.unc[ia, ta] = self.var_since[ia].sqrt()
            self.n[ia] = torch.maximum(self.n[ia], ta + 1)

    def propose(self, t, p, v, act):
        """Recognition, as location.Estimator.propose, for the lanes in act with t - EXCL > 0."""
        Fn = t - EXCL
        act = act & (Fn > 0)
        if not act.any():
            return None
        calib = self.p["calib"]
        cells = torch.nonzero(calib.world.to(DEV))[:, 0]; nW = len(cells)
        if not nW:
            return None
        B = self.B
        q = self.p["key"](p[:, cells].reshape(-1, 16), v[:, None].expand(-1, nW, -1).reshape(-1, 8),
                          cells[None].expand(B, -1).reshape(-1)).view(B, nW, 32)
        Fm = int(Fn.max())

        def score(a, b):
            valid = self.wm[:, a:b] & (self.frames[None, a:b, None] < Fn[:, None, None])
            sims = torch.einsum("bqd,bmd->bqm", q, self.keys[:, a:b].reshape(B, -1, 32))
            fr_all = self.frames[a:b, None].expand(b - a, NCELL).reshape(-1)
            sims = sims + 1e-6 * fr_all.float() / self.C                  # exact ties: newer first
            return sims.masked_fill(~valid.reshape(B, -1)[:, None], -float("inf"))
        ts_, ti = chunked_topk(score, Fm, KREC, largest=True)              # [B, nW, k]
        k = ts_.shape[-1]
        okc = torch.isfinite(ts_)
        fr, ce = ti // NCELL, ti % NCELL
        bi = torch.arange(B, device=DEV)[:, None, None]
        P_e = self.X[bi, fr] + self.off[bi, fr, ce]
        xh = P_e - calib.offset.to(DEV)[cells][None, :, None]
        w_ = ts_.clamp(min=0).masked_fill(~okc, 0)
        trl = self.trail[bi, fr]
        kept = torch.zeros(B, nW, k, dtype=torch.bool, device=DEV)
        for j in range(k):                                               # the greedy choice, batched
            prior = kept[..., :j]
            same_f = (fr[..., :j] == fr[..., j:j + 1]) & prior
            near_p = ((xh[..., :j, :] - xh[..., j:j + 1, :]).abs().max(-1).values < WIDTH) & prior
            room = prior.sum(-1) < TOPR
            kept[..., j] = okc[..., j] & ~same_f.any(-1) & ~near_p.any(-1) & room
        # compact the kept points, in cell order then candidate order (as the per-episode lists)
        flat_k = kept.reshape(B, -1)
        nmax = int(flat_k.sum(-1).max()) if flat_k.any() else 0
        has = act & flat_k.any(-1)
        if nmax == 0:
            return None
        order = torch.argsort((~flat_k).to(torch.int8), dim=-1, stable=True)[:, :nmax]
        gk = lambda a: torch.gather(a.reshape(B, -1, *a.shape[3:]), 1, order.view(B, nmax, *([1] * (a.dim() - 3))).expand(B, nmax, *a.shape[3:]))
        pts, ws, trs = gk(xh), gk(w_), gk(trl)
        cid = torch.gather(torch.arange(nW, device=DEV)[None, :, None].expand(B, nW, k).reshape(B, -1), 1, order)
        pv = torch.gather(flat_k, 1, order)
        near = ((pts[:, :, None] - pts[:, None]).abs().max(-1).values < WIDTH) & pv[:, :, None] & pv[:, None, :]
        vals = near.float() * ws[:, None, :]
        Mx = torch.zeros(B, nmax, nW, device=DEV).scatter_reduce(2, cid[:, None, :].expand(B, nmax, nmax), vals, "amax", include_self=True)
        sup = Mx.sum(-1).masked_fill(~pv, -1.0)
        b = sup.argmax(-1)
        bb = torch.arange(B, device=DEV)
        mem = near[bb, b]
        wsum = (ws * mem).sum(-1)
        xhat = (pts * (ws * mem)[..., None]).sum(1) / wsum.clamp(min=1e-6)[:, None]
        far = ((pts - pts[bb, b][:, None]).abs().max(-1).values > 2 * WIDTH) & pv
        second = torch.where(far.any(-1), sup.masked_fill(~far, -1.0).max(-1).values, torch.zeros(B, device=DEV))
        cellhit = torch.zeros(B, nW, device=DEV).scatter_add(1, cid, mem.float()) > 0
        ncell = cellhit.sum(-1).float()
        nall = (torch.zeros(B, nW, device=DEV).scatter_add(1, cid, pv.float()) > 0).sum(-1).float()
        maxtr = int(self.n_trails.max()) + 1
        tcount = torch.zeros(B, maxtr, device=DEV).scatter_add(1, trs.clamp(max=maxtr - 1), mem.float())
        trail = tcount.argmax(-1)
        delta = xhat - self.X[bb, t]
        supb = sup[bb, b]
        f = torch.stack([ncell / nW, supb / nW, second / supb.clamp(min=1e-6), delta.norm(dim=-1) / 4, torch.zeros(B, device=DEV),
                         self.var_since.sqrt(), torch.zeros(B, device=DEV), nall / nW, self.lost.float()], -1)
        return has, delta, f, trail

    def apply(self, t, c, trail, ok):
        """Corrections c [B, 2] at frames t for the lanes in ok: the current frame; with repair, the frames since the
        anchor by their share of the cumulative step variance (both ends fixed)."""
        bi = torch.arange(self.B, device=DEV)
        if self.mode == "full":
            rng = (self.frames[None] > self.anchor[:, None]) & (self.frames[None] <= t[:, None])
            cs = torch.cumsum(self.var * rng, 1)
            tot = cs[bi, t].clamp(min=1e-12)
            share = cs / tot[:, None]
            rep_ = (rng & (self.frames[None] < t[:, None]) & (ok & (t - self.anchor > 1))[:, None]).float()
            add = rep_[..., None] * share[..., None] * c[:, None]
            self.X += add; self.R += add
        self.X[bi[ok], t[ok]] += c[ok]
        join = ok & (self.lost | (self.cur != trail))
        for b in torch.nonzero(join)[:, 0].tolist():
            self.trail[b][self.trail[b] == self.cur[b]] = trail[b]
        self.cur = torch.where(join, trail, self.cur)
        self.anchor = torch.where(ok, t, self.anchor); self.var_since = torch.where(ok, torch.zeros_like(self.var_since), self.var_since)
        self.lost = self.lost & ~ok
        self.rec_t = torch.where(ok[:, None], torch.full_like(self.rec_t, -10 ** 9), self.rec_t)
        self.applied += int(ok.sum())


def run_lanes(parts, eps, mode="full", masks=None, hook=None, keep_log=False, post_hook=None):
    """Episodes as lanes, stepped together. eps: dicts with p, v, g, fix, prev_action [T + 1, ...] and pos (motion
    sensor). masks: per episode, True where the sense is withheld. hook(mem, t, x, active): called before each step
    (the reading's features). Returns the memory and per lane x before correction."""
    B = len(eps); T = [len(e["p"]) - 1 for e in eps]; C = max(T) + 1
    mem = BatchMemory(parts, B, C, mode); mem.keep_log = keep_log
    pad = lambda k, shape, dt: torch.stack([torch.cat([torch.as_tensor(e[k]).to(DEV).to(dt), torch.zeros((C - len(e[k]),) + shape, device=DEV, dtype=dt)]) for e in eps])
    P = pad("p", (NCELL, 16), torch.float32); V = pad("v", (8,), torch.float32); G = pad("g", (9, 64), torch.float32)
    FX = pad("fix", (2,), torch.long); A = pad("prev_action", (), torch.long)
    pos = torch.stack([torch.cat([torch.as_tensor(e["pos"], dtype=torch.float32), torch.zeros(C - len(e["pos"]), 2)]) for e in eps]).to(DEV)
    d = torch.cat([torch.zeros(B, 1, 2, device=DEV), pos[:, 1:] - pos[:, :-1]], 1)
    W = torch.zeros(B, C, dtype=torch.bool, device=DEV)
    if masks is not None:
        for b, m in enumerate(masks):
            if m is not None:
                W[b, :len(m)] = torch.as_tensor(m, device=DEV)
    xpre = torch.zeros(B, C, 2, device=DEV)
    Tt = torch.tensor(T, device=DEV)
    with torch.no_grad():
        for t in range(C):
            tt = torch.full((B,), t, device=DEV, dtype=torch.long)
            active = tt <= Tt
            sensed = ~W[:, t]
            if t == 0:
                x, var = torch.zeros(B, 2, device=DEV), torch.full((B,), LOC.SENSED_VAR, device=DEV)
            else:
                x, var = mem.predict(tt, A[:, t], d[:, t], sensed)
            xpre[:, t] = x
            if hook is not None:
                hook(mem, tt, x, active)
            mem.step(tt, P[:, t], V[:, t], G[:, t], FX[:, t], x, var, sensed, active)
            if post_hook is not None:
                post_hook(mem, tt, active)
    return mem, xpre


def selftest(n=6):
    """The batched memory against location.Estimator on panel episodes with the training outages: positions after
    correction, corrections applied, proposals."""
    parts = LOC.load()
    data = torch.load(LM.DATA / "tokens_panel.pt")[:n]
    rng = np.random.default_rng(7)
    masks = [LOC.outage_mask(r["T"], rng) for r in data]
    t0 = time.time()
    mem, xpre = run_lanes(parts, data, "full", masks, keep_log=True)
    tb = time.time() - t0
    res = {"episodes": n, "batched_seconds": tb}
    agree_x, agree_ok, ref_time = [], [], 0.0
    for b, r in enumerate(data):
        t1 = time.time()
        est, xp_ref, _ = LOC.run_episode(parts, r, "full", True, masks[b])
        ref_time += time.time() - t1
        T = r["T"]
        dx = (mem.X[b, :T + 1] - est.X[:T + 1]).abs().max(-1).values
        agree_x.append(float((dx < 1e-3).float().mean()))
        okr = {t for t, f, dl, g, ok in est.log if ok}; okb = {t for t, f, dl, g, ok in mem.log[b] if ok}
        agree_ok.append((len(okr), len(okb), len(okr & okb)))
    res.update({"reference_seconds": ref_time, "frames": int(sum(r["T"] + 1 for r in data)),
                "positions_identical_share": agree_x, "corrections_ref_batched_both": agree_ok})
    print(json.dumps(res, indent=1))
    (LOC.HERE / "batch_selftest.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    {"selftest": lambda: selftest()}[sys.argv[1]]()
