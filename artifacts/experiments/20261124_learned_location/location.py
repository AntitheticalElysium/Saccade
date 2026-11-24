"""Decision 6's locked design, the location stage (stage A of PLAN.md), and its gates.

The parts:
- a running calibration of the sensor from the agent's own correspondences under sensed motion;
- the motion model (a mixture density network taught by the motion sense);
- the position code (commuting rotations with learned frequencies over the path-integrated displacement);
- the recognition key (contrastive, positives from the agent's own integrated motion, lookalike negatives);
- the gate;
- the estimator: path integration, recognition, gated correction accepted over three frames, repair along the trail,
  a fresh trail when lost.

Simulator positions enter only as the declared motion sensor (their difference) and in scoring.

    LOC_DEVICE=cuda JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261124_learned_location.location smoke | stage_a | gates
"""
import importlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LM = importlib.import_module("artifacts.experiments.20261123_located_memory.memory")
PR, S2, D = LM.PR, LM.S2, LM.D
HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
DEV = torch.device(os.environ.get("LOC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
NCELL, PH, KM = 81, 32, 4                 # screen cells; phases (D = 64); motion mixture components (PLAN)
TOPR, EXCL = 8, 16                        # recognition candidates per cell; recent frames excluded (PLAN)
WIDTH = 0.5                               # the read's similarity width, in motion units
ACC = 3                                   # frames a large correction must hold (ORB-SLAM3)
OUT_P, OUT_MIN, OUT_MAX, WIN = 0.5, 8, 48, 64
HOLD = 30                                 # TRAIN episodes 270-299 held out for the gates
SIGMA = 1.0                               # the code's similarity kernel width (choice, PLAN)
SENSED_VAR = 1e-6                         # a sensed step's variance (Classic's sensor is exact)


# ---------------------------------------------------------------- data

def train_data():
    """located_memory's 300 TRAIN episodes with their S0 positions (the motion sensor)."""
    data = torch.load(LM.DATA / "tokens_train.pt")
    ck = torch.load(S2.P15, map_location="cpu", weights_only=False)
    split = D.load_split(ck["cfg"])
    rng = np.random.default_rng(20261123)                              # memory.tokens()'s draw, reproduced
    tr = [split["train"][i] for i in rng.choice(len(split["train"]), 300, replace=False)]
    spos, start = PR.label_positions(tr)
    for r, e, s0 in zip(data, tr, start):
        assert r["T"] == e.T
        r["pos"] = torch.as_tensor(spos[s0:s0 + e.T + 1])
    return data


def sensed(r):
    """The motion sensor: displacement since the previous frame [T + 1, 2] (zero at the start)."""
    pos = r["pos"].float()
    return torch.cat([torch.zeros(1, 2), pos[1:] - pos[:-1]])


# ---------------------------------------------------------------- calibration

class Calibration:
    """Each screen cell's offset and whether it shows the world, from the agent's own token correspondences under
    sensed motion (Pierce & Kuipers 1997). No renderer layout: reciprocal best matches between all 81 cells, then a
    robust L1 fit of o_k - o_k' = d over the largest connected component of links."""

    def __init__(self):
        self.S, self.N = {}, {}
        self.offset = torch.zeros(NCELL, 2)
        self.world = torch.zeros(NCELL, dtype=torch.bool)
        self.versions = []

    def observe(self, p0, p1, d):
        if float(d.abs().sum()) == 0:
            return
        key = (round(float(d[0]), 2), round(float(d[1]), 2))
        dm = torch.cdist(p0[None].float(), p1[None].float(), p=1)[0].double().cpu()
        self.S[key] = self.S.get(key, 0) + dm
        self.N[key] = self.N.get(key, 0) + 1

    def fit(self, min_n=50):
        links = []
        for key, Ssum in self.S.items():
            if self.N[key] < min_n:
                continue
            Dm = Ssum / self.N[key]
            fwd, bwd = Dm.argmin(1), Dm.argmin(0)
            links += [(k, int(fwd[k]), key) for k in range(NCELL) if int(bwd[fwd[k]]) == k and int(fwd[k]) != k]
        if not links:
            return
        par = list(range(NCELL))
        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for k, kp, _ in links:
            par[find(k)] = find(kp)
        comp = defaultdict(list)
        for k in {k for k, _, _ in links} | {kp for _, kp, _ in links}:
            comp[find(k)].append(k)
        cells = sorted(max(comp.values(), key=len))
        ix = {k: i for i, k in enumerate(cells)}
        L = [(k, kp, d) for k, kp, d in links if k in ix and kp in ix]
        A = np.zeros((2 * len(L) + 2, 2 * len(cells))); y = np.zeros(2 * len(L) + 2)
        for i, (k, kp, d) in enumerate(L):
            for a in range(2):
                A[2 * i + a, 2 * ix[k] + a] = 1; A[2 * i + a, 2 * ix[kp] + a] = -1; y[2 * i + a] = d[a]
        A[-2, 0] = A[-1, 1] = 1                                          # gauge: the first cell at the origin
        w = np.ones(len(y))
        for _ in range(30):                                              # IRLS for the L1 fit
            sw = np.sqrt(w)[:, None]
            sol = np.linalg.lstsq(A * sw, y * sw[:, 0], rcond=None)[0]
            w = 1 / np.maximum(np.abs(A @ sol - y), 1e-3)
        sol = sol.reshape(-1, 2)
        res = np.abs(A @ sol.reshape(-1) - y)[:-2].reshape(-1, 2).max(1)
        good = {c for (k, kp, _), r_ in zip(L, res) if r_ <= 0.5 for c in (k, kp)}
        off = torch.zeros(NCELL, 2); world = torch.zeros(NCELL, dtype=torch.bool)
        sol = sol - np.median(sol, 0)
        for k in cells:
            if k in good:
                off[k] = torch.as_tensor(sol[ix[k]], dtype=torch.float32); world[k] = True
        self.offset, self.world = off, world
        self.versions.append({"transitions": int(sum(self.N.values())), "links": len(links),
                              "inconsistent_links": int((res > 0.5).sum()), "world_cells": int(world.sum())})

    def state(self):
        return {"offset": self.offset, "world": self.world, "versions": self.versions,
                "S": {str(k): v for k, v in self.S.items()}, "N": {str(k): v for k, v in self.N.items()}}

    def load(self, st):
        self.offset, self.world, self.versions = st["offset"], st["world"], st["versions"]
        self.S = {tuple(float(x) for x in k.strip("()").split(",")): v for k, v in st["S"].items()}
        self.N = {tuple(float(x) for x in k.strip("()").split(",")): v for k, v in st["N"].items()}

    def score(self):
        """Agreement with the renderer (scoring only): offsets up to a constant; world cells against the map cells
        without the player's."""
        truth = torch.stack([LM.KR[:LM.MAP].float(), LM.KC[:LM.MAP].float()], -1)
        mapc = torch.arange(LM.MAP) != LM.SELF
        w = self.world[:LM.MAP]
        common = w & mapc
        dif = (self.offset[:LM.MAP][common] - truth[common])
        const = dif.median(0).values
        return {"world_cells": int(self.world.sum()), "map_cells_world": int(common.sum()), "map_cells_expected": int(mapc.sum()),
                "nonmap_flagged_world": int(self.world[LM.MAP:].sum() + int(self.world[LM.SELF])),
                "offsets_exact_up_to_constant": float(((dif - const).abs() < 0.05).all(1).float().mean()) if common.any() else 0.0}


# ---------------------------------------------------------------- learned parts

class Motion(nn.Module):
    """p(δ | view, lighting, command): a mixture density network (Bishop 1994) taught by the motion sense."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NCELL * 16 + 8 + 17, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, KM * 5))

    def forward(self, p, v, a):
        o = self.net(torch.cat([p.flatten(1).float(), v.float(), F.one_hot(a.clamp(min=0), 17).float()], -1)).view(-1, KM, 5)
        return o[..., 0].log_softmax(-1), o[..., 1:3], o[..., 3:5].clamp(-7, 3)

    def nll(self, p, v, a, d):
        lp, mu, ls = self(p, v, a)
        lg = (-0.5 * ((d[:, None] - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -torch.logsumexp(lp + lg, -1)

    def mode(self, p, v, a):
        """The most probable component's mean (one coherent motion) and the mixture's total variance."""
        lp, mu, ls = self(p, v, a)
        w = lp.exp()
        m = (w[..., None] * mu).sum(1)
        var = ((w[..., None] * (ls.exp() ** 2 + mu ** 2)).sum(1) - m ** 2).sum(-1)
        return mu[torch.arange(len(mu)), lp.argmax(-1)], var

    def sample(self, p, v, a, gen):
        lp, mu, ls = self(p, v, a)
        c = torch.multinomial(lp.exp(), 1, generator=gen)[:, 0]
        i = torch.arange(len(mu), device=mu.device)
        return mu[i, c] + ls[i, c].exp() * torch.randn(mu[i, c].shape, device=mu.device, generator=gen)


class Code(nn.Module):
    """The position code: phases Ω x over the path-integrated displacement x, commuting rotations with learned
    frequencies (Gao et al. 2019's block-diagonal form)."""

    def __init__(self, seed=20261124):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        per = torch.exp(torch.empty(PH).uniform_(math.log(2.0), math.log(64.0), generator=g))
        ang = torch.empty(PH).uniform_(0, 2 * math.pi, generator=g)
        self.W = nn.Parameter(torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / per)[:, None])

    def phase(self, x):
        return x @ self.W.T

    def sim(self, a, b):
        return torch.cos(self.phase(a - b)).mean(-1)

    def rel(self, q, e):
        ph = self.phase(e - q)
        return torch.cat([ph.cos(), ph.sin()], -1)


class Key(nn.Module):
    """The recognition key κ(token, lighting, screen cell): unit vectors compared by dot product."""

    def __init__(self):
        super().__init__()
        self.cell = nn.Embedding(NCELL, 8)
        self.net = nn.Sequential(nn.Linear(16 + 8 + 8, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 32))

    def forward(self, tok, light, cell):
        return F.normalize(self.net(torch.cat([tok.float(), light.float(), self.cell(cell)], -1)), dim=-1)


class Gate(nn.Module):
    """γ ∈ (0, 1) for a proposed correction, from its agreement features and the lighting."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9 + 8, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, f, v):
        return torch.sigmoid(self.net(torch.cat([f, v.float()], -1)))[..., 0]


# ---------------------------------------------------------------- the estimator

class Estimator:
    """One episode's location and memory. mode: 'full' (correction and repair), 'current' (correction of the current
    frame only), 'none' (no recognition). gate None: proposals are recorded, never applied (gate training round 1).
    The entries are every frame's world cells, each with the offset it was written with."""

    def __init__(self, parts, T, mode="full", gate=True):
        self.m = parts; self.mode = mode; self.use_gate = gate
        dev = DEV
        self.X = torch.zeros(T + 1, 2, device=dev); self.var = torch.zeros(T + 1, device=dev)
        self.trail = torch.zeros(T + 1, dtype=torch.long, device=dev)
        self.keys = torch.zeros(T + 1, NCELL, 32, device=dev)
        self.off = torch.zeros(T + 1, NCELL, 2, device=dev)
        self.wm = torch.zeros(T + 1, NCELL, dtype=torch.bool, device=dev)
        self.anchor, self.var_since, self.cur, self.lost, self.n_trails = 0, 0.0, 0, False, 1
        self.recent = []                                                 # (t, Δ) of recent large proposals
        self.log = []                                                    # proposals: (t, features, Δ, applied γ)

    def predict(self, t, p_prev, v_prev, a, d_sensed):
        """The displacement before frame t: sensed when given, else the motion model's mode."""
        if d_sensed is not None:
            return self.X[t - 1] + d_sensed.to(DEV), SENSED_VAR
        d, var = self.m["motion"].mode(p_prev[None].to(DEV), v_prev[None].to(DEV), a[None].to(DEV))
        return self.X[t - 1] + d[0], float(var[0])

    def propose(self, t, p, v):
        """Recognition: each world cell's top matches among entries older than EXCL frames, diversified by source frame
        and implied place; the consensus place and its agreement features."""
        Fn = t - EXCL
        if Fn <= 0:
            return None
        cells = torch.nonzero(self.m["calib"].world.to(DEV))[:, 0]
        if not len(cells):
            return None
        q = self.m["key"](p[cells].to(DEV), v[None].to(DEV).expand(len(cells), 8), cells)
        E, wm = self.keys[:Fn], self.wm[:Fn]
        fi, ci = torch.nonzero(wm, as_tuple=True)
        if not len(fi):
            return None
        sims = q @ E[fi, ci].T                                           # [nq, M]
        k = min(32, sims.shape[1])
        top_s, top_i = sims.topk(k, dim=1)
        P_e = self.X[fi[top_i]] + self.off[fi[top_i], ci[top_i]]          # [nq, k, 2] the entries' places
        o_now = self.m["calib"].offset.to(DEV)[cells]
        xh = P_e - o_now[:, None]                                        # implied agent places
        frame = fi[top_i].cpu().numpy(); xh_ = xh.cpu().numpy(); ts_ = top_s.clamp(min=0).cpu().numpy()
        trl = self.trail[fi[top_i]].cpu().numpy()
        pts, ws, cid, tr = [], [], [], []
        for a in range(len(cells)):                                      # at most one per source frame and place
            seen_f, places = set(), []
            for j in range(k):
                f = frame[a, j]
                if f in seen_f:
                    continue
                x = xh_[a, j]
                if any(np.abs(x - y).max() < WIDTH for y in places):
                    continue
                seen_f.add(f); places.append(x)
                pts.append(x); ws.append(ts_[a, j]); cid.append(a); tr.append(trl[a, j])
                if len(places) == TOPR:
                    break
        if not pts:
            return None
        pts, ws, cid, tr = np.array(pts), np.array(ws), np.array(cid), np.array(tr)
        near = np.abs(pts[:, None] - pts[None]).max(-1) < WIDTH           # [n, n]
        M_ = np.zeros((len(pts), len(cells)))
        for c in np.unique(cid):                                         # support: best weight per distinct cell
            m = cid == c
            M_[:, c] = (near[:, m] * ws[m][None]).max(1)
        sup = M_.sum(1)
        b = int(sup.argmax()); mem = near[b]
        xhat = torch.as_tensor((pts[mem] * ws[mem, None]).sum(0) / max(ws[mem].sum(), 1e-6), dtype=torch.float32, device=DEV)
        far = np.abs(pts - pts[b]).max(-1) > 2 * WIDTH
        second = float(sup[far].max()) if far.any() else 0.0
        ncell = len(np.unique(cid[mem]))
        trail = int(np.bincount(tr[mem]).argmax())
        delta = xhat - self.X[t]
        feats = torch.tensor([ncell / len(cells), float(sup[b]) / len(cells), second / max(float(sup[b]), 1e-6),
                              float(delta.norm()) / 4, 0.0, math.sqrt(self.var_since), 0.0,
                              len(np.unique(cid)) / len(cells), float(self.lost)], device=DEV)
        return {"delta": delta, "feats": feats, "trail": trail}

    def step(self, t, p, v, x_pred, var, sensed_now, p_prev=None, v_prev=None):
        """Frame t observed: place it at x_pred, recognise, maybe correct (and repair), write."""
        self.X[t] = x_pred; self.var[t] = var; self.var_since += var
        if not self.lost and self.var_since > WIDTH ** 2:
            self.lost = True; self.cur = self.n_trails; self.n_trails += 1
        self.trail[t] = self.cur
        applied = None
        if self.mode != "none" and t > 0:
            pr = self.propose(t, p, v)
            if pr is not None:
                f = pr["feats"].clone(); f[4] = float(sensed_now)
                big = float(pr["delta"].abs().max()) > WIDTH
                if big:
                    self.recent = [(s, d) for s, d in self.recent if s >= t - ACC + 1] + [(t, pr["delta"])]
                    cons = sum(float((d - pr["delta"]).abs().max()) < WIDTH for _, d in self.recent)
                    f[6] = cons / ACC
                else:
                    cons = ACC; f[6] = 1.0
                gam = float(self.m["gate"](f[None], v[None].to(DEV))[0]) if self.use_gate else 0.0
                ok = self.use_gate and cons >= ACC and gam > 0.5 and float(pr["delta"].abs().max()) > 1e-3
                self.log.append((t, f.cpu(), pr["delta"].cpu(), gam, ok))
                if ok:
                    applied = self.apply(t, gam * pr["delta"], pr["trail"])
        cells = torch.arange(NCELL, device=DEV)
        self.keys[t] = self.m["key"](p.to(DEV), v[None].to(DEV).expand(NCELL, 8), cells)
        self.off[t] = self.m["calib"].offset.to(DEV); self.wm[t] = self.m["calib"].world.to(DEV)
        return applied

    def apply(self, t, c, trail):
        """A correction c at frame t: the current frame always; with repair, the frames since the anchor by their
        share of the cumulative step variance (the chain's least squares with both ends fixed)."""
        if self.mode == "full" and t - self.anchor > 1:
            v = self.var[self.anchor + 1:t + 1]
            share = torch.cumsum(v, 0) / v.sum().clamp(min=1e-12)
            self.X[self.anchor + 1:t] += share[:-1, None] * c
        self.X[t] += c
        if self.lost or self.cur != trail:                               # join the fresh trail to the recognised one
            self.trail[self.trail == self.cur] = trail
            self.cur = trail
        self.anchor, self.var_since, self.lost, self.recent = t, 0.0, False, []
        return c


def run_episode(parts, r, mode="full", gate=True, outages=None):
    """The estimator over one episode. outages: a boolean mask [T + 1], True where the sense is withheld. Returns the
    estimator and, per frame, x before correction and the true integrated displacement (scoring)."""
    T = r["T"]
    est = Estimator(parts, T, mode, gate)
    d = sensed(r)
    xtrue = torch.cumsum(d, 0).to(DEV)
    xpre = torch.zeros(T + 1, 2, device=DEV)
    p, v, a = r["p"].float(), r["v"].float(), r["prev_action"]
    with torch.no_grad():
        for t in range(T + 1):
            if t == 0:
                x, var = torch.zeros(2, device=DEV), SENSED_VAR
            else:
                withheld = outages is not None and bool(outages[t])
                x, var = est.predict(t, p[t - 1], v[t - 1], a[t], None if withheld else d[t])
            xpre[t] = x
            est.step(t, p[t], v[t], x, var, not (outages is not None and bool(outages[t])))
    return est, xpre, xtrue


def outage_mask(T, rng):
    """The training schedule: in each 64-step window, with probability 0.5, the sense is withheld for 8-48 steps."""
    m = np.zeros(T + 1, bool)
    for w0 in range(1, T + 1, WIN):
        if rng.random() < OUT_P:
            L = int(rng.integers(OUT_MIN, OUT_MAX + 1)); s = w0 + int(rng.integers(0, max(1, WIN - L)))
            m[s:min(T + 1, s + L)] = True
    return m


# ---------------------------------------------------------------- stage A training

def train_motion(data, passes=4):
    mot = Motion().to(DEV)
    opt = torch.optim.Adam(mot.parameters(), lr=1e-3)
    X = [(r["p"][:-1].float(), r["v"][:-1].float(), r["prev_action"][1:], sensed(r)[1:]) for r in data]
    P, V, A, Dd = (torch.cat([x[i] for x in X]) for i in range(4))
    g = torch.Generator().manual_seed(0)
    for ps in range(passes):
        perm = torch.randperm(len(P), generator=g); tot = 0.0
        for i in range(0, len(P), 512):
            b = perm[i:i + 512]
            loss = mot.nll(P[b].to(DEV), V[b].to(DEV), A[b].to(DEV), Dd[b].to(DEV)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        print(f"motion pass {ps} nll {tot / len(P):.4f}", flush=True)
    return mot


def train_code(data, steps=2000):
    code = Code().to(DEV)
    opt = torch.optim.Adam(code.parameters(), lr=1e-2)
    xs = [torch.cumsum(sensed(r), 0) for r in data]
    g = torch.Generator().manual_seed(1)
    for s in range(steps):
        e = [xs[int(i)] for i in torch.randint(len(xs), (64,), generator=g)]
        a = torch.stack([x[int(torch.randint(len(x), (1,), generator=g))] for x in e])
        b = torch.stack([x[int(torch.randint(len(x), (1,), generator=g))] for x in e])
        near = a + torch.randint(-3, 4, (64, 2), generator=g).float()
        A_, B_ = torch.cat([a, a]).to(DEV), torch.cat([b, near]).to(DEV)
        target = torch.exp(-((A_ - B_) ** 2).sum(-1) / (2 * SIGMA ** 2))
        loss = ((code.sim(A_, B_) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"code step {s} loss {float(loss):.5f}", flush=True)
    return code


def episode_world(r, calib):
    """Every world cell's place (integrated sensed displacement plus its offset), tokens, lighting [T + 1, NCELL]."""
    x = torch.cumsum(sensed(r), 0)
    return x[:, None] + calib.offset[None], calib.world


def train_key(data, calib, steps=3000, nq=256, nneg=127, tau=0.1):
    """InfoNCE (temperature 0.1): a world cell's token against an older entry at the same place per the agent's own
    integrated motion (any age or light), and 127 entries elsewhere, half the most similar raw tokens (lookalikes)."""
    key = Key().to(DEV)
    opt = torch.optim.Adam(key.parameters(), lr=1e-3)
    g = np.random.default_rng(2)
    cells_w = torch.nonzero(calib.world)[:, 0]
    nW = len(cells_w)
    prep = []
    for r in data:
        T = r["T"]
        if T < EXCL + 2:
            continue
        P_, _ = episode_world(r, calib)
        pk = (P_[:, cells_w] * 4).round().long().numpy()
        code_ = (pk[..., 0] * 100003 + pk[..., 1]).reshape(-1)          # flat index = frame * nW + a
        order = np.argsort(code_, kind="stable")
        prep.append((r["p"][:, cells_w].float(), r["v"].float(), code_, order, code_[order], T))
    for s_ in range(steps):
        qt, qv, qc, pt, pv, pc, nt, nv, nc = [], [], [], [], [], [], [], [], []
        while len(qt) < nq:
            tok, v, code_, order, cs, T = prep[g.integers(len(prep))]
            t = int(g.integers(EXCL + 1, T + 1)); a = int(g.integers(nW))
            lim = (t - EXCL) * nW                                        # entries from frames <= t - EXCL - 1
            c = code_[t * nW + a]
            lo, hi = np.searchsorted(cs, c, "left"), np.searchsorted(cs, c, "right")
            same = order[lo:hi]; same = same[same < lim]
            if not len(same):
                continue
            j = int(same[g.integers(len(same))])
            pool = g.integers(0, lim, 2048); pool = pool[code_[pool] != c]
            if len(pool) < nneg:
                continue
            d = (tok.view(-1, 16)[pool] - tok[t, a]).abs().sum(-1)
            hard = pool[d.topk(nneg // 2, largest=False).indices.numpy()]
            rnd = pool[g.integers(len(pool), size=nneg - nneg // 2)]
            neg = np.concatenate([hard, rnd])
            qt.append(tok[t, a]); qv.append(v[t]); qc.append(cells_w[a])
            pt.append(tok.view(-1, 16)[j]); pv.append(v[j // nW]); pc.append(cells_w[j % nW])
            nt.append(tok.view(-1, 16)[neg]); nv.append(v[neg // nW]); nc.append(cells_w[neg % nW])
        kq = key(torch.stack(qt).to(DEV), torch.stack(qv).to(DEV), torch.stack(qc).to(DEV))
        kp = key(torch.stack(pt).to(DEV), torch.stack(pv).to(DEV), torch.stack(pc).to(DEV))
        kn = key(torch.cat(nt).to(DEV), torch.cat(nv).to(DEV), torch.cat(nc).to(DEV)).view(nq, nneg, 32)
        logits = torch.cat([(kq * kp).sum(-1, keepdim=True), torch.einsum("qd,qnd->qn", kq, kn)], -1) / tau
        loss = F.cross_entropy(logits, torch.zeros(nq, dtype=torch.long, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if s_ % 250 == 0 or s_ == steps - 1:
            print(f"key step {s_} loss {float(loss):.4f} top1 {float((logits.argmax(-1) == 0).float().mean()):.3f}", flush=True)
    return key


def collect(parts, data, rng, gate):
    """Proposals over episodes with training outages: features, Δ, and the error before and after (scoring the gate
    against the withheld sense)."""
    rows = []
    for r in data:
        m = outage_mask(r["T"], rng)
        est, xpre, xtrue = run_episode(parts, r, "full", gate, m)
        for t, f, dl, _, _ in est.log:
            rows.append((f, r["v"][t].float(), dl, (xtrue[t] - xpre[t]).cpu()))
    return rows


def train_gate(parts, rows, steps=1500):
    gate = Gate().to(DEV)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    Fm = torch.stack([x[0] for x in rows]).to(DEV); V = torch.stack([x[1] for x in rows]).to(DEV)
    Dl = torch.stack([x[2] for x in rows]).to(DEV); E = torch.stack([x[3] for x in rows]).to(DEV)   # E = x_true - x_pre
    g = torch.Generator().manual_seed(3)
    for s in range(steps):
        b = torch.randint(len(Fm), (1024,), generator=g)
        gam = gate(Fm[b], V[b])
        loss = F.huber_loss(gam[:, None] * Dl[b], E[b], delta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    return gate


def save(parts):
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"motion": parts["motion"].state_dict(), "code": parts["code"].state_dict(), "key": parts["key"].state_dict(),
                "gate": parts["gate"].state_dict(), "calib": parts["calib"].state()}, RUN / "location.pt")


def load():
    st = torch.load(RUN / "location.pt", map_location=DEV, weights_only=False)
    parts = {"motion": Motion().to(DEV), "code": Code().to(DEV), "key": Key().to(DEV), "gate": Gate().to(DEV), "calib": Calibration()}
    for k in ("motion", "code", "key", "gate"):
        parts[k].load_state_dict(st[k]); parts[k].eval().requires_grad_(False)
    parts["calib"].load(st["calib"])
    return parts


def stage_a(n_train=270, smoke=False):
    data = train_data()
    tr = data[:8] if smoke else data[:n_train]
    calib = Calibration()
    for i, r in enumerate(tr):
        p, d = r["p"].float(), sensed(r)
        for t in range(1, r["T"] + 1):
            calib.observe(p[t - 1], p[t], d[t])
        if (i + 1) % 50 == 0 or i + 1 == len(tr):
            calib.fit(min_n=10 if smoke else 50)
            print("calibration", json.dumps(calib.versions[-1]), json.dumps(calib.score()), flush=True)
    parts = {"calib": calib}
    parts["motion"] = train_motion(tr, passes=1 if smoke else 4)
    parts["code"] = train_code(tr, steps=50 if smoke else 2000)
    parts["key"] = train_key(tr, calib, steps=20 if smoke else 3000, nq=32 if smoke else 256)
    for k in ("motion", "code", "key"):
        parts[k].eval().requires_grad_(False)
    parts["gate"] = Gate().to(DEV)
    rng = np.random.default_rng(20261124)
    for rnd in (1, 2):                                                    # proposals without, then with, the gate
        rows = collect(parts, tr[:4] if smoke else tr[:120], rng, gate=(rnd == 2))
        print(f"gate round {rnd}: {len(rows)} proposals", flush=True)
        parts["gate"] = train_gate(parts, rows, steps=50 if smoke else 1500).eval().requires_grad_(False)
    save(parts)
    print("stage A saved", flush=True)


# ---------------------------------------------------------------- stage A's gates

def smoke_code(code):
    """Closed paths return, orders agree, offsets compose, retrieval is unchanged by the origin (PLAN gate 1)."""
    g = torch.Generator().manual_seed(5)
    steps = torch.tensor([[1., 0], [0, 1], [-1, 0], [0, -1]], device=DEV)
    x0 = torch.randn(2, generator=g).to(DEV) * 10
    loop = code.phase(x0 + steps.sum(0)) - code.phase(x0)
    ab = code.phase(x0 + steps[0] + steps[1]); ba = code.phase(x0 + steps[1] + steps[0])
    o = torch.tensor([2., -3], device=DEV); d = torch.tensor([1., 1], device=DEV)
    comp = code.phase(x0 + d + o) - (code.phase(x0) + code.phase(d) + code.phase(o))
    c = torch.randn(2, generator=g).to(DEV) * 17
    sim0, sim1 = code.sim(x0, x0 + o), code.sim(x0 + c, x0 + o + c)
    wrap = lambda z: torch.remainder(z + math.pi, 2 * math.pi) - math.pi
    res = {"closed_loop_max": float(wrap(loop).abs().max()), "order_max": float(wrap(ab - ba).abs().max()),
           "offset_composition_max": float(wrap(comp).abs().max()), "origin_change": float((sim0 - sim1).abs())}
    res["pass"] = all(v < 1e-4 for v in res.values())
    return res


def revisit_coverage(parts, r):
    """With the sense: on revisit lookups (reading 2's definition), whether the right world cell is among the read's 5
    best distinct places by code similarity to the query's place (PLAN's diversified shortlist). Places are binned at
    the read's width; the true place is scored through the true positions."""
    est, xpre, xtrue = run_episode(parts, r, "full", True, None)
    pos, T = r["pos"], r["T"]
    code, calib = parts["code"], parts["calib"]
    world, off = calib.world.to(DEV), calib.offset.to(DEV)
    seen, hit, n = set(), 0, 0
    W_true = lambda t, k: (int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k]))
    for t in range(1, T + 1):
        for k in range(LM.MAP):
            seen.add(W_true(t - 1, k))
        prev = {W_true(t - 1, k) for k in range(LM.MAP)}
        ks = [k for k in range(LM.MAP) if k != LM.SELF and bool(world[k]) and W_true(t, k) not in prev and W_true(t, k) in seen]
        if not ks:
            continue
        Pe = (est.X[:t, None] + est.off[:t])[est.wm[:t]]                  # entries' places
        bins = torch.unique(torch.round(Pe / WIDTH), dim=0) * WIDTH        # distinct places
        q = xpre[t] + off[ks]                                            # [nq, 2]
        s_bins = code.sim(q[:, None], bins[None])                        # [nq, nb]
        true_place = xtrue[t] + off[ks]
        tb = torch.round(true_place / WIDTH) * WIDTH
        s_true = code.sim(q, tb)
        rank = (s_bins > s_true[:, None] + 1e-6).sum(1)
        present = ((bins[None] - tb[:, None]).abs().max(-1).values < 1e-3).any(1)
        hit += int(((rank < 5) & present).sum()); n += len(ks)
    return hit, n


def drifted_returns(parts, r, rng, light=None):
    """Reading 2b's events: outages of 8, 16 or 32 steps starting at movement steps that precede a return to old memory
    within 32 steps. At the return, drifted if the code is off by more than the read's width; recovered if it comes back
    within the width in the 16 steps after; each accepted correction scored by whether it moved toward the truth."""
    T, pos = r["T"], r["pos"]
    first_seen = {}
    for t in range(T + 1):
        for k in range(LM.MAP):
            first_seen.setdefault((int(pos[t][0]) + int(LM.KR[k]), int(pos[t][1]) + int(LM.KC[k])), t)
    events, t0 = [], 9
    Ls = [8, 16, 32]
    while t0 < T - 33:
        if (pos[t0] != pos[t0 - 1]).any():
            for s in range(t0 + 1, t0 + 33):
                if any(first_seen[(int(pos[s][0]) + int(LM.KR[k]), int(pos[s][1]) + int(LM.KC[k]))] < t0 - 1 for k in range(LM.MAP)
                       if abs(int(pos[s][0]) + int(LM.KR[k]) - int(pos[s - 1][0])) > 3 or abs(int(pos[s][1]) + int(LM.KC[k]) - int(pos[s - 1][1])) > 4):
                    events.append((t0, Ls[len(events) % 3], s)); t0 = s + 17
                    break
        t0 += 1
    out = []
    for mode in ("full", "none"):
        m = np.zeros(T + 1, bool)
        for a, L, s in events:
            m[a:a + L] = True
        est, xpre, xtrue = run_episode(parts, r, mode, True, m)
        X = est.X
        for a, L, s in events:
            err0 = float((xpre[s] - xtrue[s]).abs().max())
            rec = any(float((X[u] - xtrue[u]).abs().max()) < WIDTH for u in range(s, min(T + 1, s + 17)))
            accepted = [(t, dl, g) for t, f, dl, g, ok in est.log if a <= t <= s + 16 and ok]
            good = 0
            for t, dl, g in accepted:
                before = float((xpre[t] - xtrue[t]).norm()); after = float((xpre[t] + g * dl.to(DEV) - xtrue[t]).norm())
                good += int(after < before)
            lc = None
            if light is not None:                                        # since the returned-to cells were first seen
                fs = [first_seen[(int(pos[s][0]) + int(LM.KR[k]), int(pos[s][1]) + int(LM.KC[k]))] for k in range(LM.MAP)]
                fs = [f for f in fs if f < a - 1]
                lc = float(np.median([abs(light[s] - light[f]) for f in fs])) if fs else None
            out.append({"mode": mode, "t0": a, "L": L, "return": s, "return_in_outage": s < a + L,
                        "drift_at_return": err0, "drifted": err0 > WIDTH,
                        "recovered": rec, "accepted": len(accepted), "accepted_toward_truth": good,
                        "dark": (bool(light[s] < 0.5) if light is not None else None), "light_change": lc})
    return out


def gates(smoke=False):
    parts = load()
    res = {"code_smoke": smoke_code(parts["code"]), "calibration": parts["calib"].score(), "calibration_versions": parts["calib"].versions}
    data = train_data()
    held = data[270:272] if smoke else data[270:]
    panel = torch.load(LM.DATA / "tokens_panel.pt")[:2 if smoke else 120]
    cov = [revisit_coverage(parts, r) for r in (held[:1] + panel[:1] if smoke else held[:10] + panel[:12])]
    res["revisit_coverage"] = sum(h for h, _ in cov) / max(sum(n for _, n in cov), 1)
    rng = np.random.default_rng(20261125)
    ev = []
    for r in held:
        ev += drifted_returns(parts, r, rng)
    for r in panel:
        ev += [dict(e, panel=True) for e in drifted_returns(parts, r, rng, r["light"].numpy())]
    full = [e for e in ev if e["mode"] == "full" and e["drifted"]]
    none = [e for e in ev if e["mode"] == "none" and e["drifted"]]
    acc = sum(e["accepted"] for e in ev if e["mode"] == "full"); good = sum(e["accepted_toward_truth"] for e in ev if e["mode"] == "full")
    res["drifted_returns"] = {"full": len(full), "panel": sum(1 for e in full if e.get("panel"))}
    res["recall_full"] = float(np.mean([e["recovered"] for e in full])) if full else None
    res["recall_no_recognition"] = float(np.mean([e["recovered"] for e in none])) if none else None
    res["precision_full"] = good / acc if acc else None
    bands = {}
    for e in full:
        if e.get("light_change") is None:
            continue
        b = ("dark" if e["dark"] else "day") + "/" + ("<0.02" if e["light_change"] < 0.02 else "0.02-0.1" if e["light_change"] < 0.1 else ">=0.1")
        bands.setdefault(b, []).append(e["recovered"])
    res["recall_by_band"] = {b: {"n": len(v), "recall": float(np.mean(v))} for b, v in sorted(bands.items())}
    res["gates"] = {"1_code": res["code_smoke"]["pass"], "2_coverage": res["revisit_coverage"] >= 0.95,
                    "3_recognition": bool(res["precision_full"] is not None and res["precision_full"] >= 0.95 and (res["recall_full"] or 0) >= 0.35),
                    "4_drifted_returns": res["drifted_returns"]["panel"] >= 200}
    (HERE / ("gates_smoke.json" if smoke else "gates.json")).write_text(json.dumps({"summary": res, "events": ev}, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == "__main__":
    {"smoke": lambda: (stage_a(smoke=True), gates(smoke=True)), "stage_a": lambda: stage_a(), "gates": gates}[sys.argv[1]]()
