"""Decision 6's place memory, the world's side, tested whole. PLAN.md (2026-11-23): Markov localization on the agent's
own grid (a learned motion prior times the new view matched against the memory; label-free), entries of the world's
own tokens with time, confidence, source context and replay pointer, reversible writes, a cross-attention read keyed
by content, relative offset, age and confidence, dreams that advance by the motion prior with a scratch overlay.

    MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_located_memory.memory tokens
    ... memory localizer        # train the localizer (label-free), then score it (reading 2)
    ... memory smoke | train | eval 2000 | identity 2000 | inject 2000 | dream 2000
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import dataclasses, importlib, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import gates as GT, s1 as S

PR = importlib.import_module("artifacts.experiments.20261123_place_read.place_read")
S2, B, PM = PR.S2, PR.B, PR.PM
HERE = Path(__file__).resolve().parent
RUN, DATA = HERE / "run", HERE / "data"
DEV = S2.DEV
G, O, MAP, N = 96, 48, 63, 81                                           # the agent's own grid, its origin, map cells
KR, KC = PR.KR, PR.KC                                                   # map cell k -> (row - 3, col - 4) from the agent
NR = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]); NC = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1])   # 9 offsets
SHIFT = PR.SHIFT                                                        # the swap's wrong address (3 rows)
# one entry per grid cell: coarse token, time, confidence, lighting, replay pointer; fine token, time, crop position,
# coarse token at the fine write, confidence, lighting
CT, CTIME, CCONF, CLIGHT, CSRC = slice(0, 16), 16, 17, slice(18, 26), 26
FT, FTIME, FCROP, FCOARSE, FCONF, FLIGHT = slice(27, 91), 91, 92, slice(93, 109), 109, slice(110, 118)
CTRUE = slice(118, 120)                                                 # scoring only: the entry's true world cell
NE, FEAT, COMMIT = 120, 141, 0.5
COARSE = torch.zeros(NE, dtype=torch.bool); COARSE[:27] = True; COARSE[CTRUE] = True
LOC_DEV = torch.device(os.environ.get("LOC_DEVICE", "cuda"))           # the localizer may train on the CPU
FINE_FEATS = slice(31, 135)                                             # the read feature's fine part (the fine swap)
GEOM = slice(138, 141)                                                  # relative offset and current pose confidence


def blank(f):
    """The features of absent entries: content, presence and flags zero, geometry kept (no memory)."""
    out = torch.zeros_like(f)
    out[..., GEOM] = f[..., GEOM]
    return out


class Localizer(nn.Module):
    """The motion prior p(d | view, lighting, command) over 9 offsets, and the robust per-cell match of a new view
    against an entry (Laplace around the entry, scale and trust from the entry's age and the current lighting, mixed
    with a background Laplace), plus the relocation probability."""

    def __init__(self):
        super().__init__()
        self.prior = nn.Sequential(nn.Linear(MAP * 16 + 8 + 17, 256), nn.GELU(), nn.Linear(256, 9))
        self.lik = nn.Linear(5 + 8, 2)
        self.log_b_bg = nn.Parameter(torch.zeros(()))
        self.logit_eps = nn.Parameter(torch.tensor(-9.0))

    def kernel(self, p, v, a):
        """p [B, 81, 16], v [B, 8], a [B] the command -> log p(d) [B, 9]."""
        return self.prior(torch.cat([p[:, :MAP].flatten(1), v, F.one_hot(a.clamp(min=0), 17).float()], -1)).log_softmax(-1)

    def bg(self, c):
        return -c.abs().sum(-1) / self.log_b_bg.exp() - 16 * (math.log(2) + self.log_b_bg)

    def trust(self, age, present, v):
        """Each entry's (log scale, trust logit) from its age and the current lighting [..., 2]."""
        return self.lik(torch.cat([PR.age_features(age, present), v], -1))

    def score(self, c, e, age, present, v, h=None):
        """log p(c | entry) - log p_bg(c), zero where there is no entry. c, e [..., 16]; age, present [...]; v [..., 8];
        h: precomputed trust(age, present, v)."""
        h = self.trust(age, present, v) if h is None else h
        log_b, logit = h[..., 0], h[..., 1]
        match = -(c - e).abs().sum(-1) / log_b.exp() - 16 * (math.log(2) + log_b)
        bg = self.bg(c)
        mix = torch.logaddexp(F.logsigmoid(logit) + match, F.logsigmoid(-logit) + bg)
        return torch.where(present, mix - bg, torch.zeros_like(mix))


def empty(n, dev):
    x = torch.zeros(n, G, G, NE, device=dev)
    x[..., CTIME] = -1; x[..., FTIME] = -1; x[..., CTRUE] = -999
    return x


def shifted(x, dr, dc, fill):
    """x [..., G, G, C] moved by (dr, dc) cells, `fill` entering from the edge."""
    out = torch.empty_like(x).copy_(fill.expand_as(x) if torch.is_tensor(fill) else torch.full_like(x, fill))
    rs, rd = (slice(0, G - dr), slice(dr, G)) if dr >= 0 else (slice(-dr, G), slice(0, G + dr))
    cs, cd = (slice(0, G - dc), slice(dc, G)) if dc >= 0 else (slice(-dc, G), slice(0, G + dc))
    out[..., rd, cd, :] = x[..., rs, cs, :]
    return out


class Map:
    """Per stream: committed entries, the pending layer (writes made while the pose is doubtful), an optional dream
    overlay, the pose belief over the grid, the pose and its confidence, the last frame's coarse view (for the motion
    prior) and, for scoring only, each entry's true world cell."""

    def __init__(self, n, dev):
        self.main, self.pend = empty(n, dev), empty(n, dev)
        self.belief = torch.zeros(n, G, G, device=dev)
        self.pose = torch.full((n, 2), O, dtype=torch.long, device=dev)
        self.conf = torch.ones(n, device=dev)
        self.last_p, self.last_v = torch.zeros(n, N, 16, device=dev), torch.zeros(n, 8, device=dev)
        self.overlay = None
        self.reset(torch.ones(n, dtype=torch.bool, device=dev))

    def clone(self):
        c = Map.__new__(Map)
        for k, v in self.__dict__.items():
            setattr(c, k, v.clone() if torch.is_tensor(v) else v)
        return c

    def reset(self, m):
        if not m.any():
            return
        self.main[m] = empty(1, self.main.device)[0]; self.pend[m] = empty(1, self.main.device)[0]
        self.belief[m] = 0; self.belief[m, O, O] = 1
        self.pose[m] = O; self.conf[m] = 1

    def view(self):
        """(composed entries: pending over committed, dream overlay where neither holds the coarse cell; pending coarse,
        pending fine, dream flags)."""
        cp, fp = self.pend[..., CTIME] >= 0, self.pend[..., FTIME] >= 0
        if self.overlay is None and not (cp.any() or fp.any()):
            return self.main, cp, fp, torch.zeros_like(cp)                # nothing pending: the committed entries
        out = self.main.clone()
        coarse = COARSE.to(out.device)
        out = torch.where((cp[..., None] & coarse) | (fp[..., None] & ~coarse), self.pend, out)
        dream = torch.zeros_like(cp)
        if self.overlay is not None:
            dream = (out[..., CTIME] < 0) & (self.overlay[..., CTIME] >= 0)
            out = torch.where(dream[..., None], self.overlay, out)
        return out, cp, fp, dream


def grid_loglik(loc, ent, c, v, t, memory_term=True):
    """Sum over the new view's 63 map cells of loc.score under every grid hypothesis x (the agent at x) -> [B, G, G].
    memory_term False: only the previous frame's entries count (odometry only)."""
    Bn = len(c)
    age = t[:, None, None] - ent[..., CTIME]
    present = (ent[..., CTIME] >= 0) & ((age <= 1) if not memory_term else torch.ones_like(age, dtype=torch.bool))
    h = loc.trust(age, present, v[:, None, None].expand(Bn, G, G, 8))     # once per step: age and lighting only
    pad = torch.zeros(Bn, G + 6, G + 8, 19, device=c.device)
    pad[:, 3:3 + G, 4:4 + G] = torch.cat([ent[..., CT], present[..., None].float(), h], -1)
    out = torch.zeros(Bn, G, G, device=c.device)
    for k in range(MAP):
        e = pad[:, 3 + int(KR[k]):3 + int(KR[k]) + G, 4 + int(KC[k]):4 + int(KC[k]) + G]
        out += loc.score(c[:, None, None, k], e[..., :16], None, e[..., 16] > 0.5, None, h=e[..., 17:19])
    return out


def loglik_at(loc, ent, c, v, t, hyp, memory_term=True):
    """The same sum at given hypotheses hyp [B, H, 2] (gathered, vectorised over cells) -> [B, H]."""
    Bn, H = hyp.shape[:2]
    r = hyp[..., 0, None] + KR.to(hyp.device); cc = hyp[..., 1, None] + KC.to(hyp.device)          # [B, H, 63]
    inside = (r >= 0) & (r < G) & (cc >= 0) & (cc < G)
    e = ent[torch.arange(Bn, device=hyp.device)[:, None, None], r.clamp(0, G - 1), cc.clamp(0, G - 1)]
    age = t[:, None, None] - e[..., CTIME]
    present = inside & (e[..., CTIME] >= 0) & ((age <= 1) if not memory_term else torch.ones_like(age, dtype=torch.bool))
    return loc.score(c[:, None], e[..., CT], age, present, v[:, None, None].expand(Bn, H, MAP, 8)).sum(-1)


def window_hyp(center, W):
    d = torch.arange(-W, W + 1, device=center.device)
    off = torch.stack(torch.meshgrid(d, d, indexing="ij"), -1).reshape(-1, 2)
    return (center[:, None] + off[None]).clamp(0, G - 1)


def predict(loc, mp, logK):
    """The belief moved by the motion prior, mixed with relocation -> [B, G, G]."""
    K = logK.exp()
    pr = sum(K[:, i, None, None] * shifted(mp.belief[..., None], int(NR[i]), int(NC[i]), 0.0)[..., 0] for i in range(9))
    eps = torch.sigmoid(loc.logit_eps)
    return (1 - eps) * pr / pr.sum((1, 2), keepdim=True).clamp(min=1e-30) + eps / (G * G)


def argmax2(x):
    i = x.flatten(1).argmax(1)
    return torch.stack([i // G, i % G], 1), x.flatten(1).gather(1, i[:, None])[:, 0] / x.flatten(1).sum(1)


def read_features(ent, cp, fp, dream, at, conf_now, t):
    """Each map cell's 9 candidate entries around its address at + offset -> [B, 63, 9, FEAT]."""
    Bn = len(at)
    r = (at[:, 0, None, None] + KR.to(at.device)[None, :, None] + NR.to(at.device)[None, None, :]).clamp(0, G - 1)
    c = (at[:, 1, None, None] + KC.to(at.device)[None, :, None] + NC.to(at.device)[None, None, :]).clamp(0, G - 1)
    ok = ((at[:, 0, None, None] + KR.to(at.device)[None, :, None] + NR.to(at.device)[None, None, :]) == r) & \
         ((at[:, 1, None, None] + KC.to(at.device)[None, :, None] + NC.to(at.device)[None, None, :]) == c)
    bi = torch.arange(Bn, device=at.device)[:, None, None]
    e = ent[bi, r, c]                                                    # [B, 63, 9, NE]
    pc = (e[..., CTIME] >= 0) & ok; pf = (e[..., FTIME] >= 0) & ok
    tt = t[:, None, None].float()
    crop = F.one_hot(e[..., FCROP].long().clamp(0, 8), 9).float() * pf[..., None]
    off = torch.stack([NR, NC], -1).float().to(at.device).expand(Bn, MAP, 9, 2)
    f = torch.cat([e[..., CT] * pc[..., None], pc[..., None].float(), PR.age_features(tt - e[..., CTIME], pc),
                   (e[..., CCONF] * pc)[..., None], e[..., CLIGHT] * pc[..., None],
                   e[..., FT] * pf[..., None], pf[..., None].float(), PR.age_features(tt - e[..., FTIME], pf), crop,
                   e[..., FCOARSE] * pf[..., None], (e[..., FCONF] * pf)[..., None], e[..., FLIGHT] * pf[..., None],
                   (cp[bi, r, c] & pc)[..., None].float(), (fp[bi, r, c] & pf)[..., None].float(),
                   (dream[bi, r, c] & pc)[..., None].float(), off, conf_now[:, None, None, None].expand(Bn, MAP, 9, 1)], -1)
    assert f.shape[-1] == FEAT
    return f


def write(mp, pose, conf, p, v, g, fix, t, src, valid, truth=None, layer=None):
    """Write a frame's coarse tokens at pose + offset and its glimpsed map cells' fine tokens: to the pending layer
    where the pose is doubtful, else to the committed entries (or to `layer`, a dream overlay)."""
    Bn = len(pose)
    rr = pose[:, 0:1] + KR.to(pose.device); cc = pose[:, 1:2] + KC.to(pose.device)
    ok = (rr >= 0) & (rr < G) & (cc >= 0) & (cc < G) & valid[:, None]
    doubtful = conf < COMMIT
    for dst, sel in ((mp.main, ~doubtful), (mp.pend, doubtful)) if layer is None else ((layer, torch.ones_like(doubtful)),):
        m = ok & sel[:, None]
        if not m.any():
            continue
        bi = torch.arange(Bn, device=pose.device)[:, None].expand_as(rr)[m]
        r_, c_ = rr[m], cc[m]
        k = torch.arange(MAP, device=pose.device)[None].expand_as(rr)[m]
        dst[bi, r_, c_, CT] = p[bi, k].float()
        dst[bi, r_, c_, CTIME] = t[bi].float(); dst[bi, r_, c_, CCONF] = conf[bi]
        dst[bi, r_, c_, CLIGHT] = v[bi].float(); dst[bi, r_, c_, CSRC] = src[bi].float()
        if truth is not None:
            dst[bi, r_, c_, CTRUE] = truth[bi, k].float()
    win = M.window(fix)                                                  # [B, 9] screen cells of the glimpse
    for j in range(9):
        k = win[:, j]
        isg = (k < MAP) & valid
        kk = k.clamp(max=MAP - 1)
        r1, c1 = pose[:, 0] + KR.to(pose.device)[kk], pose[:, 1] + KC.to(pose.device)[kk]
        ok1 = isg & (r1 >= 0) & (r1 < G) & (c1 >= 0) & (c1 < G)
        for dst, sel in ((mp.main, ~doubtful), (mp.pend, doubtful)) if layer is None else ((layer, torch.ones_like(doubtful)),):
            m = ok1 & sel
            if not m.any():
                continue
            bi = torch.nonzero(m)[:, 0]
            dst[bi, r1[bi], c1[bi], FT] = g[bi, j].float(); dst[bi, r1[bi], c1[bi], FTIME] = t[bi].float()
            dst[bi, r1[bi], c1[bi], FCROP] = float(j); dst[bi, r1[bi], c1[bi], FCOARSE] = p[bi, kk[bi]].float()
            dst[bi, r1[bi], c1[bi], FCONF] = conf[bi]; dst[bi, r1[bi], c1[bi], FLIGHT] = v[bi].float()


def commit(mp, delta):
    """Move each stream's pending layer by its relocation `delta` [B, 2] (zero: none), and merge it into the committed
    entries for the streams now confident."""
    for b in torch.nonzero((delta != 0).any(1))[:, 0].tolist():
        fill = empty(1, mp.pend.device)[0, 0, 0]
        mp.pend[b] = shifted(mp.pend[b], int(delta[b, 0]), int(delta[b, 1]), fill)
    go = mp.conf >= COMMIT
    cp, fp = mp.pend[..., CTIME] >= 0, mp.pend[..., FTIME] >= 0
    go = go & (cp.flatten(1).any(1) | fp.flatten(1).any(1))
    if not go.any():
        return
    coarse = COARSE.to(mp.main.device)
    sel = go[:, None, None, None] & ((cp[..., None] & coarse) | (fp[..., None] & ~coarse))
    mp.main = torch.where(sel, mp.pend, mp.main)
    mp.pend[go] = empty(1, mp.pend.device)[0]


@torch.no_grad()
def advance(loc, mp, p, v, g, fix, a_prev, t, src, first, valid, truth=None, memory_term=True, inject=None, window=None,
            drift=None):
    """One real frame for every stream: reset at `first`; else predict the pose from the motion prior on the last frame
    and the command that led here, read the memory there for this frame (before it writes), observe it, update the
    belief (relocating where the evidence says so), write. Returns (read features [B, 63, 9, FEAT], predicted pose,
    its confidence). `inject` [B, 2]: a pose error added to the belief before this step (reading 3); with `drift`, the
    previous frame's writes move with it (a self-consistent drift). `window`: score only hypotheses within that many
    cells of the last pose (training)."""
    mp.reset(first & valid)
    live = valid & ~first
    if inject is not None:
        for b in torch.nonzero((inject != 0).any(1))[:, 0].tolist():
            dr, dc = int(inject[b, 0]), int(inject[b, 1])
            mp.belief[b] = shifted(mp.belief[b][..., None], dr, dc, 0.0)[..., 0]
            mp.pose[b] += inject[b]
            if drift is not None:
                fill = empty(1, mp.main.device)[0, 0, 0]
                for layer in (mp.main, mp.pend):
                    last = (layer[b, ..., CTIME] == float(t[b] - 1)) | (layer[b, ..., FTIME] == float(t[b] - 1))
                    moved = shifted(torch.where(last[..., None], layer[b], fill), dr, dc, fill)
                    keep = torch.where(last[..., None], fill.expand_as(layer[b]), layer[b])
                    layer[b] = torch.where((moved[..., CTIME] >= 0)[..., None], moved, keep)
    logK = loc.kernel(mp.last_p, mp.last_v, a_prev)
    pri = predict(loc, mp, logK)
    xhat, chat = argmax2(pri)
    xhat = torch.where(live[:, None], xhat, mp.pose); chat = torch.where(live, chat, torch.ones_like(chat))
    ent, cp, fp, dream = mp.view()
    feats = read_features(ent, cp, fp, dream, xhat, chat, t)
    if window is None:
        L = grid_loglik(loc, ent, p[:, :MAP].float(), v.float(), t.float(), memory_term)
    else:
        hyp = window_hyp(mp.pose, window)
        Lw = loglik_at(loc, ent, p[:, :MAP].float(), v.float(), t.float(), hyp, memory_term)
        L = torch.full((len(p), G, G), -1e30, device=p.device)
        L[torch.arange(len(p), device=p.device)[:, None], hyp[..., 0], hyp[..., 1]] = Lw
    post = pri * (L - L.flatten(1).max(1).values[:, None, None]).exp()
    post = post / post.sum((1, 2), keepdim=True)
    pose, conf = argmax2(post)
    jump = (pose - mp.pose).abs().max(1).values > 1
    delta = torch.where((jump & live)[:, None], pose - xhat, torch.zeros_like(pose))
    mp.belief = torch.where(live[:, None, None], post, mp.belief)
    mp.pose = torch.where(live[:, None], pose, mp.pose); mp.conf = torch.where(live, conf, mp.conf)
    commit(mp, delta)
    write(mp, mp.pose, mp.conf, p, v, g, fix, t, src, valid, truth)
    mp.last_p = torch.where(valid[:, None, None], p.float(), mp.last_p); mp.last_v = torch.where(valid[:, None], v.float(), mp.last_v)
    mp.last_L = L
    return feats, xhat, chat


def install(den):
    """The world's read: each map cell's input token queries its 9 candidate entries and a null slot (4 heads of 16);
    keys and values from an MLP of the whole entry; output zero-initialised, added to the cell's token."""
    dev = next(den.parameters()).device
    den.mem_kv = nn.Sequential(nn.Linear(FEAT, 128), nn.GELU(), nn.Linear(128, 128)).to(dev)
    den.mem_q = nn.Linear(den.W, 64).to(dev)
    den.mem_null = nn.Parameter(torch.zeros(128, device=dev))
    den.mem_out = nn.Linear(64, den.W).to(dev)
    nn.init.zeros_(den.mem_out.weight); nn.init.zeros_(den.mem_out.bias)
    den.place_input = torch.zeros(1, 1, MAP, 9, FEAT, device=dev)
    embed = den.embed

    def with_memory(x, memory, fix):
        tok, ang = embed(x, memory, fix)
        pin = den.place_input if den.place_input.shape[0] == tok.shape[0] else den.place_sub
        cell = tok[..., :MAP, :]
        q = den.mem_q(cell).unflatten(-1, (4, 16))                         # [B, T, 63, 4, 16]
        kv = den.mem_kv(pin.to(cell.dtype))                                # [B, T, 63, 9, 128]
        kv = torch.cat([kv, den.mem_null.to(kv.dtype).expand(*kv.shape[:-2], 1, 128)], -2)
        k, v = kv.unflatten(-1, (2, 4, 16)).unbind(-3)                     # [B, T, 63, 10, 4, 16] each
        w = (torch.einsum("btchd,btcnhd->btchn", q, k) / 4.0).softmax(-1)
        o = torch.einsum("btchn,btcnhd->btchd", w, v).flatten(-2)
        return torch.cat([cell + den.mem_out(o).to(cell.dtype), tok[..., MAP:, :]], -2), ang
    den.embed = with_memory
    loss = M.shortcut_loss

    def shortcut_loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
        d.place_sub = d.place_input[bootstrap]                          # the bootstrap target's row subset
        return loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels, noise)
    M.shortcut_loss = shortcut_loss


def mem_params(den):
    return [p for m in (den.mem_kv, den.mem_q, den.mem_out) for p in m.parameters()] + [den.mem_null]


# ---------------------------------------------------------------- the localizer: data, training, scoring (reading 2)

@torch.no_grad()
def tokens():
    """Coarse tokens, lighting, glimpse tokens, commands of 300 TRAIN episodes (localizer training) and of the
    evaluation panel's 120 DEV episodes, with the panel's simulator positions and light (scoring only)."""
    model, cfg = S2.load()
    DATA.mkdir(parents=True, exist_ok=True)
    split = D.load_split(cfg)
    rng = np.random.default_rng(20261123)
    tr = [split["train"][i] for i in rng.choice(len(split["train"]), 300, replace=False)]
    Ld = dict(np.load(S.LABELS / "s0_dev_labels.npz")); so = np.r_[0, np.cumsum(Ld["ep_T"] + 1)]
    dl = S.episodes(Ld)
    panel = np.flatnonzero(Ld["ep_eps"] < 0.99)[:120]                   # place_read's evaluation episodes
    for name, eps in (("train", [(e, None) for e in tr]), ("panel", [(dl[e], e) for e in panel])):
        out = []
        for ep, e in eps:
            f = D.RP.episode_fields(ep.shard, ep.slot)
            fr = f["observations"].to(DEV)
            rng_f = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
            fix = torch.as_tensor(np.array([D.fixation_prior(rng_f, ()) for _ in range(ep.T + 1)]), device=DEV)
            ps, vs, gs = [], [], []
            for a in range(0, ep.T + 1, 512):
                with TR.autocast(DEV, cfg):
                    P_ = model.perception(fr[a:a + 512][None], fix[a:a + 512][None])
                ps.append(P_.p[0].half().cpu()); gs.append(P_.g[0].half().cpu()); vs.append(P_.v[0].half().cpu())
            acts = f["actions_taken"].numpy().astype(np.int64)
            rec = {"p": torch.cat(ps), "g": torch.cat(gs), "v": torch.cat(vs), "fix": fix.cpu(),
                   "prev_action": torch.as_tensor(np.r_[-1, acts]), "T": ep.T}
            if e is not None:
                rec.update(e=int(e), pos=torch.as_tensor(Ld["s_pos"][so[e]:so[e] + ep.T + 1].astype(np.int64)),
                           light=torch.as_tensor(Ld["s_light"][so[e]:so[e] + ep.T + 1]))
            out.append(rec)
        torch.save(out, DATA / f"tokens_{name}.pt")
        print(name, len(out), "episodes", sum(r["T"] for r in out), "steps", flush=True)


def truth_cells(pos):
    """True world cell of each map cell [T + 1, 63, 2] (scoring only)."""
    return torch.stack([pos[:, 0:1] + KR.to(pos.device), pos[:, 1:2] + KC.to(pos.device)], -1)


def train_localizer(passes=4, batch=16, chunk=32):
    """Label-free: maximise the filter's marginal likelihood of each new view (the past belief detached). Gradients
    reach the motion prior through the predicted belief at the 9 hypotheses next to the last pose, and the match
    through those hypotheses' cell scores; relocation through the whole-grid term."""
    data = torch.load(DATA / "tokens_train.pt")
    dev = DEV = LOC_DEV
    loc = Localizer().to(dev)
    opt = torch.optim.Adam(loc.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    log = []
    for ps in range(passes):
        by_len = np.argsort([r["T"] for r in data], kind="stable")             # batches of similar length, shuffled
        batches = [by_len[i:i + batch] for i in range(0, len(by_len), batch)]
        tot, cnt, t0 = 0.0, 0, time.time()
        for bi, idx in enumerate(rng.permutation(len(batches))):
            eps = [data[i] for i in batches[idx]]
            Bn, Tm = len(eps), max(r["T"] for r in eps)
            mp = Map(Bn, dev)
            loss_acc, n_acc = 0.0, 0
            for t in range(Tm + 1):
                valid = torch.tensor([t <= r["T"] for r in eps], device=dev)
                get = lambda k: torch.stack([r[k][min(t, r["T"])] for r in eps]).to(dev)
                p, v, g, fix, a = get("p").float(), get("v").float(), get("g").float(), get("fix"), get("prev_action")
                first = torch.full((Bn,), t == 0, device=dev)
                tt = torch.full((Bn,), t, device=dev)
                if t > 0:
                    live = valid
                    logK = loc.kernel(mp.last_p, mp.last_v, a)
                    with torch.no_grad():
                        ent, cp, fp, dream = mp.view()
                        prev = mp.belief.clone()
                    # the 9 hypotheses next to the last pose: prior mass and cell scores, with gradients
                    hyp = (mp.pose[:, None] + torch.stack([NR, NC], -1).to(dev)[None]).clamp(0, G - 1)   # [B, 9, 2]
                    bi_ = torch.arange(Bn, device=dev)[:, None]
                    K = logK.exp()
                    pri = torch.zeros(Bn, 9, device=dev)
                    for i in range(9):                                       # prior(x) = sum_d K(d) b(x - d)
                        src_r = (hyp[..., 0] - int(NR[i])).clamp(0, G - 1); src_c = (hyp[..., 1] - int(NC[i])).clamp(0, G - 1)
                        pri = pri + K[:, i:i + 1] * prev[bi_, src_r, src_c]
                    eps_ = torch.sigmoid(loc.logit_eps)
                    Lh = loglik_at(loc, ent, p[:, :MAP], v, tt.float(), hyp)
                with torch.no_grad():
                    advance(loc, mp, p, v, g, fix, a, tt, tt, first, valid, window=12)
                if t > 0:
                    Lfull = mp.last_L                                        # the window's scores (-1e30 outside)
                    Mx = torch.maximum(Lfull.flatten(1).max(1).values, Lh.max(1).values).detach()
                    local = ((1 - eps_) * pri * (Lh - Mx[:, None]).exp()).sum(1)
                    glob = eps_ * (Lfull - Mx[:, None, None]).exp().sum((1, 2)) / (G * G)
                    ll = (local + glob).clamp(min=1e-30).log() + Mx + loc.bg(p[:, :MAP]).sum(-1)
                    loss_acc = loss_acc - (ll * live).sum(); n_acc += int(live.sum())
                if (t % chunk == chunk - 1 or t == Tm) and n_acc:
                    opt.zero_grad(); (loss_acc / n_acc).backward(); opt.step()
                    tot += float(loss_acc.detach()); cnt += n_acc; loss_acc, n_acc = 0.0, 0
            print(f"pass {ps} batch {bi} nll/frame {tot / max(cnt, 1):.2f} {time.time() - t0:.0f}s", flush=True)
        log.append({"pass": ps, "nll_per_frame": tot / cnt, "eps": float(torch.sigmoid(loc.logit_eps)),
                    "b_bg": float(loc.log_b_bg.exp())})
        torch.save(loc.state_dict(), RUN / "localizer.pt")
    (RUN / "localizer_log.json").write_text(json.dumps(log, indent=1) + "\n")


@torch.no_grad()
def score_localizer(memory_term=True, tag="full"):
    """Reading 2 on the panel's 120 episodes (simulator positions score only): displacement accuracy before the frame
    (prior) and after it (posterior) on movement commands, moved / blocked / dark; exact relative pose over 16 steps;
    correct revisit lookups (the entry at the inferred address is the same world cell), by the prior's address."""
    DEV = LOC_DEV
    loc = Localizer().to(DEV); loc.load_state_dict(torch.load(RUN / "localizer.pt", map_location=DEV)); loc.eval()
    data = torch.load(DATA / "tokens_panel.pt")
    st = {k: [] for k in ("prior_ok", "post_ok", "moved", "move_cmd", "dark", "rel16_ok", "lookup_ok", "lookup_n")}
    for r in data:
        mp = Map(1, DEV)
        T, pos = r["T"], r["pos"].to(DEV)
        tc = truth_cells(pos)
        poses, xh = [], []
        lk_ok = lk_n = 0
        seen = {}
        for t in range(T + 1):
            one = lambda k: r[k][t][None].to(DEV)
            first = torch.tensor([t == 0], device=DEV); valid = torch.tensor([True], device=DEV)
            tt = torch.tensor([t], device=DEV)
            pose_before = mp.pose.clone()
            truth_before = mp.view()[0][0, ..., CTRUE].long()             # [G, G, 2]: what each address holds now
            feats, xhat, chat = advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"),
                                        one("prev_action"), tt, tt, first, valid, truth=tc[t][None], memory_term=memory_term)
            if t > 0 and (mp.pose[0] - pose_before[0]).abs().max() > 1:   # a natural relocalisation
                frac, n_ = aligned(mp, mp.pose[0], r["pos"][t].numpy(), max(0, t - 64), t, doubtful=True)
                if n_:
                    st.setdefault("jumps", []).append((frac, n_))
                st.setdefault("n_jumps", []).append(1)
            if t > 0:
                a = int(r["prev_action"][t]); dtrue = (pos[t] - pos[t - 1]).tolist()
                if a in GT.MOVE:
                    st["move_cmd"].append(1); st["moved"].append(int(dtrue != [0, 0]))
                    st["prior_ok"].append(int((xhat[0] - pose_before[0]).tolist() == dtrue))
                    st["post_ok"].append(int((mp.pose[0] - pose_before[0]).tolist() == dtrue))
                    st["dark"].append(int(float(r["light"][t]) < 0.5))
                # lookups: map cells entering view at t whose world cell was seen before t, through the prior's address
                prev = {(int(pos[t - 1][0]) + int(KR[k]), int(pos[t - 1][1]) + int(KC[k])) for k in range(MAP)}
                for k in range(MAP):
                    w = (int(pos[t][0]) + int(KR[k]), int(pos[t][1]) + int(KC[k]))
                    if w in prev or w not in seen:
                        continue
                    lk_n += 1
                    rr, cc = int(xhat[0, 0]) + int(KR[k]), int(xhat[0, 1]) + int(KC[k])
                    if 0 <= rr < G and 0 <= cc < G and tuple(truth_before[rr, cc].tolist()) == w:
                        lk_ok += 1
            for k in range(MAP):
                seen[(int(pos[t][0]) + int(KR[k]), int(pos[t][1]) + int(KC[k]))] = t
            poses.append(mp.pose[0].clone())
        P_ = torch.stack(poses)
        rel_inf, rel_true = P_[16:] - P_[:-16], pos[16:] - pos[:-16]
        st["rel16_ok"] += (rel_inf == rel_true).all(1).int().tolist()
        st["lookup_ok"].append(lk_ok); st["lookup_n"].append(lk_n)
    a = {k: np.array(v) for k, v in st.items() if k not in ("jumps", "n_jumps")}
    mv, dk = a["moved"].astype(bool), a["dark"].astype(bool)
    res = {"tag": tag, "movement_commands": int(len(mv)),
           "prior": {"all": a["prior_ok"].mean(), "moved": a["prior_ok"][mv].mean(), "blocked": a["prior_ok"][~mv].mean(),
                     "dark": a["prior_ok"][dk].mean()},
           "posterior": {"all": a["post_ok"].mean(), "moved": a["post_ok"][mv].mean(), "blocked": a["post_ok"][~mv].mean(),
                         "dark": a["post_ok"][dk].mean()},
           "relocalisations": int(len(st.get("n_jumps", []))),
           "pending_aligned_after_relocalisation": (float(sum(f * n for f, n in st.get("jumps", [])) / max(sum(n for _, n in st.get("jumps", [])), 1))
                                                    if st.get("jumps") else None),
           "pending_entries_at_relocalisations": int(sum(n for _, n in st.get("jumps", []))),
           "relative_pose_exact_16": a["rel16_ok"].mean(), "revisit_lookups": int(a["lookup_n"].sum()),
           "lookup_correct": a["lookup_ok"].sum() / max(a["lookup_n"].sum(), 1)}
    res = json.loads(json.dumps(res, default=float))
    (HERE / f"localizer_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)
    return res



# ---------------------------------------------------------------- the read's fine-tune (stage 1's recipe)

def load_localizer(dev=DEV):
    loc = Localizer().to(dev); loc.load_state_dict(torch.load(HERE / "run" / "localizer.pt", map_location=dev)); loc.eval()
    return loc.requires_grad_(False)


def chunk_memory(loc, mp, p, v, g, fix, prev_action, ts, first, valid, P):
    """Read features before each frame [n, L, 63, 9, FEAT], from the localizer's own poses; writes every valid frame;
    returns the map after frame L - P - 1 (the next chunk's prefix repeats this chunk's last P frames)."""
    n, L = first.shape
    feats = torch.zeros(n, L, MAP, 9, FEAT, device=p.device)
    snap, confs = mp, []
    for j in range(L):
        f, _, chat = advance(loc, mp, p[:, j], v[:, j], g[:, j], fix[:, j], prev_action[:, j], ts[:, j], ts[:, j],
                             first[:, j], valid[:, j])
        feats[:, j] = f * valid[:, j, None, None, None]
        confs.append(chat[valid[:, j] & ~first[:, j]])
        if j == L - P - 1:
            snap = mp.clone()
    return feats, snap, torch.cat(confs)


def train(updates=2000, saves=(1000, 2000)):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    RUN.mkdir(parents=True, exist_ok=True)
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
    model.gaze_forecast = None                                          # not trained here (stage 1's recipe)
    model.train()
    den.checkpoint = cfg["p1"]["activation_checkpointing"]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    split = D.load_split(cfg)
    streams = PR.PosStreams(split["train"], d["streams"], P, seed=ck["world_seed"], targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    carry, cache, record = ck["carry"], ck["cache"].to(DEV), ck["record"].to(DEV)
    loc = load_localizer()
    mp = Map(d["streams"], DEV)
    base, micro = ck["state"]["update"], d["accumulate_at_64"]
    log, t0, losses = [], time.time(), []
    for u in range(1, updates + 1):
        chunk = streams.next(64)
        fr = torch.as_tensor(chunk["frames"], device=DEV)
        fx = torch.as_tensor(chunk["fixation"], device=DEV)
        with torch.no_grad(), TR.autocast(DEV, cfg):
            Pc = model.perception(fr, fx)
        feats, mp, confs = chunk_memory(loc, mp, Pc.p.float(), Pc.v.float(), Pc.g.float(), fx,
                                        torch.as_tensor(chunk["prev_action"], device=DEV), torch.as_tensor(chunk["ts"], device=DEV),
                                        torch.as_tensor(chunk["first"], device=DEV), torch.as_tensor(chunk["valid"], device=DEV), P)
        boot = TR.bootstrap_rows(cfg, ck["world_seed"], base + u)
        for grp in opt.param_groups:
            grp["lr"] = 1e-4 * min(1.0, u / 100)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), micro):
            den.place_input = feats[rows]
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
                        "pose_conf_mean": float(confs.mean()) if len(confs) else None, "sec_per_update": (time.time() - t0) / u})
            print(json.dumps(log[-1]), flush=True)
            (RUN / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
        if u in saves:
            torch.save({"read": {k: v for k, v in den.state_dict().items()}, "update": u}, RUN / f"denoiser_{u}.pt")


# ---------------------------------------------------------------- readings 1 and 3 (one-step dreams from real history)

@torch.no_grad()
def one_step(model, cfg, R, enc, sub, rows, feat_of, variants, seed=20261123):
    """place_read.evaluate's one-step dreams: P context frames then the next frame, 4 draws, terrain read. feat_of(v,
    e, t, drawn) -> the place features of frame t under variant v [63, 9, FEAT] (drawn: t is the frame being drawn,
    else a context frame) -> drawn terrain {v: {(e, t): [4, 7, 9]}}."""
    den = model.denoiser
    P = cfg["data"]["context_prefix"]
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    MF, NB = 4, 16
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
            g = torch.Generator(device=DEV).manual_seed(seed + bi)
            cache = den.new_cache()
            with TR.autocast(DEV, cfg):
                for dt in range(1 - P, 1):
                    den.place_input = rep(torch.stack([feat_of(v, e, t + dt, False) for e, t in batch]))[:, None]
                    cache = M.context_pass(den, tuple(rep(at(k, dt)).float()[:, None] for k in ("p", "g", "v")),
                                           rep(at("prev_action", dt))[:, None],
                                           S.cond(model.belief.memory(rep(at("bp", dt)).float())[:, None], rep(rin[:, P - 1 + dt])[:, None]),
                                           rep(at("fix", dt))[:, None], cache, cfg, g)
                den.place_input = rep(torch.stack([feat_of(v, e, t + 1, True) for e, t in batch]))[:, None]
                parts = M.sample(den, dims, rep(at("action", 0))[:, None],
                                 S.cond(model.belief.memory(rep(at("bp", 1)).float())[:, None], rep(rs)[:, None]),
                                 rep(at("fix", 1))[:, None], cache, cfg, g)
            te = B.terrain(R, parts[0].float(), parts[2].float())[:, 0]
            for b, (e, t) in enumerate(batch):
                drawn[v][(e, t)] = te[MF * b:MF * (b + 1)]
        if bi % (NB * 20) == 0:
            print("batch", bi // NB, flush=True)
    return drawn


def load_world(update):
    model, cfg = S2.load(); S2.EV.guard()
    for prm in model.parameters():
        prm.requires_grad_(False)
    install(model.denoiser)
    if update:
        st = torch.load(RUN / f"denoiser_{update}.pt", map_location=DEV, weights_only=False)["read"]
        model.denoiser.load_state_dict(st)
    model.eval()
    return model, cfg


def panel(model, cfg, smoke=False):
    """place_read.evaluate's episodes, starts and encodings."""
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
    return L, so, eps, rows, sub, enc


def entering(L, so, e, t):
    """[63] map cells entering view at t + 1 (not in view at t), by simulator positions (which cells a variant swaps)."""
    p0, p1 = L["s_pos"][so[e] + t].astype(int), L["s_pos"][so[e] + t + 1].astype(int)
    return torch.tensor([bool(abs(p1[0] + int(KR[k]) - p0[0]) > 3 or abs(p1[1] + int(KC[k]) - p0[1]) > 4) for k in range(MAP)], device=DEV)


@torch.no_grad()
def evaluate(update, smoke=False):
    """Reading 1: stage 1's cells with inferred addresses; variants correct / empty / wrong (the entry 3 rows away) /
    fine_swap (the fine part from the wrong address, coarse kept)."""
    model, cfg = load_world(update)
    R = S2.load_readers(model, cfg)
    loc = load_localizer()
    L, so, eps, rows, sub, enc = panel(model, cfg, smoke)
    mem = {}
    for e in eps:
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)
        mem[e] = episode_memory(loc, enc[sub[e]], pos)
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    new = {(e, t): entering(L, so, e, t) for e, t in rows}

    def feat_of(v, e, t, drawn):
        f = mem[e]["feat"][t]
        if v == "correct" or not drawn:
            return f
        m = new[(e, t - 1)][:, None, None]
        if v == "empty":
            return torch.where(m, blank(f), f)
        if v == "wrong":
            return torch.where(m, mem[e]["wrong"][t], f)
        alt = f.clone(); alt[..., FINE_FEATS] = mem[e]["wrong"][t][..., FINE_FEATS]
        return torch.where(m, alt, f)
    variants = ("correct", "empty", "wrong", "fine_swap") if update else ("correct",)
    drawn = one_step(model, cfg, R, enc, sub, rows, feat_of, variants)
    res = {"update": update, "variants": {}}
    for v in variants:
        z = PR.cells(L, so, rows, drawn[v], rterr)
        np.savez_compressed(HERE / f"cells_{update}_{v}{'_smoke' if smoke else ''}.npz", **z)
        res["variants"][v] = PM.summarise(z, np.random.default_rng(0), reps=300 if smoke else 1000)
        print(v, json.dumps(res["variants"][v]["matched_minus_first"])[:600], flush=True)
    res["pose_conf_mean"] = float(np.mean([float(mem[e]["conf"][1:].mean()) for e in eps]))
    (HERE / f"eval_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")


def episode_memory(loc, en, pos, inject_at=None, delta=(4, 0), memory_term=True, stop=None):
    """One real episode through the localizer: read features before each frame at the predicted address [T + 1, 63, 9,
    FEAT], the same at the wrong address (+ SHIFT, read before the frame writes), poses, predicted confidences;
    optionally a self-consistent drift injected before frame `inject_at`. Simulator positions only label the entries."""
    T1 = len(en["p"])
    mp = Map(1, DEV)
    tc = truth_cells(torch.as_tensor(pos, device=DEV))
    out = {"feat": torch.zeros(T1, MAP, 9, FEAT, device=DEV), "wrong": torch.zeros(T1, MAP, 9, FEAT, device=DEV),
           "pose": torch.zeros(T1, 2, dtype=torch.long), "xhat": torch.zeros(T1, 2, dtype=torch.long), "conf": torch.zeros(T1)}
    sh = torch.tensor([SHIFT], device=DEV)
    for t in range(T1):
        one = lambda k: en[k][t][None].to(DEV)
        tt = torch.tensor([t], device=DEV)
        inj = torch.tensor([delta], device=DEV) if inject_at is not None and t == inject_at else None
        before = mp.clone() if t > 0 else None
        f, xhat, chat = advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"),
                                torch.tensor([int(en["prev_action"][t])], device=DEV), tt, tt,
                                torch.tensor([t == 0], device=DEV), torch.tensor([True], device=DEV), truth=tc[t][None],
                                memory_term=memory_term, inject=inj, drift=inj)
        if before is not None:
            ent, cp, fp, dream = before.view()
            out["wrong"][t] = read_features(ent, cp, fp, dream, xhat + sh, chat, tt)[0]
        out["feat"][t] = f[0]; out["xhat"][t] = xhat[0].cpu(); out["pose"][t] = mp.pose[0].cpu(); out["conf"][t] = float(chat[0])
        if stop is not None and t == stop:
            break
    out["map"] = mp
    return out


def aligned(mp, pose_now, pos_now, since, until, doubtful=None):
    """Of the entries written in frames [since, until) (doubtful: only those written with confidence < COMMIT), the
    share sitting where the current pose says their true world cell is (scoring only)."""
    ent = mp.view()[0][0]
    tm = ent[..., CTIME]
    m = (tm >= since) & (tm < until)
    if doubtful is not None:
        m = m & ((ent[..., CCONF] < COMMIT) == doubtful)
    idx = torch.nonzero(m)
    if not len(idx):
        return None, 0
    want = ent[idx[:, 0], idx[:, 1]][:, CTRUE].long() - torch.as_tensor(pos_now, device=idx.device) + pose_now.to(idx.device)
    return float((want == idx).all(1).float().mean()), int(len(idx))


@torch.no_grad()
def inject(update, smoke=False):
    """Reading 3: a self-consistent drift of 4 rows injected at movement steps that precede a return to old memory
    within 32 steps; three localizers on the same trajectories (oracle = no injection, full rule, odometry only);
    recovery of the exact relative pose; the following view rendered from each; alignment of the drifted writes."""
    model, cfg = load_world(update)
    R = S2.load_readers(model, cfg)
    loc = load_localizer()
    L, so, eps, rows, sub, enc = panel(model, cfg, smoke)
    rng = np.random.default_rng(20261124)
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    events, res_ev = [], []
    for e in eps:
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)
        last = {}
        old_return = np.full(T + 1, -1)
        for t in range(T + 1):                                          # first sighting times per world cell
            for k in range(MAP):
                w = (int(pos[t][0] + KR[k]), int(pos[t][1] + KC[k]))
                last.setdefault(w, t)
        cand = []
        for t0 in rng.permutation(np.arange(9, max(9, T - 33))).tolist():
            if len(cand) >= 1:
                break
            if not (pos[t0] != pos[t0 - 1]).any():
                continue
            for s in range(t0 + 1, t0 + 33):                            # a later view showing a cell first seen before t0 - 1
                if any(last[(int(pos[s][0] + KR[k]), int(pos[s][1] + KC[k]))] < t0 - 1 for k in range(MAP)) and \
                        any((abs(pos[s][0] + KR[k] - pos[s - 1][0]) > 3 or abs(pos[s][1] + KC[k] - pos[s - 1][1]) > 4)
                            and last[(int(pos[s][0] + KR[k]), int(pos[s][1] + KC[k]))] < t0 - 1 for k in range(MAP)):
                    cand.append((t0, s)); break
        for t0, s in cand:
            events.append((int(e), t0, s))
    print("events", len(events), flush=True)
    mems = {}
    for e, t0, s_ret in events:
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)
        stop = min(T, t0 + 32)
        run = {"oracle": episode_memory(loc, enc[sub[e]], pos, stop=stop),
               "full": episode_memory(loc, enc[sub[e]], pos, inject_at=t0, stop=stop),
               "odometry": episode_memory(loc, enc[sub[e]], pos, inject_at=t0, memory_term=False, stop=stop)}
        ev = {"e": e, "t0": t0, "return": s_ret}
        for k, m in run.items():
            rel = m["pose"][t0:stop + 1] - m["pose"][t0 - 1]
            true = torch.as_tensor(pos[t0:stop + 1] - pos[t0 - 1])
            ok = (rel == true).all(1)
            ev[f"{k}_recovered_at"] = int(torch.nonzero(ok)[0, 0]) if ok.any() else None
            ev[f"{k}_drifted_writes_aligned"] = aligned(m["map"], m["pose"][stop], pos[stop], t0, stop, doubtful=False)
        res_ev.append(ev)
        mems[(e, t0)] = run
    # the following view: per episode its first event; the frame drawn is the first return step after the full rule
    # recovered (or the return step), rendered from each localizer's own memory
    follow, run_of = [], {}
    for ev in res_ev:
        if ev["e"] in run_of:
            continue
        r = ev["full_recovered_at"]
        tr = max(ev["return"], ev["t0"] + (r if r is not None else 0))
        if tr <= min(int(L["ep_T"][ev["e"]]), ev["t0"] + 32):
            follow.append((ev["e"], tr - 1)); run_of[ev["e"]] = mems[(ev["e"], ev["t0"])]
    new = {(e, t): entering(L, so, e, t) for e, t in follow}

    def feat_of(v, e, t, drawn):
        run = run_of[e]
        if v == "empty":
            f = run["oracle"]["feat"][t]
            return torch.where(new[(e, t - 1)][:, None, None], blank(f), f) if drawn else f
        return run[v]["feat"][t]
    rows_f = follow
    drawn = one_step(model, cfg, R, enc, sub, rows_f, feat_of, ("oracle", "full", "odometry", "empty")) if rows_f else {}
    out = {"update": update, "events": len(res_ev),
           "recovered_within_16": {k: float(np.mean([ev[f"{k}_recovered_at"] is not None and ev[f"{k}_recovered_at"] <= 16 for ev in res_ev]))
                                   for k in ("oracle", "full", "odometry")} if res_ev else None}
    for k in ("full", "odometry"):
        vals = [ev[f"{k}_drifted_writes_aligned"][0] for ev in res_ev if ev[f"{k}_drifted_writes_aligned"][0] is not None]
        out[f"{k}_drifted_writes_aligned_mean"] = float(np.mean(vals)) if vals else None
    if rows_f:
        for v in ("oracle", "full", "odometry", "empty"):
            z = PR.cells(L, so, rows_f, drawn[v], rterr)
            sel = (z["context"] == "surprising") & (z["group"] != "first")
            out.setdefault("following_view_surprising_revisits", {})[v] = {"cells": int(sel.sum()), "agree": float(z["agree"][sel].mean()) if sel.any() else None}
        f = out["following_view_surprising_revisits"]
        if f["oracle"]["agree"] is not None and f["oracle"]["agree"] > f["empty"]["agree"]:
            out["full_share_of_oracle_gain"] = (f["full"]["agree"] - f["empty"]["agree"]) / (f["oracle"]["agree"] - f["empty"]["agree"])
            out["odometry_share_of_oracle_gain"] = (f["odometry"]["agree"] - f["empty"]["agree"]) / (f["oracle"]["agree"] - f["empty"]["agree"])
    (HERE / f"inject_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps({"summary": out, "events": res_ev}, indent=1, default=str) + "\n")
    print(json.dumps(out), flush=True)


@torch.no_grad()
def dream(update, smoke=False, H=16, MF=4):
    """Reading 4: 16-step self-fed dreams under the recorded commands and fixations from real starts (s2b.rollout's
    free mode: the world's own sampled reward and termination). Arms on the same seeds: no memory (zero features),
    memory with oracle dream poses, memory with inferred dream poses (the motion prior on the dreamed frame and the
    command). The memory is copied at the start; the dream writes an overlay. Scored on world cells entering view (by
    the true pose) that the real prefix verified and that are unchanged in the real frame of that step."""
    model, cfg = load_world(update)
    R = S2.load_readers(model, cfg)
    loc = load_localizer()
    L, so, eps, rows, sub, enc = panel(model, cfg, smoke)
    den, P = model.denoiser, cfg["data"]["context_prefix"]
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    rterr = {e: PM.real_terrain(R, enc[sub[e]]) for e in eps}
    ARMS = ("empty", "oracle", "inferred")
    Bn = len(ARMS) * MF
    arm_of = torch.arange(Bn, device=DEV) // MF
    rng = np.random.default_rng(20261125)
    acc = {}
    motion = []
    for e in eps:
        T = int(L["ep_T"][e]); pos = L["s_pos"][so[e]:so[e] + T + 1].astype(np.int64)
        starts = list(range(P + 8, T - H))
        if not starts:
            continue
        chosen = sorted(rng.choice(starts, min(1 if smoke else 2, len(starts)), replace=False).tolist())
        en, i = enc[sub[e]], sub[e]
        mp = Map(1, DEV); tc = truth_cells(torch.as_tensor(pos, device=DEV))
        feats_real, snaps = [], {}
        for t in range(max(chosen) + 1):
            one = lambda k: en[k][t][None].to(DEV)
            tt = torch.tensor([t], device=DEV)
            f, _, _ = advance(loc, mp, one("p").float(), one("v").float(), one("g").float(), one("fix"),
                              torch.tensor([int(en["prev_action"][t])], device=DEV), tt, tt, torch.tensor([t == 0], device=DEV),
                              torch.tensor([True], device=DEV), truth=tc[t][None])
            feats_real.append(f[0])
            if t in chosen:
                snaps[t] = mp.clone()
        for t in chosen:
            rep = lambda x: x.expand(Bn, *x.shape[1:]).clone() if x.shape[0] == 1 else x.repeat_interleave(Bn, 0)
            # the real prefix's belief, record and denoiser cache (s2b.rollout)
            pp = M.Percept(*(en[k][:t + 1].to(DEV).float()[None] for k in ("p", "g", "v")), en["fix"][:t + 1].to(DEV)[None])
            with TR.autocast(DEV, cfg):
                reads, _, last = S.history(model, pp.p)
                y, c = model.belief(model.belief.tokens(S.embed(model, pp, reads), en["reward"][:t + 1].to(DEV)[None],
                                    en["terminated"][:t + 1].to(DEV)[None], en["action"][:t + 1].to(DEV)[None]),
                                    model.belief.initial(1, DEV))
            carry = tuple(type(c[l])(*(rep(getattr(c[l], f.name)) for f in dataclasses.fields(c[l]))) for l in range(len(c)))
            bp, rs = rep(y[:, -1].float()), rep(last)
            rm = rs
            rin = S.window(model, enc, [(i, t)], P - 1)
            g = torch.Generator(device=DEV).manual_seed(20261125 + 7919 * int(e) + t)
            cache = den.new_cache()
            noarm = (arm_of == 0)[:, None, None, None]
            at = lambda key, dt: en[key][t + dt][None].to(DEV)
            with TR.autocast(DEV, cfg):
                for dt in range(1 - P, 1):
                    fr = feats_real[t + dt][None].expand(Bn, -1, -1, -1)
                    den.place_input = torch.where(noarm, blank(fr), fr)[:, None]
                    cache = M.context_pass(den, tuple(rep(at(k, dt)).float()[:, None] for k in ("p", "g", "v")),
                                           rep(at("prev_action", dt))[:, None],
                                           S.cond(model.belief.memory(rep(at("bp", dt)).float())[:, None], rep(rin[:, P - 1 + dt])[:, None]),
                                           rep(at("fix", dt))[:, None], cache, cfg, g)
            dm = Map(Bn, DEV)
            sn = snaps[t]
            dm.main, dm.pend = sn.main.expand(Bn, -1, -1, -1).clone(), sn.pend.expand(Bn, -1, -1, -1).clone()
            dm.overlay = empty(Bn, DEV)
            pose0 = sn.pose.expand(Bn, 2).clone(); pose = pose0.clone()
            last_p, last_v = en["p"][t][None].to(DEV).float().expand(Bn, -1, -1), en["v"][t][None].to(DEV).float().expand(Bn, -1)
            seen = {}                                                    # verified world cells of the real prefix
            for s_ in range(t + 1):
                for k in range(MAP):
                    cl = int(rterr[e][s_][k // 9, k % 9])
                    if cl >= 0:
                        seen[(int(pos[s_][0] + KR[k]), int(pos[s_][1] + KC[k]))] = (s_, cl)
            for j in range(1, H + 1):
                a = int(en["action"][t + j - 1])
                pa = torch.full((Bn,), a, device=DEV, dtype=torch.long)
                fixj = rep(at("fix", j))
                logK = loc.kernel(last_p, last_v, pa)
                step = torch.stack([NR, NC], -1).to(DEV)[logK.argmax(-1)]
                orc = pose0 + torch.as_tensor(pos[t + j] - pos[t], device=DEV)
                pose = torch.where((arm_of == 1)[:, None], orc, pose + step)
                conf = torch.where(arm_of == 1, torch.ones(Bn, device=DEV), logK.exp().max(-1).values)
                ent, cp, fp, drm = dm.view()
                fj = read_features(ent, cp, fp, drm, pose, conf, torch.full((Bn,), t + j, device=DEV))
                fj = torch.where(noarm, blank(fj), fj)
                with TR.autocast(DEV, cfg):
                    mem = S.cond(model.belief.memory(bp)[:, None], rs[:, None])
                    den.place_input = fj[:, None]
                    parts = M.sample(den, dims, pa[:, None], mem, fixj[:, None], cache, cfg, g)
                    parts = tuple(x[:, 0].float() for x in parts)
                    reads0, _, _ = S.history(model, parts[0], state=rm)
                    h = model.outcome.features(bp, S.embed(model, M.Percept(*parts, fixj), reads0)).float()
                    r = model.outcome.sample_reward(h, g)
                    done = torch.rand(Bn, device=DEV, generator=g) > torch.sigmoid(model.outcome.cont_logit(h, r).float())
                    reads1, _, next_rm = S.history(model, parts[0], state=rm)
                    emb = S.embed(model, M.Percept(*parts, fixj), reads1)
                    nxt = torch.full((Bn,), int(en["action"][t + j]) if j < H else 0, device=DEV, dtype=torch.long)
                    yy, carry = model.belief(model.belief.tokens(emb[:, None], r[:, None], done[:, None], nxt[:, None]), carry)
                    cache = M.context_pass(den, tuple(x[:, None] for x in parts), pa[:, None], mem, fixj[:, None], cache, cfg, g)
                bp, rs, rm = yy[:, 1].float(), next_rm, next_rm
                write(dm, pose, torch.ones(Bn, device=DEV), parts[0], parts[2], parts[1], fixj, torch.full((Bn,), t + j, device=DEV),
                      torch.full((Bn,), -1, device=DEV), arm_of > 0, layer=dm.overlay)
                last_p, last_v = parts[0], parts[2]
                te = B.terrain(R, parts[0][:, None], parts[2][:, None])[:, 0].cpu().numpy()   # [Bn, 7, 9]
                right = ((pose - pose0).cpu().numpy() == (pos[t + j] - pos[t])).all(1)         # per row
                motion.append(right[arm_of.cpu().numpy() == 2].mean())
                prev = {(int(pos[t + j - 1][0] + KR[k]), int(pos[t + j - 1][1] + KC[k])) for k in range(MAP)}
                for k in range(MAP):
                    w = (int(pos[t + j][0] + KR[k]), int(pos[t + j][1] + KC[k]))
                    if w in prev or w not in seen:
                        continue
                    s0, cl = seen[w]
                    if int(rterr[e][t + j][k // 9, k % 9]) != cl:
                        continue                                         # changed in reality: not a verified revisit
                    band = "le16" if t + j - s0 <= 16 else "17-64" if t + j - s0 <= 64 else "65+"
                    for b in range(Bn):
                        stratum = "pose_right" if right[2 * MF + b % MF] else "pose_wrong"
                        key = (ARMS[b // MF], band, stratum)
                        a_, n_ = acc.get(key, (0, 0))
                        acc[key] = (a_ + int(te[b, k // 9, k % 9] == cl), n_ + 1)
        print("episode", int(e), "done", flush=True)
    res = {"update": update, "dream_motion_right": float(np.mean(motion)) if motion else None,
           "agreement": {f"{a}/{b}/{c}": {"agree": v[0] / v[1], "cells": v[1]} for (a, b, c), v in sorted(acc.items())}}
    (HERE / f"dream_{update}{'_smoke' if smoke else ''}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)


def smoke():
    """A 30-update fine-tune into run_smoke/ and the smoke evaluation (8 episodes) at 0 and 30."""
    global RUN
    RUN = HERE / "run_smoke"
    if not (RUN / "denoiser_30.pt").exists():
        train(updates=30, saves=(30,))
    evaluate(0, smoke=True)
    evaluate(30, smoke=True)


def identity(update):
    PR.HERE = HERE                                                      # place_read's reading on these cells
    PR.identity(update)

if __name__ == "__main__":
    {"tokens": tokens, "localizer": lambda: (train_localizer(), score_localizer(True, "full"), score_localizer(False, "odometry")),
     "score": lambda: score_localizer(True, "full"), "train": lambda: train(), "eval": lambda: evaluate(int(sys.argv[2])),
     "identity": lambda: identity(int(sys.argv[2])), "smoke": lambda: smoke(), "inject": lambda: inject(int(sys.argv[2])),
     "dream": lambda: dream(int(sys.argv[2])), "readings": lambda: (evaluate(int(sys.argv[2])), identity(int(sys.argv[2])),
                                                                  inject(int(sys.argv[2])), dream(int(sys.argv[2])))}[sys.argv[1]]()
