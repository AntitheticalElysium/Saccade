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
import importlib, json, math, sys, time
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

    def score(self, c, e, age, present, v):
        """log p(c | entry) - log p_bg(c), zero where there is no entry. c, e [..., 16]; age, present [...]; v [..., 8]."""
        h = self.lik(torch.cat([PR.age_features(age, present), v], -1))
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
    pad = torch.zeros(Bn, G + 6, G + 8, 17, device=c.device); pad[..., 16] = -1
    pad[:, 3:3 + G, 4:4 + G] = ent[..., :17]                             # the coarse token and its time
    out = torch.zeros(Bn, G, G, device=c.device)
    vv = v[:, None, None].expand(Bn, G, G, 8)
    for k in range(MAP):
        e = pad[:, 3 + int(KR[k]):3 + int(KR[k]) + G, 4 + int(KC[k]):4 + int(KC[k]) + G]
        age = t[:, None, None] - e[..., 16]
        present = (e[..., 16] >= 0) & ((age <= 1) if not memory_term else torch.ones_like(age, dtype=torch.bool))
        out += loc.score(c[:, None, None, k], e[..., :16], age, present, vv)
    return out


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
    if not go.any():
        return
    cp, fp = mp.pend[..., CTIME] >= 0, mp.pend[..., FTIME] >= 0
    coarse = COARSE.to(mp.main.device)
    sel = go[:, None, None, None] & ((cp[..., None] & coarse) | (fp[..., None] & ~coarse))
    mp.main = torch.where(sel, mp.pend, mp.main)
    mp.pend[go] = empty(1, mp.pend.device)[0]


@torch.no_grad()
def advance(loc, mp, p, v, g, fix, a_prev, t, src, first, valid, truth=None, memory_term=True, inject=None):
    """One real frame for every stream: reset at `first`; else predict the pose from the motion prior on the last frame
    and the command that led here, read the memory there for this frame (before it writes), observe it, update the
    belief (relocating where the evidence says so), write. Returns (read features [B, 63, 9, FEAT], predicted pose,
    its confidence). `inject` [B, 2]: a pose error added to the belief before this step (reading 3)."""
    mp.reset(first & valid)
    live = valid & ~first
    if inject is not None:
        for b in torch.nonzero((inject != 0).any(1))[:, 0].tolist():
            mp.belief[b] = shifted(mp.belief[b][..., None], int(inject[b, 0]), int(inject[b, 1]), 0.0)[..., 0]
            mp.pose[b] += inject[b]
    logK = loc.kernel(mp.last_p, mp.last_v, a_prev)
    pri = predict(loc, mp, logK)
    xhat, chat = argmax2(pri)
    xhat = torch.where(live[:, None], xhat, mp.pose); chat = torch.where(live, chat, torch.ones_like(chat))
    ent, cp, fp, dream = mp.view()
    feats = read_features(ent, cp, fp, dream, xhat, chat, t)
    L = grid_loglik(loc, ent, p[:, :MAP].float(), v.float(), t.float(), memory_term)
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
        order = rng.permutation(len(data))
        tot, cnt, t0 = 0.0, 0, time.time()
        for bi in range(0, len(order), batch):
            eps = [data[i] for i in order[bi:bi + batch]]
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
                        Lfull = grid_loglik(loc, ent, p[:, :MAP], v, tt.float())
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
                    rr = (hyp[..., 0, None] + KR.to(dev)).clamp(0, G - 1); cc = (hyp[..., 1, None] + KC.to(dev)).clamp(0, G - 1)
                    e = ent[torch.arange(Bn, device=dev)[:, None, None], rr, cc]          # [B, 9, 63, NE]
                    age = tt[:, None, None].float() - e[..., CTIME]
                    Lh = loc.score(p[:, None, :MAP], e[..., CT], age, e[..., CTIME] >= 0, v[:, None, None].expand(Bn, 9, MAP, 8)).sum(-1)
                    Mx = torch.maximum(Lfull.flatten(1).max(1).values, Lh.max(1).values).detach()
                    local = ((1 - eps_) * pri * (Lh - Mx[:, None]).exp()).sum(1)
                    glob = eps_ * (Lfull - Mx[:, None, None]).exp().mean((1, 2))
                    ll = (local + glob).clamp(min=1e-30).log() + Mx + loc.bg(p[:, :MAP]).sum(-1)
                    loss_acc = loss_acc - (ll * live).sum(); n_acc += int(live.sum())
                with torch.no_grad():
                    advance(loc, mp, p, v, g, fix, a, tt, tt, first, valid)
                if (t % chunk == chunk - 1 or t == Tm) and n_acc:
                    opt.zero_grad(); (loss_acc / n_acc).backward(); opt.step()
                    tot += float(loss_acc) ; cnt += n_acc; loss_acc, n_acc = 0.0, 0
            print(f"pass {ps} batch {bi // batch} nll/frame {tot / max(cnt, 1):.2f} {time.time() - t0:.0f}s", flush=True)
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
    a = {k: np.array(v) for k, v in st.items()}
    mv, dk = a["moved"].astype(bool), a["dark"].astype(bool)
    res = {"tag": tag, "movement_commands": int(len(mv)),
           "prior": {"all": a["prior_ok"].mean(), "moved": a["prior_ok"][mv].mean(), "blocked": a["prior_ok"][~mv].mean(),
                     "dark": a["prior_ok"][dk].mean()},
           "posterior": {"all": a["post_ok"].mean(), "moved": a["post_ok"][mv].mean(), "blocked": a["post_ok"][~mv].mean(),
                         "dark": a["post_ok"][dk].mean()},
           "relative_pose_exact_16": a["rel16_ok"].mean(), "revisit_lookups": int(a["lookup_n"].sum()),
           "lookup_correct": a["lookup_ok"].sum() / max(a["lookup_n"].sum(), 1)}
    res = json.loads(json.dumps(res, default=float))
    (HERE / f"localizer_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    {"tokens": tokens, "localizer": lambda: (train_localizer(), score_localizer(True, "full"), score_localizer(False, "odometry")),
     "score": lambda: score_localizer(True, "full")}[sys.argv[1]]()
