"""Learned fixation during P1 (canonical since P1-F2; moved into the core 11-07 with P1-F4): a cheap coarse-only head
(101,584 parameters) predicts each target frame's periphery tokens from the 8 preceding frames of its own episode, the
action and the outcomes, and the window to look at is drawn from 0.8 softmax(0.5 C + 0.5 S) + 0.2 / 49 over the 49
centres: C = the action's predicted effect (taken minus NOOP), S = the head's own error (surprise), each normalised by
its mean (TEACHING 9-10: contingency says where, learning progress / surprise whether). The head learns online from the
world's coarse percepts, isolated from the world's optimiser, RNG and gradients. Sources: artifacts/experiments/
20261102_direct_chooser/direct.py (head, policy) and 20261103_p1f2/chooser.py (controller), unchanged in arithmetic;
the hyperparameters are cfg["fixation"]["chooser"] (saccade/m0.yaml). P1.6's world-eye (cfg["fixation"]["world_eye"]):
after the checkpoints `refresh_after`, S is the world's own error from a cache filled then (world_errors); C stays the
head's (20261120_switching_dissection C9-C11)."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from saccade import data as D, model as M

CENTRES = torch.cartesian_prod(torch.arange(1, 8), torch.arange(1, 8))
WIN = M.window(CENTRES)
TRACE = np.dtype([("target", "<i4"), ("episode", "<i4"), ("window", "u1"), ("uniform", "u1"),
                  ("C", "u1"), ("S", "u1"), ("probability", "<f4")])


def policy(c, s, cfg):
    """P(window) [.., 49] from the contingency c and surprise s scores of every window (each normalised by its mean)."""
    norm = lambda x: torch.where(x.mean(-1, keepdim=True) > 1e-12,
                                 x / x.mean(-1, keepdim=True).clamp(min=1e-12), torch.zeros_like(x))
    u, wc, ws = cfg["uniform_floor"], cfg["contingency_weight"], cfg["surprise_weight"]
    return (1 - u) * ((wc * norm(c) + ws * norm(s)) / cfg["temperature"]).softmax(-1) + u / 49


class Block(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(w), nn.LayerNorm(w)
        self.qkv, self.proj = nn.Linear(w, 3 * w), nn.Linear(w, w)
        self.ff = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))

    def forward(self, x):
        b, n, w = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, 4, w // 4).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, w)
        x = x + self.proj(y)
        return x + self.ff(self.n2(x))


class Head(nn.Module):
    """Next periphery tokens [B, 81, 16] from (the H past frames' tokens [B, H, 81, 16], context [B, H * 29], action [B])."""

    def __init__(self, history):
        super().__init__()
        self.local = nn.Linear(history * 16, 64)
        self.context = nn.Sequential(nn.Linear(history * 29, 64), nn.GELU(), nn.Linear(64, 64))
        self.action = nn.Embedding(17, 64)
        self.position = nn.Parameter(torch.randn(81, 64) * .02)
        self.blocks = nn.Sequential(Block(), Block())
        self.output = nn.Linear(64, 16)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, inputs):
        p, ctx, act = inputs
        x = self.local(p.transpose(1, 2).flatten(2))
        x = x + self.context(ctx)[:, None] + self.action(act)[:, None] + self.position
        return p[:, -1] + self.output(self.blocks(x))


def coarse(perception, frames):
    """The world's own periphery / lighting operations; never computes a fine crop."""
    lead = frames.shape[:-3]; f = frames.reshape(-1, 63, 63, 3); x = f.float() / 255
    v = M.out_map(perception.light(torch.cat([x.mean((1, 2)), x.std((1, 2))], -1)), perception.s)
    cells = f.reshape(-1, 9, 7, 9, 7, 3).permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, 7, 7).float()
    cells = torch.round(F.adaptive_avg_pool2d(cells, perception.acuity)) if perception.acuity < 7 else cells
    h = cells.reshape(len(f), 81, -1) / 127.5 - 1
    for layer, film in zip(perception.p_layers, perception.film):
        gain, shift = film(v).unsqueeze(1).chunk(2, -1); h = F.gelu(layer(h)) * (1 + gain) + shift
    p = M.out_map(perception.p_out(h), perception.s)
    return p.reshape(*lead, 81, -1), v.reshape(*lead, -1)


def inputs(p, v, chunk, prefix, history):
    """Every new target frame sees only its own episode's preceding `history` frames."""
    dev = p.device; n, L = p.shape[:2]
    target = torch.arange(prefix, L, device=dev); h = target[:, None] - torch.arange(history, 0, -1, device=dev)[None]
    episode = torch.as_tensor(chunk["episode"], device=dev); valid = torch.as_tensor(chunk["valid"], device=dev)
    ok = valid[:, h] & (episode[:, h] == episode[:, target, None])
    ph = p[:, h].float() * ok[..., None, None]; vh = v[:, h].float() * ok[..., None]
    pa = torch.as_tensor(chunk["prev_action"], device=dev); r = torch.as_tensor(chunk["reward"], device=dev)
    d = torch.as_tensor(chunk["terminated"], device=dev)
    ah = F.one_hot((pa[:, h] + 1).clamp(0, 17), 18).float() * ok[..., None]
    ctx = torch.cat([vh, ah, (M.SYMLOG(r[:, h]) * ok)[..., None], (d[:, h] * ok)[..., None], ok[..., None]], -1)
    return (ph.reshape(-1, history, 81, 16), ctx.reshape(-1, history * 29), pa[:, prefix:].reshape(-1)), p[:, prefix:].reshape(-1, 81, 16).float()


def _digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def world_errors(model, cfg, episodes, offsets, looks, path, seed, segment=256):
    """P1.6's world-eye (C9-C11): every frame's periphery error under the world's own estimate, per cell,
    ||p_t - x_hat_t||^2, written as [frames, 81] fp16 at `path` (rows as D.frame_offsets). x_hat_t is the flow estimate at
    signal 0 (finest step: the conditional mean, C2's) given the real past: the previous `context_prefix` frames at the
    context signal, the record and the predicted belief, built through each frame's actual look (`looks` [frames], a
    window or -1: the fixation prior). One two-stream pass (Denoiser.estimate) per segment of an episode."""
    from saccade import train as TR                                      # train imports this module
    dev = next(model.parameters()).device
    P, sc = cfg["data"]["context_prefix"], cfg["edm"]["shortcut"]
    k, tc = sc["k_max"], sc["context_signal"]
    code = lambda n, s_: torch.full((1, n), M.shortcut_code(s_, k.bit_length() - 1, k), device=dev)
    tmp = path.with_name(path.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(int(offsets[-1]), M.N))
    pc, rng, centres = cfg["perception"], np.random.default_rng(seed), CENTRES.numpy()
    dims = ((M.N, pc["periphery"]["token_dim"]), (M.G, pc["fovea"]["token_dim"]), (pc["lighting"]["dim"],))
    for i, ep in enumerate(episodes):
        w = looks[offsets[i]:offsets[i] + ep.T + 1].astype(np.int64)
        fix = np.where(w[:, None] >= 0, centres[w.clip(0)], D.fixation_prior(rng, (ep.T + 1,)))
        gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + i)   # per frame, whatever the segments:
        nc = [torch.randn((1, ep.T + 1, *d), device=dev, generator=gen) for d in dims]   # context-role noise
        nq = [torch.randn((1, ep.T + 1, *d), device=dev, generator=gen) for d in dims[::2]]   # the target's (p, v)
        state, carry, last, tail = model.record.zero(1, dev), model.belief.initial(1, dev), None, None
        for a in range(0, ep.T + 1, segment):
            ts = range(a, min(ep.T + 1, a + segment))
            c = TR.tensors(D.gather(episodes, [[(i, 1, t, *map(int, fix[t])) for t in ts]]), dev)
            with TR.autocast(dev, cfg):
                X = model.perception(c["frames"], c["fixation"])
                rin, rst = model.record.scan(model.record.input(X.p, X.g, X.fix), c["first"], state)
                y, carry = model.belief(model.belief.tokens(model.belief.percept_embed(X, rst), c["reward"], c["terminated"], c["action"]), carry)
                y = y[:, 1::2]                                           # the predicted belief of the next frame
                bp = torch.cat([torch.zeros_like(y[:, :1]) if last is None else last, y[:, :-1]], 1)
                cur = (*X.parts(), c["prev_action"], bp, rin, X.fix, c["episode"], c["valid"])
                win = cur if tail is None else tuple(torch.cat([u, z], 1) for u, z in zip(tail, cur))
                p, g, v, act, b, r, f, e, ok = win
                n, m = p.shape[1], p.shape[1] - len(ts)
                fr = slice(a - m, a + len(ts))                           # the window's frames
                xc = tuple((1 - tc / k) * nz[:, fr] + tc / k * z.float() for nz, z in zip(nc, (p, g, v)))
                xq = (nq[0][:, fr], torch.zeros(g.shape, device=dev), nq[1][:, fr])
                o = model.denoiser.estimate(xc, xq, (code(n, tc), code(n, 0)), act, (model.belief.memory(b), r), f, e, ok)
                err = (p[:, m:].float() - o[0][:, m:].float()).square().sum(-1)
            out[offsets[i] + a:offsets[i] + a + len(ts)] = err[0].cpu().numpy()
            state, last, tail = rst[:, -1], y[:, -1:], tuple(z[:, -P:] for z in win)
    out.flush()
    del out
    os.replace(tmp, path)


class Controller:
    """train.p1's fixation hooks: prepare (choose each target frame's window before the world's update; rewrite the
    chunk's and the streams' fixations), learn (the head's own updates, by world frames), state / load_state (resume),
    close. Traces: out_dir/gaze.bin (TRACE rows), out_dir/chooser.jsonl."""

    def __init__(self, model, cfg, out_dir, device):
        self.model, self.cfg, self.policy, self.device = model, cfg, dict(cfg["fixation"]["chooser"]), torch.device(device)
        with torch.random.fork_rng(devices=[]):                          # its own draws: the world's stream untouched
            torch.random.default_generator.manual_seed(self.policy["head_seed"]); self.head = Head(self.policy["history"]).to(self.device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=self.policy["learning_rate"], weight_decay=self.policy["weight_decay"])
        self.gaze = torch.Generator(device=self.device).manual_seed(self.policy["gaze_seed"])
        self.batch = torch.Generator(device=self.device).manual_seed(self.policy["batch_seed"])
        self.frames = self.updates = self.rows = 0; self.pending = None
        self.out = Path(out_dir); self.trace = self.out / "gaze.bin"; self.log = self.out / "chooser.jsonl"
        if self.trace.exists() and self.trace.stat().st_size and not (self.out / "latest.pt").exists():
            raise RuntimeError("Gaze trace without a world checkpoint; inspect before restarting")
        self.centres = CENTRES.to(self.device); self.win = WIN.to(self.device)
        self.world_eye, self.eye, self.eye_state = cfg["fixation"].get("world_eye"), None, None   # P1.6

    def state(self):
        assert self.pending is None
        return dict(config=self.policy, head=self.head.state_dict(), optimizer=self.opt.state_dict(),
                    gaze_rng=self.gaze.get_state(), batch_rng=self.batch.get_state(),
                    frames=self.frames, updates=self.updates, trace_rows=self.rows,
                    log_bytes=self.log.stat().st_size if self.log.exists() else 0, eye=self.eye_state)

    def load_state(self, state):
        assert state["config"] == self.policy, "Fixation recipe differs on resume"
        self.head.load_state_dict(state["head"]); self.opt.load_state_dict(state["optimizer"])
        self.gaze.set_state(state["gaze_rng"].cpu()); self.batch.set_state(state["batch_rng"].cpu())
        self.frames, self.updates, self.rows = state["frames"], state["updates"], state["trace_rows"]
        size = self.rows * TRACE.itemsize
        if size and (not self.trace.exists() or self.trace.stat().st_size < size): raise RuntimeError("Incomplete gaze trace")
        if self.trace.exists():
            with self.trace.open("r+b") as f: f.truncate(size)
        if state.get("eye"):                                             # P1.6: the cache this state scored with
            self.eye_state = state["eye"]; path = self.out / self.eye_state["file"]
            if not path.exists() or _digest(path) != self.eye_state["sha256"]: raise RuntimeError("World-eye cache missing or changed")
            self.eye = np.load(path, mmap_mode="r")
        log_size = state["log_bytes"]
        if log_size and (not self.log.exists() or self.log.stat().st_size < log_size): raise RuntimeError("Incomplete chooser log")
        if self.log.exists():
            with self.log.open("r+b") as f: f.truncate(log_size)

    @torch.no_grad()
    def prepare(self, chunk, streams):
        from saccade import train as TR                                  # train imports this module
        prefix, H = self.cfg["data"]["context_prefix"], self.policy["history"]; original = chunk["fixation"].copy()
        fr = torch.as_tensor(chunk["frames"], device=self.device); parts = []
        flat = fr.flatten(0, 1)
        for x in flat.split(self.policy["encoding_batch"]):
            with TR.autocast(self.device, self.cfg): parts.append(coarse(self.model.perception, x))
        p = torch.cat([x[0] for x in parts]).reshape(*fr.shape[:2], 81, 16)
        v = torch.cat([x[1] for x in parts]).reshape(*fr.shape[:2], 8)
        xx, y = inputs(p, v, chunk, prefix, H)
        eligible = torch.as_tensor(chunk["valid"][:, prefix:], device=self.device).flatten() & (xx[2] >= 0)
        ids = torch.where(eligible)[0]; self.pending = (tuple(x.detach() for x in xx), y.detach(), ids)
        targets = chunk["target"][:, prefix:].reshape(-1)
        assert self.eye is None or (targets[ids.cpu().numpy()] >= 0).all()
        rows = np.zeros(len(y), dtype=TRACE)
        rows["target"] = chunk["target"][:, prefix:].reshape(-1); rows["episode"] = chunk["episode"][:, prefix:].reshape(-1)
        uniform = (original[:, prefix:, 0] - 1) * 7 + original[:, prefix:, 1] - 1
        rows["uniform"] = uniform.reshape(-1); rows["window"] = rows["uniform"]; rows["probability"] = 1 / 49
        cs = []; ss = []; entropy = []; teacher = []
        for ix in ids.split(self.policy["scoring_batch"]):
            x = tuple(z[ix] for z in xx)
            with torch.autocast(self.device.type, enabled=False):
                taken = self.head(x); noop = self.head((x[0], x[1], torch.zeros_like(x[2])))
                c = (taken - noop).square().sum(-1)[:, self.win].sum(-1)
                s = (y[ix] - taken).square().sum(-1)[:, self.win].sum(-1)
                if self.eye is not None:                                 # P1.6: the world's own error (C9-C11)
                    s = torch.as_tensor(np.asarray(self.eye[targets[ix.cpu().numpy()]]), device=self.device).float()[:, self.win].sum(-1)
                assert torch.count_nonzero(c[x[2] == 0]) == 0
                prob = policy(c, s, self.policy)
            u = torch.rand(len(ix), 1, device=self.device, generator=self.gaze, dtype=torch.float64)
            choice = torch.searchsorted(prob.double().cumsum(-1), u).squeeze(-1).clamp(max=48)
            ci = ix.cpu().numpy(); rows["window"][ci] = choice.cpu().numpy()
            rows["C"][ci] = c.argmax(-1).cpu().numpy(); rows["S"][ci] = s.argmax(-1).cpu().numpy()
            rows["probability"][ci] = prob.gather(1, choice[:, None]).squeeze(1).cpu().numpy()
            cs.append(float(c.mean())); ss.append(float(s.mean())); entropy.append(float(-(prob * prob.log()).sum(-1).mean()))
            teacher.append(prob)
        self.teacher = torch.cat(teacher) if teacher else torch.zeros(0, 49, device=self.device)   # P1.5: per row of ids,
        chosen = torch.as_tensor(rows["window"].astype(np.int64), device=self.device)           # the gaze forecast's target
        chunk["fixation"][:, prefix:] = self.centres[chosen].reshape(*chunk["fixation"][:, prefix:].shape).cpu().numpy()
        assert np.array_equal(chunk["fixation"][:, :prefix], original[:, :prefix])
        for i, hist in enumerate(streams.hist):                          # the actual past observations, when these frames
            streams.hist[i] = [(*r[:3], *map(int, chunk["fixation"][i, -len(hist) + j])) for j, r in enumerate(hist)]   # become prefixes
        with self.trace.open("ab") as f: rows.tofile(f)
        self.rows += len(rows)
        self.description = dict(C=float(np.mean(cs)) if cs else 0., S=float(np.mean(ss)) if ss else 0.,
                                entropy=float(np.mean(entropy)) if entropy else math.log(49),
                                window_histogram=np.bincount(rows["window"], minlength=49).tolist(),
                                frames=len(rows), changed_fraction=float(np.mean(rows["window"] != rows["uniform"])))

    def looks(self, n):
        """Each frame's most recent look in this run's trace (window index; -1: never a target)."""
        last = np.full(n, -1, np.int16)
        if self.rows:
            raw = np.memmap(self.trace, dtype=TRACE, mode="r", shape=(self.rows,))
            for a in range(0, self.rows, 1 << 22):                       # in order: later rows overwrite earlier
                q = raw[a:a + (1 << 22)]; ok = q["target"] >= 0
                last[q["target"][ok]] = q["window"][ok]
        return last

    def refresh(self, episodes, offsets, index):
        """P1.6's world-eye: fill the cache with the world as it is now (after checkpoint `index`), then score S with it."""
        name = f"eye_cache_{index:02d}.npy"; was = self.model.training
        self.model.eval()
        try:
            world_errors(self.model, self.cfg, episodes, offsets, self.looks(int(offsets[-1])), self.out / name, self.world_eye["seed"] + index)
        finally:
            self.model.train(was)
        self.eye_state = {"file": name, "after_checkpoint": index, "sha256": _digest(self.out / name)}
        self.eye = np.load(self.out / name, mmap_mode="r")

    def learn(self, frames):
        xx, y, ids = self.pending; self.frames += int(frames)
        goal = self.frames // self.policy["world_frames_per_head_update"]; losses = []
        total = self.cfg["p1"]["frames"] // self.policy["world_frames_per_head_update"]
        while self.updates < goal:
            if not len(ids): self.updates += 1; continue
            step = self.updates; warm = min(1., (step + 1) / self.policy["warmup_updates"])
            frac = max(0., (step - self.policy["warmup_updates"]) / max(1, total - self.policy["warmup_updates"]))
            lr = warm * (self.policy["minimum_learning_rate"] + .5 * (self.policy["learning_rate"] - self.policy["minimum_learning_rate"]) * (1 + math.cos(math.pi * frac)))
            self.opt.param_groups[0]["lr"] = lr
            ix = ids[torch.randint(len(ids), (self.policy["batch_size"],), device=self.device, generator=self.batch)]
            with torch.autocast(self.device.type, enabled=False): loss = (self.head(tuple(z[ix] for z in xx)) - y[ix]).square().mean()
            self.opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.policy["grad_clip"]); self.opt.step()
            losses.append(float(loss.detach())); self.updates += 1
        self.pending = None
        with self.log.open("a") as f: f.write(json.dumps(dict(world_frames=self.frames, head_updates=self.updates,
                                                              head_loss=float(np.mean(losses)) if losses else None, **self.description)) + "\n")

    def close(self):
        torch.save(self.state(), self.out / "chooser_last.pt")
