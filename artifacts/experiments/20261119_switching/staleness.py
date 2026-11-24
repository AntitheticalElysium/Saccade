"""Staleness of a cached substitute (user 11-19: is an epoch-stale cache of the denoiser's own estimates a usable
substitute for fresh ones?). Frozen P1.5 checkpoints 1-10, 2.56M frames apart: about one pass over the 2.43M-frame
training split, the cache's staleness (2,500 updates in the 64-step half, 625 in the 256-step half).

32 fixed training chunks of 72 frames from episode starts (zero states; seeded prior fixations). At every checkpoint,
with the same signal levels (finest grid, uniform, as G1's flow rows) and the same noise draws: Mamba and the record
from the recorded frames, then the denoiser's one-step estimate of every frame (the substitute switching uses; frames
8-71 kept). Per consecutive pair (k, k + 1), per element of the periphery tokens, by signal quartile and region:
- err: the fresh estimate's error, |x_hat_{k+1} - x_{k+1}|^2;
- stale_err: the stale estimate's error against the current tokens, |x_hat_k - x_{k+1}|^2 (what a cache would feed);
- drift: |x_hat_{k+1} - x_hat_k|^2 (XBM's D(x, t; dt)); token_drift: |x_{k+1} - x_k|^2 (the encoder's own drift);
- cos: the cosine between the stale and the fresh error vectors per frame (does a stale substitute carry the current
  model's errors, or different ones).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261119_switching.staleness
"""
import importlib, json
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[0] / "20261116_p15/run"
AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
N_EP, L, KEEP_FROM, B = 32, 72, 8, 8


@torch.no_grad()
def estimates(model, cfg, chunks, seed):
    dev = next(model.parameters()).device
    den, bel, k = model.denoiser, model.belief, cfg["edm"]["shortcut"]["k_max"]
    est_p, tok_p, sig = [], [], []
    for bi, c in enumerate(chunks):
        t = TR.tensors(c, dev)
        g = torch.Generator(device=dev).manual_seed(seed + bi)
        with TR.autocast(dev, cfg):
            Pc = model.perception(t["frames"], t["fixation"])
            rin, rst = model.record.scan(Pc.p, t["first"], model.record.zero(len(c["frames"]), dev))
            emb = bel.percept_embed(Pc, rst)
            y, _ = bel(bel.tokens(emb, t["reward"], t["terminated"], t["action"]), bel.initial(len(c["frames"]), dev))   # as encode
            nxt = y[:, 1::2]
            bpred = torch.cat([torch.zeros_like(nxt[:, :1]), nxt[:, :-1]], 1) * (~t["first"])[..., None]
            x0 = Pc.parts()
            Bn = len(c["frames"])
            lv = [M.shortcut_levels((Bn, L), k, torch.zeros(Bn, dtype=torch.bool, device=dev), dev, g) for _ in range(2)]
            tau = [s.float() / k for s in M._parts(tuple(l[0] for l in lv))]
            z = tuple((1 - M._bcast(tt, x)) * torch.randn(x.shape, device=dev, generator=g, dtype=torch.float32)
                      + M._bcast(tt, x) * x.float() for x, tt in zip(x0, tau))
            out = den(tuple(zz.to(x.dtype) for zz, x in zip(z, x0)), tuple(M.shortcut_code(l[0], l[1], k) for l in lv),
                      t["prev_action"], (bel.memory(bpred), rin), t["fixation"], t["episode"], t["valid"])
            clamp = cfg["perception"]["periphery"]["tanh_clamp"]
        est_p.append(out[0].float().clamp(-clamp, clamp)[:, KEEP_FROM:].cpu())
        tok_p.append(x0[0].float()[:, KEEP_FROM:].cpu())
        sig.append(lv[0][0][:, KEEP_FROM:].cpu())
    return torch.cat(est_p), torch.cat(tok_p), torch.cat(sig)


def main():
    model, cfg = AI.load(RUN / "ckpt_10.pt")
    split = D.load_split(cfg)
    eps = [e for e in split["train"] if e.T >= L]
    rng = np.random.default_rng(20261122)
    pick = np.sort(rng.choice(len(eps), N_EP, replace=False))
    eps = [eps[i] for i in pick]
    chunks = []
    for a in range(0, N_EP, B):
        recs = []
        for j, ep in enumerate(eps[a:a + B]):
            r = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
            recs.append([(j, a + j, t, *D.fixation_prior(r, ())) for t in range(L)])
        chunks.append(D.gather(eps[a:a + B], recs))
    res, prev = {"episodes": [(int(e.shard), int(e.slot)) for e in eps], "pairs": {}}, None
    for kk in range(1, 11):
        model, cfg = AI.load(RUN / f"ckpt_{kk:02d}.pt")
        model.requires_grad_(False).eval()
        est, tok, sig = estimates(model, cfg, chunks, 20261122)
        if prev is not None:
            pe, pt, _ = prev
            q = (sig.float() / cfg["edm"]["shortcut"]["k_max"] * 4).floor().clamp(max=3).long()
            out = {}
            for region, cells in (("map", slice(0, 63)), ("hud", slice(63, 81)), ("all", slice(0, 81))):
                e_f, e_s = est[:, :, cells] - tok[:, :, cells], pe[:, :, cells] - tok[:, :, cells]
                m = lambda x: x.square().mean((-1, -2))                    # per frame, per element
                cos = torch.nn.functional.cosine_similarity(e_f.flatten(2), e_s.flatten(2), dim=-1)
                row = {}
                for qi in range(4):
                    sel = q == qi
                    row[f"signal_q{qi}"] = {"err": float(m(e_f)[sel].mean()), "stale_err": float(m(e_s)[sel].mean()),
                                            "drift": float(m(est[:, :, cells] - pe[:, :, cells])[sel].mean()),
                                            "token_drift": float(m(tok[:, :, cells] - pt[:, :, cells])[sel].mean()),
                                            "cos": float(cos[sel].mean())}
                out[region] = row
            res["pairs"][f"{kk - 1}->{kk}"] = out
            a = out["all"]
            print(kk - 1, "->", kk, {q_: {x: round(v, 5) for x, v in a[q_].items()} for q_ in ("signal_q0", "signal_q3")}, flush=True)
        prev = (est, tok, sig)
        del model; torch.cuda.empty_cache()
    (HERE / "staleness.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
