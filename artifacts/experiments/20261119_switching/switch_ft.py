"""Random switching (JEDI, 2605.13013 section 3.4 and Algorithm 1; CURRICULUM 2(b)'s first ablation) as a fine-tune of
P1.5's denoiser (PLAN.md). The protocol of the earlier denoiser fine-tunes (traces_ft, swax_ft, record_ft): from P1.5
checkpoint 10 with its streams, carry, cache and record state; the denoiser alone trains (every other module frozen);
G1 (shortcut forcing, bootstrap rows included) alone; AdamW 1e-4, 100 warm-up updates, clip 1.0; 64-step chunks,
16 streams in 2 microbatches; the checkpoint's chooser picks the fixations (frozen), as in P1.5's training.

Switching: every chunk frame (never a prefix frame), with probability p independently, is replaced by the denoiser's
own one-step estimate of it on the routes chosen: pass 1 (no gradient) denoises every frame from an independent noise
draw at an independent flow level (signal uniform on the finest grid), conditioned on the recorded past and memory,
and clamps it to the token range. Routes:
- ctx: the frame's clean signal under its own noise in the G1 pass becomes the estimate, so later frames attend to it
  (and the frame's own target, always the recorded frame, is reached from a noised estimate);
- mem: Mamba and the record read the estimate for that frame; the memory tokens and record the denoiser reads come
  from them (Mamba is frozen; the carried state, cache and record for the next chunk come from the recorded frames).
The estimate enters shortcut_loss through its own (levels, noise) arguments: noise' = noise + tau / (1 - tau) (x_in -
x0) makes its input (1 - tau) noise + tau x_in while its target stays x0 (tau < 1 on every grid); checked at update 1.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261119_switching.switch_ft OUT [--updates N] [--p P] [--routes ctx,mem]
"""
import argparse, dataclasses, importlib, json, os, time
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P15 = ROOT / "artifacts/experiments/20261116_p15/run/ckpt_10.pt"
AI = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
WORLD_SEED = importlib.import_module("artifacts.experiments.20261116_p15.train").WORLD_SEED
SEED = 20261121


def g1(model, chunk, carry, cache, record, cfg, boot, update, gen, p, routes, check=False):
    """G1 of one microbatch with switching; returns (g1, carry, cache, record, switched share)."""
    P, dev = cfg["data"]["context_prefix"], cache.device
    t = TR.tensors(chunk, dev)
    first, fix = t["first"], t["fixation"]
    B, L = first.shape
    den, bel, sc = model.denoiser, model.belief, cfg["edm"]["shortcut"]
    k = sc["k_max"]
    seq = first[:, P:].long().cumsum(1).repeat_interleave(2, 1)
    with TR.autocast(dev, cfg):
        with torch.no_grad():
            Pc = model.perception(t["frames"][:, P:], fix[:, P:])
            Pp = model.perception(t["frames"][:, :P], fix[:, :P])

            def belief_pass(Pchunk):
                rin, rst = model.record.scan(torch.cat([Pp.p, Pchunk.p], 1), first, record)
                emb = bel.percept_embed(Pchunk, rst[:, P:])
                y, c = bel(bel.tokens(emb, t["reward"][:, P:], t["terminated"][:, P:], t["action"][:, P:]), carry, seq)
                nxt = y[:, 1::2]
                return rin, rst, torch.cat([cache.to(nxt.dtype), nxt[:, :-1]], 1) * (~first)[..., None], c, nxt
            rin, rst, bpred, carry_out, nxt = belief_pass(Pc)
            new_record, new_cache = rst[:, L - P - 1], torch.cat([bpred[:, -P:], nxt[:, -1:]], 1).float()
            x0 = tuple(torch.cat([a, b], 1) for a, b in zip(Pp.parts(), Pc.parts()))
            memory, x_ctx, share = (bel.memory(bpred), rin), x0, 0.0
            if p > 0:
                lv = [M.shortcut_levels((B, L), k, torch.zeros(B, dtype=torch.bool, device=dev), dev, gen) for _ in range(2)]
                tau = [s.float() / k for s in M._parts(tuple(l[0] for l in lv))]
                z1 = tuple((1 - M._bcast(tt, x)) * torch.randn(x.shape, device=dev, generator=gen, dtype=torch.float32)
                           + M._bcast(tt, x) * x.float() for x, tt in zip(x0, tau))
                est = den(tuple(z.to(x.dtype) for z, x in zip(z1, x0)), tuple(M.shortcut_code(l[0], l[1], k) for l in lv),
                          t["prev_action"], memory, fix, t["episode"], t["valid"])
                clamp = cfg["perception"]["periphery"]["tanh_clamp"]
                est = tuple(e.float().clamp(-clamp, clamp).to(x.dtype) for e, x in zip(est, x0))
                m = (torch.rand(B, L, device=dev, generator=gen) < p) & t["valid"]
                m[:, :P] = False
                share = float(m[:, P:].float().mean())
                x_in = tuple(torch.where(M._bcast(m, x), e, x) for x, e in zip(x0, est))
                if "mem" in routes:
                    rin, _, bpred, _, _ = belief_pass(M.Percept(*(x[:, P:] for x in x_in), fix[:, P:]))
                    memory = (bel.memory(bpred), rin)
                if "ctx" in routes:
                    x_ctx = x_in
        levels = [M.shortcut_levels((B, L), k, torch.as_tensor(boot, device=dev), dev) for _ in range(2)]
        tau = [s.float() / k for s in M._parts(tuple(l[0] for l in levels))]
        base = tuple(torch.randn_like(x) for x in x0)
        noise = tuple(n + M._bcast(tt / (1 - tt), x) * (xi.float() - x.float()) for n, x, xi, tt in zip(base, x0, x_ctx, tau))
        if check:                                                        # the input the loss builds is (1 - tau) n + tau x_in
            for n, x, xi, tt, nn_ in zip(base, x0, x_ctx, tau, noise):
                built = (1 - M._bcast(tt, x)) * nn_ + M._bcast(tt, x) * x
                want = (1 - M._bcast(tt, x)) * n + M._bcast(tt, x) * xi
                assert float((built.float() - want.float()).abs().max()) < 1e-3, "switching input identity"
        loss = M.shortcut_loss(den, x0, t["prev_action"], memory, fix, t["episode"], t["valid"], sc,
                               torch.as_tensor(boot, device=dev), update >= sc["bootstrap_start"], levels=levels, noise=noise)
    return loss[:, P:].mean().float(), carry_out, new_cache, new_record, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--reads", default="1500,3000")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--routes", default="ctx,mem")
    args = ap.parse_args()
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    routes, reads = tuple(args.routes.split(",")), {int(x) for x in args.reads.split(",")}
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(P15, map_location="cpu", weights_only=False)
    model, cfg = AI.load(P15)
    dev = next(model.parameters()).device
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    model.requires_grad_(False)
    den = model.denoiser
    params = list(den.parameters())
    for q in params:
        q.requires_grad_(True)
    split = D.load_split(cfg)
    res = {"recipe": {"from": str(P15.relative_to(ROOT)), "checkpoint_sha256": TR.file_sha256(P15), "updates": args.updates,
                      "p": args.p, "routes": routes, "lr": 1e-4, "warmup": 100, "clip": 1.0, "chunk": 64, "seed": SEED,
                      "script_sha256": TR.file_sha256(__file__)}, "log": [], "monitor": {}}
    model.eval()
    res["monitor"][0] = TR.monitor(model, cfg, split["monitor"])
    print("monitor 0", json.dumps(res["monitor"][0]), flush=True)
    model.train()
    den.checkpoint = cfg["p1"]["activation_checkpointing"]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    streams = D.Streams(split["train"], d["streams"], P, seed=WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    carry = tuple(type(c)(*(getattr(c, f.name).to(dev) for f in dataclasses.fields(c))) for c in ck["carry"])
    cache, record = ck["cache"].to(dev), ck["record"].to(dev)
    st = dict(ck["fixation_controller"], trace_rows=0, log_bytes=0)     # a fresh trace in this folder; the head frozen
    ctl = C.Controller(model, cfg, out, dev); ctl.load_state(st)
    base, gen = ck["state"]["update"], torch.Generator(device=dev).manual_seed(SEED)
    losses, shares, t0 = [], [], time.time()
    for u in range(1, args.updates + 1):
        chunk = streams.next(64)
        ctl.prepare(chunk, streams)
        boot = TR.bootstrap_rows(cfg, WORLD_SEED, base + u)
        for g in opt.param_groups:
            g["lr"] = 1e-4 * min(1.0, u / 100)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), 2):
            loss, c_new, cache_new, rec_new, share = g1(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                        record[rows], cfg, boot[rows], base + u, gen, args.p, routes, check=u == 1)
            (loss / 2).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows], record[rows] = cache_new, rec_new
            losses.append(float(loss)); shares.append(share)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if u == 1:
            print("peak GiB", torch.cuda.max_memory_allocated() / 2 ** 30, "sec", time.time() - t0, flush=True)
        if u % 100 == 0:
            res["log"].append({"update": u, "g1": float(np.mean(losses[-200:])), "switched": float(np.mean(shares[-200:])),
                               "grad_norm": gn, "sec_per_update": (time.time() - t0) / u})
            print(json.dumps(res["log"][-1]), flush=True)
        if u in reads:
            model.eval()
            res["monitor"][u] = TR.monitor(model, cfg, split["monitor"])
            print(f"monitor {u}", json.dumps(res["monitor"][u]), flush=True)
            torch.save(dict(ck, model=model.state_dict(), switching=res["recipe"], switch_update=u), out / f"ckpt_sw{u:04d}.pt")
            model.train()
        (out / "switch_ft.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
