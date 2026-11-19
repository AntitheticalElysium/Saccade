"""Random switching (JEDI, 2605.13013 section 3.4 and Algorithm 1; CURRICULUM 2(b)'s first ablation) as a fine-tune of
P1.5's denoiser (PLAN.md). The protocol of the earlier denoiser fine-tunes (traces_ft, swax_ft, record_ft): from P1.5
checkpoint 10 with its streams, carry, cache and record state; the denoiser alone trains (every other module frozen);
G1 (shortcut forcing, bootstrap rows included) alone; AdamW 1e-4, 100 warm-up updates, clip 1.0; 64-step chunks,
16 streams in 2 microbatches; the checkpoint's chooser picks the fixations (frozen), as in P1.5's training.

Switching: every chunk frame (never a prefix frame) is drawn into the switch mask with probability p independently and,
with substitution on, is replaced by the denoiser's own one-step estimate of it on the routes chosen. On the ctx route
a masked frame serves as history only (its own G1 terms are excluded: JEDI switches the next prediction's conditioning;
Codex's review); on the mem route alone a switched frame changes the conditioning from the next frame on, so every G1
target is kept (Codex verified frame t's flow and bootstrap terms bitwise unchanged): pass 1 (no gradient) denoises every frame from an independent noise
draw at an independent flow level (signal uniform on the finest grid), conditioned on the recorded past and memory,
and clamps it to the token range. Routes:
- ctx: the frame's clean signal under its own noise in the G1 pass becomes the estimate, so later frames attend to it;
- mem: Mamba and the record read the estimate for that frame; rec: the record alone (Mamba's percept input keeps the
  recorded frame and reads the switched record); the memory tokens and record the denoiser reads come
  from them (Mamba is frozen; the carried state, cache and record for the next chunk come from the recorded frames).
The control arm (--substitute 0) draws the same mask and excludes the same frames' terms without substituting. Three
seeded generators, shared by both arms: G1's levels and noise, the mask, and the substitute's own draws (Codex: the
arms must see identical frames, fixations, levels, noise and masks). On bootstrap rows the target is the model's own
finer-step prediction, as in P1 (only flow rows target the recorded frame). The estimate enters shortcut_loss through
its own (levels, noise) arguments: noise' = noise + tau / (1 - tau) (x_in -
x0) makes its input (1 - tau) noise + tau x_in while its target stays x0 (tau < 1 on every grid); checked at update 1.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261119_switching.switch_ft OUT [--updates N] [--p P]
        [--substitute 1|0] [--routes ctx,mem]
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


def g1(model, chunk, carry, cache, record, cfg, boot, update, gens, p, substitute, routes, check=False):
    """G1 of one microbatch with switching; gens = (G1's generator, None = the process RNG; the mask's; the substitute's).
    Returns (g1, carry, cache, record, stats: masked share, the substitute's error by signal quartile, draw fingerprints)."""
    g1_gen, mask_gen, est_gen = gens
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

            def belief_pass(Pchunk, p_rec=None):                             # p_rec: what the record reads (rec route)
                rin, rst = model.record.scan(torch.cat([Pp.p, Pchunk.p if p_rec is None else p_rec], 1), first, record)
                emb = bel.percept_embed(Pchunk, rst[:, P:])
                y, c = bel(bel.tokens(emb, t["reward"][:, P:], t["terminated"][:, P:], t["action"][:, P:]), carry, seq)
                nxt = y[:, 1::2]
                return rin, rst, torch.cat([cache.to(nxt.dtype), nxt[:, :-1]], 1) * (~first)[..., None], c, nxt
            rin, rst, bpred, carry_out, nxt = belief_pass(Pc)
            new_record, new_cache = rst[:, L - P - 1], torch.cat([bpred[:, -P:], nxt[:, -1:]], 1).float()
            x0 = tuple(torch.cat([a, b], 1) for a, b in zip(Pp.parts(), Pc.parts()))
            memory, x_ctx, stats = (bel.memory(bpred), rin), x0, {}
            m = (torch.rand(B, L, device=dev, generator=mask_gen) < p) & t["valid"]
            m[:, :P] = False
            stats["masked"] = float(m[:, P:].float().mean())
            if p > 0 and substitute:
                lv = [M.shortcut_levels((B, L), k, torch.zeros(B, dtype=torch.bool, device=dev), dev, est_gen) for _ in range(2)]
                tau = [s.float() / k for s in M._parts(tuple(l[0] for l in lv))]
                z1 = tuple((1 - M._bcast(tt, x)) * torch.randn(x.shape, device=dev, generator=est_gen, dtype=torch.float32)
                           + M._bcast(tt, x) * x.float() for x, tt in zip(x0, tau))
                est = den(tuple(z.to(x.dtype) for z, x in zip(z1, x0)), tuple(M.shortcut_code(l[0], l[1], k) for l in lv),
                          t["prev_action"], memory, fix, t["episode"], t["valid"])
                clamp = cfg["perception"]["periphery"]["tanh_clamp"]
                est = tuple(e.float().clamp(-clamp, clamp).to(x.dtype) for e, x in zip(est, x0))
                err = (est[0].float() - x0[0].float()).square().mean((-1, -2))[:, P:]          # periphery, per frame
                q = (tau[0][:, P:] * 4).floor().clamp(max=3).long()
                stats["substitute_mse_by_signal_quartile"] = [float(err[q == i].mean()) if (q == i).any() else None for i in range(4)]
                x_in = tuple(torch.where(M._bcast(m, x), e, x) for x, e in zip(x0, est))
                if "mem" in routes:
                    rin, _, bpred, _, _ = belief_pass(M.Percept(*(x[:, P:] for x in x_in), fix[:, P:]))
                    memory = (bel.memory(bpred), rin)
                elif "rec" in routes:                                         # Codex 11-19: the record alone; Mamba keeps
                    rin, _, bpred, _, _ = belief_pass(Pc, x_in[0][:, P:])     # the recorded percepts, reads the record
                    memory = (bel.memory(bpred), rin)
                if "ctx" in routes:
                    x_ctx = x_in
        levels = [M.shortcut_levels((B, L), k, torch.as_tensor(boot, device=dev), dev, g1_gen) for _ in range(2)]
        tau = [s.float() / k for s in M._parts(tuple(l[0] for l in levels))]
        base = tuple(torch.randn(x.shape, device=dev, dtype=x.dtype, generator=g1_gen) for x in x0)
        stats["fingerprint"] = [float(levels[0][0].float().sum()), float(levels[1][0].float().sum()), float(base[0].float().sum()),
                                int(m.sum()), float(t["frames"].float().mean()), int(fix.sum())]
        noise = tuple(n + M._bcast(tt / (1 - tt), x) * (xi.float() - x.float()) for n, x, xi, tt in zip(base, x0, x_ctx, tau))
        if check:                                                        # the input the loss builds is (1 - tau) n + tau x_in
            for n, x, xi, tt, nn_ in zip(base, x0, x_ctx, tau, noise):
                built = (1 - M._bcast(tt, x)) * nn_ + M._bcast(tt, x) * x
                want = (1 - M._bcast(tt, x)) * n + M._bcast(tt, x) * xi
                assert float((built.float() - want.float()).abs().max()) < 1e-3, "switching input identity"
        loss = M.shortcut_loss(den, x0, t["prev_action"], memory, fix, t["episode"], t["valid"], sc,
                               torch.as_tensor(boot, device=dev), update >= sc["bootstrap_start"], levels=levels, noise=noise)
    keep = ~m[:, P:] if "ctx" in routes else torch.ones_like(m[:, P:])   # ctx: switched frames are history only
    return loss[:, P:][keep].mean().float(), carry_out, new_cache, new_record, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--reads", default="2000,4000")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--substitute", type=int, default=1)
    ap.add_argument("--routes", default="ctx,mem")
    ap.add_argument("--resume", type=Path)                               # a state_sw*.pt of this folder: continue it (Codex)
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
                      "p": args.p, "substitute": bool(args.substitute), "routes": routes, "lr": 1e-4, "warmup": 100, "clip": 1.0, "chunk": 64, "seed": SEED,
                      "script_sha256": TR.file_sha256(__file__)}, "log": [], "monitor": {}}
    if args.resume is None:
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
    base = ck["state"]["update"]
    gens = tuple(torch.Generator(device=dev).manual_seed(SEED + i) for i in (1, 2, 3))   # G1, mask, substitute
    u0 = 0
    if args.resume is not None:                                          # the live state saved at the last read
        rs = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert rs["recipe"] == {k: v for k, v in res["recipe"].items() if k != "updates"}, "resume with another recipe"
        model.load_state_dict(rs["model"]); opt.load_state_dict(rs["opt"]); streams.load_state(rs["streams"])
        carry = tuple(type(c)(*(getattr(c, f.name).to(dev) for f in dataclasses.fields(c))) for c in rs["carry"])
        cache, record = rs["cache"].to(dev), rs["record"].to(dev)
        for g, s in zip(gens, rs["gens"]):
            g.set_state(s.cpu() if s.device.type == "cpu" else s)
        ctl.load_state(rs["controller"])
        u0, res["log"], res["monitor"] = rs["update"], rs["log"], rs["monitor"]
        print("resumed at update", u0, flush=True)
    losses, stats, t0 = [], [], time.time()
    for u in range(u0 + 1, args.updates + 1):
        chunk = streams.next(64)
        ctl.prepare(chunk, streams)
        boot = TR.bootstrap_rows(cfg, WORLD_SEED, base + u)
        for g in opt.param_groups:
            g["lr"] = 1e-4 * min(1.0, u / 100)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), 2):
            loss, c_new, cache_new, rec_new, sx = g1(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                     record[rows], cfg, boot[rows], base + u, gens, args.p, args.substitute, routes,
                                                     check=u == 1)
            (loss / 2).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows], record[rows] = cache_new, rec_new
            losses.append(float(loss)); stats.append(sx)
            if u <= 2:
                res.setdefault("fingerprints", []).append(sx["fingerprint"])
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if u == 1:
            print("peak GiB", torch.cuda.max_memory_allocated() / 2 ** 30, "sec", time.time() - t0, flush=True)
        if u % 100 == 0:
            sub = [s["substitute_mse_by_signal_quartile"] for s in stats[-200:] if "substitute_mse_by_signal_quartile" in s]
            res["log"].append({"update": u, "g1": float(np.mean(losses[-200:])), "masked": float(np.mean([s["masked"] for s in stats[-200:]])),
                               "substitute_mse_by_signal_quartile": [float(np.mean([x[i] for x in sub if x[i] is not None])) if sub else None
                                                                     for i in range(4)],
                               "grad_norm": gn, "sec_per_update": (time.time() - t0) / (u - u0)})
            print(json.dumps(res["log"][-1]), flush=True)
        if u in reads:
            model.eval()
            res["monitor"][u] = TR.monitor(model, cfg, split["monitor"])
            print(f"monitor {u}", json.dumps(res["monitor"][u]), flush=True)
            torch.save(dict(ck, model=model.state_dict(), switching=res["recipe"], switch_update=u), out / f"ckpt_sw{u:04d}.pt")
            if u == max(reads):                                          # the live state, to extend the run later
                ctl.pending = None                                       # the frozen chooser never learns: nothing pending
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "streams": streams.state(), "carry": carry,
                            "cache": cache, "record": record, "gens": [g.get_state() for g in gens], "controller": ctl.state(),
                            "update": u, "log": res["log"], "monitor": res["monitor"],
                            "recipe": {k: v for k, v in res["recipe"].items() if k != "updates"}}, out / f"state_sw{u:04d}.pt")
            model.train()
        (out / "switch_ft.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
