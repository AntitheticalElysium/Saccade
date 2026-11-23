"""Time real P1.5 updates without writing into the training run."""

import argparse
import gc
import importlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import MethodType

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from saccade import chooser as C, data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "20261116_p15/run"
ARMS = ("native", "deferred_logging_fused_adam", "temporal_checkpoint_only", "no_checkpoint",
        "larger_microbatch", "retain_one", "retain_two", "flex_temporal", "record_fused_flex")


def temporal_checkpoint(self, x, code, prev_action, memory, fix, episode, valid):
    tok, ang = self.embed(x, memory, fix)
    c = self.conds(code, prev_action)
    win, ang_t = self._window(tok.shape[1], episode, valid, tok.device)
    for block in self.blocks:
        if block.temporal:
            args = (ang_t.expand(tok.shape[0], -1, -1), win, None, self.phys)
            tok = checkpoint(lambda z, b=block, a=args: b(z, c, *a)[0], tok, use_reentrant=False)
        else:
            tok = block(tok, c, ang, None, None, self.phys)[0]
    return self.head(tok, c)


def limited_checkpoint(self, x, code, prev_action, memory, fix, episode, valid, retained):
    tok, ang = self.embed(x, memory, fix)
    c = self.conds(code, prev_action)
    win, ang_t = self._window(tok.shape[1], episode, valid, tok.device)
    for i, block in enumerate(self.blocks):
        args = ((ang_t.expand(tok.shape[0], -1, -1), win, None, self.phys)
                if block.temporal else (ang, None, None, self.phys))
        if i in retained:
            tok = block(tok, c, *args)[0]
        else:
            tok = checkpoint(lambda z, b=block, a=args: b(z, c, *a)[0], tok,
                             use_reentrant=False, preserve_rng_state=False)
    return self.head(tok, c)


def cuda_events():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def initialize(saved, cfg, scratch, arm, steps):
    dev = torch.device("cuda")
    model = M.M0(cfg).to(dev)
    model.load_state_dict(saved["model"])
    model.train()
    model.denoiser.checkpoint = arm != "no_checkpoint" and cfg["p1"]["activation_checkpointing"]
    if arm == "temporal_checkpoint_only":
        model.denoiser.forward = MethodType(temporal_checkpoint, model.denoiser)
    if arm in ("retain_one", "retain_two"):
        retained = (1,) if arm == "retain_one" else (1, 5)
        model.denoiser.forward = MethodType(lambda self, *a: limited_checkpoint(self, *a, retained), model.denoiser)

    config = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=config["lr"], betas=tuple(config["betas"]), eps=config["eps"])
    opt.load_state_dict(deepcopy(saved["opt"]))
    if arm == "deferred_logging_fused_adam":
        for group in opt.param_groups:
            group["fused"] = True
            group["foreach"] = False
        for parameter, state in opt.state.items():
            if "step" in state:
                state["step"] = state["step"].to(parameter.device)

    episodes = D.load_split(cfg)["train"]
    d = cfg["data"]
    streams = D.Streams(episodes, d["streams"], d["context_prefix"], seed=saved["world_seed"],
                        targets_offsets=D.frame_offsets(episodes))
    streams.load_state(saved["streams"])
    ctrl = C.Controller(model, cfg, scratch, dev)
    chooser = saved["fixation_controller"]
    ctrl.head.load_state_dict(chooser["head"])
    ctrl.opt.load_state_dict(deepcopy(chooser["optimizer"]))
    ctrl.gaze.set_state(chooser["gaze_rng"])
    ctrl.batch.set_state(chooser["batch_rng"])
    ctrl.updates, ctrl.frames = chooser["updates"], chooser["frames"]
    torch.set_rng_state(saved["rng"]["cpu"])
    torch.cuda.set_rng_state(saved["rng"]["cuda"])
    tensors = tuple(type(row)(**{k: v.to(dev).clone() for k, v in vars(row).items()}) for row in saved["carry"])
    carry = tensors
    cached_belief = saved["cache"].to(dev).clone()
    record = saved["record"].to(dev).clone()
    micro = d[f"accumulate_at_{steps}"]
    if arm == "larger_microbatch":
        micro //= 2
    return model, opt, streams, ctrl, carry, cached_belief, record, micro


def run(saved, cfg, arm, steps, repeats):
    with tempfile.TemporaryDirectory(prefix="saccade-training-perf-") as scratch:
        model, opt, stream, ctrl, carry, cache, record, micro = initialize(saved, cfg, scratch, arm, steps)
        prefix = cfg["data"]["context_prefix"]
        rows_all = np.array_split(np.arange(cfg["data"]["streams"]), micro)
        result = []
        signature = []
        for i in range(repeats + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            total_start = perf_counter()
            chunk = stream.next(steps)
            ready = perf_counter()
            ctrl.prepare(chunk, stream)
            torch.cuda.synchronize()
            gather_end = perf_counter()
            update = saved["state"]["update"] + i
            boot = TR.bootstrap_rows(cfg, saved["world_seed"], update)
            world_start = cuda_events()
            opt.zero_grad(set_to_none=True)
            forward, backward, loss_log = [], [], []
            for rows in rows_all:
                before = cuda_events()
                terms, new_carry, new_cache, new_record = TR.p1_losses(
                    model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], None,
                    cfg, boot[rows], update, None, record[rows], TR.gaze_rows(ctrl, rows, steps))
                loss = (model.harmoniser(TR.groups(terms)) + terms["noise"]) / micro
                after_forward = cuda_events()
                loss.backward()
                after_backward = cuda_events()
                M.carry_put(carry, rows, new_carry)
                cache[rows] = new_cache
                record[rows] = new_record
                if arm != "deferred_logging_fused_adam":
                    loss_log.append([float(t.detach()) for t in terms.values()])
                if i == 0:
                    signature.append([float(t.detach()) for t in terms.values()])
                forward.append((before, after_forward))
                backward.append((after_forward, after_backward))
            before_opt = cuda_events()
            if i == 0:
                grad = torch.cat([p.grad.detach().flatten()[:64].float()
                                  for p in model.parameters() if p.grad is not None])
                signature.append(grad.cpu().tolist())
            float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["grad_clip"], error_if_nonfinite=True))
            opt.step()
            after_opt = cuda_events()
            torch.cuda.synchronize()
            world_end = perf_counter()
            ctrl.learn(int(chunk["valid"][:, prefix:].sum()))
            torch.cuda.synchronize()
            end = perf_counter()
            if i:
                result.append({
                    "total_ms": 1000 * (end - total_start),
                    "gather_ms": 1000 * (ready - total_start),
                    "chooser_prepare_ms": 1000 * (gather_end - ready),
                    "world_ms": 1000 * (world_end - gather_end),
                    "world_stream_ms": world_start.elapsed_time(after_opt),
                    "loss_forward_ms": sum(a.elapsed_time(b) for a, b in forward),
                    "loss_backward_ms": sum(a.elapsed_time(b) for a, b in backward),
                    "optimizer_ms": before_opt.elapsed_time(after_opt),
                    "chooser_learn_ms": 1000 * (end - world_end),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                })
        return {"microbatches": micro, "measurements": result, "warmup_signature": signature,
                "median": {key: float(np.median([r[key] for r in result])) for key in result[0]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, choices=(64, 256), nargs="+", default=[64, 256])
    p.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS[:5]))
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    result = {}
    for steps in args.steps:
        source = RUN / ("ckpt_04.pt" if steps == 64 else "ckpt_10.pt")
        saved = torch.load(source, map_location="cpu", weights_only=False)
        cfg = saved["cfg"]
        result[str(steps)] = {}
        for arm in args.arms:
            gc.collect()
            torch.cuda.empty_cache()
            dense = M.Block.dense_time
            scan = M.Record.scan
            if arm in ("flex_temporal", "record_fused_flex"):
                flex = importlib.import_module("artifacts.experiments.20261107_f4_followups.flex_bench")

                def use_flex(self, h, ang, win=None, cache=None, phys=None, cur=None, ctx=None):
                    if ctx is not None:
                        return dense(self, h, ang, win, cache, phys, cur, ctx)
                    return flex.flex_dense_time(self, h, ang, win, cache, phys, cur)

                M.Block.dense_time = use_flex
            if arm == "record_fused_flex":
                fused = importlib.import_module("artifacts.experiments.20261123_step_costs.record_scan_perf")
                M.Record.scan = fused.fused_scan
            try:
                outcome = run(saved, cfg, arm, steps, args.reps)
            except torch.cuda.OutOfMemoryError:
                outcome = {"status": "out_of_memory"}
            finally:
                M.Block.dense_time = dense
                M.Record.scan = scan
            result[str(steps)][arm] = outcome
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(steps, arm, outcome.get("median", outcome), flush=True)


if __name__ == "__main__":
    main()
