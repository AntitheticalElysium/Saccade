# What one awake, watched and dreamed step costs on the 6 GB card (2026-11-23)

**Why.** Codex pointed out that an eye that reads the belief can't choose a chunk's looks in one batched pass. The
look at frame t depends on the glimpse at t - 1 through the belief and the context. Watching therefore needs a
sequential pass over each chunk before the training pass, as an awake agent does. Decision 4's budget also needs the
dreamed step's price. This is a timing measurement, not a test of a hypothesis: no training, frozen P1.5 checkpoint 10.

**What is timed** (batch 16 streams, bf16 autocast, no gradient, CUDA-synchronised, after warm-up, median of repeats):
1. **The world's streaming step:** one `Denoiser.step` on a frame at the context signal level, with a full 8-frame
   cache, plus `commit`. That is `context_pass`, the computation the look tokens would ride on.
2. **The glimpse tokens alone:** one `Denoiser.glimpse_step`, the second half of a split look-then-act pass.
3. **The glimpse encoder:** `Perception` on one frame per stream at a chosen window.
4. **The record and belief step:** `s1.history` on one frame, `s1.embed`, `Belief.tokens`, one Mamba step.
5. **A dreamed frame:** `sample` (four evaluations for the coarse view, the crop at a chosen window), the outcome
   heads, the record and belief step, and `context_pass`. Batch 16 and 64.
6. **A Craftax-Classic step and render** on the CPU (JAX), one environment.

**Readings, fixed before results:**
- the sequential look pass per training update, estimated as 256 × (items 1 + 2 + 3 + 4), against the measured P1.5
  update (7.65 s per 256-step, 16-stream update, `20261116_p15/checks_throughput.log`);
- an awake step (items 1 + 2 + 3 + 4 at batch 1 and 16, plus item 6);
- a dreamed step per stream at batch 16 and 64.

**Limits.** The agent tokens don't exist yet, so the look pass is priced by the computation it rides on, a lower bound
on its cost. Random inputs of the right shapes, since the timing doesn't depend on the content. One card.

**Run:** `JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_step_costs.step_costs`
(Craftax is timed in a separate process with JAX on the CPU: `--craftax`).

## Second reading: dream throughput by batch (added 11-23, before running)

**Why.** The user asked whether dreaming is fundamentally cheap. The first reading found the streaming calls
latency-bound up to 16 streams. A dreamed frame per stream fell from 7.3 ms at 16 streams to 3.2 ms at 64. This
reading finds where it becomes compute-bound: the floor of a dreamed step's cost on this card with eager PyTorch.

**What is timed:** the same dreamed frame as item 5 (sampler with 4 evaluations, crop at a chosen window, outcome heads,
record and belief step, context pass) at 64, 128 and 256 streams, plus peak memory.

**Reading:** milliseconds per dreamed step per stream at each batch.

**Run:** `JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_step_costs.dream_scaling`
