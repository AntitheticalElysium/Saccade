# What one awake, watched and dreamed step costs: results (2026-11-23)

Plan and code: `PLAN.md`, `step_costs.py` (commit 4cbde129). Outputs: `step_costs.json`, `craftax.json`.
- Frozen P1.5 checkpoint 10 on the RTX 3060.
- bf16 autocast, no gradient, median of 50 synchronised repeats after warm-up.

| Milliseconds per call | 1 stream | 16 streams | 64 streams |
|---|---:|---:|---:|
| World's streaming step (`Denoiser.step` + `commit`) | 13.2 | 13.7 | 29.1 |
| Glimpse tokens alone (`glimpse_step`) | 8.7 | 8.4 | 8.7 |
| Glimpse encoder (`Perception`) | 1.3 | 1.3 | 1.3 |
| Record and belief step | 12.7 | 13.2 | 14.0 |
| Dreamed frame (sample, outcome, belief, context pass) | 115 | 117 | 202 |
| Craftax-Classic step and render, CPU, one environment | 0.9 | | |

Peak GPU memory 0.5 GiB at 64 streams.

## Reading

- **The streaming calls are latency-bound.** Their cost is flat from 1 to 16 streams, so batching more streams is
  nearly free up to 16, and costs about 2x at 64 for the world's step only.
  - Launch overhead, not arithmetic, sets the price.
  - CUDA graphs or compilation could lower it. That is unmeasured.
- **The sequential look pass for watching,** 256 frames × (world step + glimpse tokens + encoder + record and belief):
  - 9.4 s for 16 streams, against P1.5's measured 7.65 s update: +122%;
  - 13.6 s for 64 streams, which is four updates' worth of looks: +44% per update. The look policy then lags by up to
    three updates, and its action probabilities are logged.
- **An awake step at one stream is about 37 ms** (36 ms on the GPU, 0.9 ms Craftax): 10 h per million real steps.
  - With 16 environments stepped together it is about 2.3 ms of GPU per real step, 0.6 h per million, plus Craftax.
- **A dreamed step costs 7.3 ms per stream at 16 streams, and 3.2 ms at 64.**
  - Starting a 15-step dream from every replayed frame (DreamerV3) at replay ratio 16 is 240 dreamed steps per real
    step. That is about 0.77 s per real step, or 210 h per million at 64 streams: out of reach.
  - One dream per replayed context (Dreamer 4) is 240 dreamed steps per update of 4,096 frames. At ratio 16 that is
    about 0.94 dreamed steps per real step, or 0.8 h per million.
  - Both exclude the backward pass of the agent tokens' learning.
- **For comparison:** world training at replay ratio 16 is 16 × 1.87 ms, about 30 ms per real step, or 8.3 h per
  million (P1.5's 256-step update rate).

## Limits

- The agent tokens don't exist yet, so the look pass is priced by the computation it rides on: a lower bound.
- The timings use random inputs of the right shapes, on one card, with eager PyTorch.

## P1.5 training and three eye paths on the 6 GB RTX 3060 Laptop GPU (11-23)

The 625 regular intervals in the original P1.5 training log sum to **15.31 h** when weighted by
their update counts, including the one-step initial reading. Median update time is **2.284 s**
for 64-frame chunks and **8.581 s** for 256-frame chunks. The projection retains both halves
of the 25.6M-frame recipe.

### Real training updates

Restarted checkpoint 4 at 64-step shape and checkpoint 10 at 256-step shape in scratch. Timed
the same real streams and original losses with one warmup and two measured updates per arm.
Both short panels have shortcut self-consistency active. Timing is a lower bound on additional
full-run experiment overhead such as new heads and checkpoint evaluations.

| Native median, ms per update | 64 frames | 256 frames |
|---|---:|---:|
| Total | 2,238 | 8,676 |
| Gather | 10 | 187 |
| Old chooser prepare and learn | 66 | 266 |
| World update, including forward/backward and optimizer | 2,162 | 8,223 |
| Loss forward | 765 | 3,022 |
| Backward with recomputation | 1,386 | 5,181 |
| Optimizer | 7.45 | 7.73 |

- Removing the old chooser's measured prepare/learn projects to **0.46 h** over the historical
  update counts. The world-update measurement includes the old gaze-forecast loss.
- Fused AdamW and deferring scalar loss reads cut the optimizer to about 4.5–4.8 ms. The short
  panels show no consistent end-to-end speedup.
- Doubling the microbatch, turning off checkpointing, or checkpointing only temporal blocks
  exhausted memory at both shapes. With expandable segments, retaining one spatial block's
  activations ran in 2.206 s and 8.728 s, against 2.253 s and 8.810 s with the same allocator
  and native checkpointing, a **0.23 h** gain if sustained. A matched follow-up saved 23 ms
  at 64 steps and cost 184 ms at 256, with allocator-mapping warnings at the long shape.
  The initial-batch scalar losses matched exactly; the relative norm difference across
  16,355 sampled gradient elements was 0.11%/0.36%. Reserved memory rises to about 4.50/4.61 GiB.
  Retaining two blocks exhausted memory. Budget this as an inconclusive speed candidate.

### Sequential look and belief update

All arms use frozen checkpoint 10, bf16, synthetic inputs at the real shapes, and warmed
blocks of repeated device calls. The old proxy uses the full native denoiser step followed by
the glimpse-only pass as charged in the first reading. The native Mamba wrapper discovers
variable-length segment boundaries using `nonzero` and selects final rows by a Boolean mask,
even when a one-frame call is known to contain one segment per stream.
The benchmark constructs that same packed layout directly. Its outputs and returned state
matched the native wrapper exactly on the checked shapes. Capturing that proxy also gives
exact checked output equality. The stateful split and lightweight eye are checked across three
different consecutive input frames with recurrent buffers carried in eager and captured modes.

| Milliseconds per sequential step across a batch | 16 streams | 32 streams | 64 streams |
|---|---:|---:|---:|
| Native eager proxy | 37.32 | 36.30 | 47.41 |
| Captured native-computation proxy | 10.94 | 19.61 | 35.44 |
| Captured split denoiser with cache and belief carried | 9.36 | 16.76 | 30.48 |
| Captured lightweight recurrent eye with actual crop sampling | 1.96 | 2.77 | 4.86 |

The split computes the physical stack before the crop and reuses its layer keys and values
when scoring the selected crop. It agrees with the original full step to at most 3.3e-6 on
percept outputs with a common FP32 math-attention backend. Default bf16 execution differs by
about 0.0030–0.0037 RMS on periphery and up to 0.0313 at an element; cache deltas are present.
The split graph and its own eager path matched exactly on all three-step checks. The split's
native-denoiser bf16 equivalence over long trajectories remains a required validation.

The lightweight readout is an untrained 103,089-parameter MLP on current coarse evidence and
the same Mamba belief, including prior glimpses through recurrent assimilation. Its graph
includes current periphery, drawing the on-device window with an exploration floor, foveal
encoding, the record update and Mamba state update. It does not run a denoiser for each look.
This is a compute counterfactual, not a learned gaze-quality result. It compares what the look
reads; the world still uses its full training objective.

### Projection at the same 25.6M frames

Add a look cost of `measured milliseconds × 25.6M / batch / 3.6M` hours after subtracting the
measured old chooser from the historical 15.31 h. The table excludes unbuilt agent-token work,
new gaze and imitation losses, caching and interleaving overhead, and additional fine-tuning.
The split and native proxy are different pieces of computation, as described above.

| Look path | Loop batch | Eye pass hours | Total proxy hours |
|---|---:|---:|---:|
| Full eager proxy | 16 | 16.59 | 31.44 |
| Full eager proxy | 64 | 5.27 | 20.12 |
| Captured full proxy | 64 | 3.94 | 18.79 |
| Captured split and stateful proxy | 64 | 3.39 | 18.24 |
| Captured recurrent eye | 32 | 0.61 | 15.46 |
| Captured recurrent eye | 64 | 0.54 | 15.39 |

These components were measured separately. Keeping capture buffers resident alongside the
world backward pass must fit the same 6 GB before combining the results. A batch of 64
needs independently runnable histories and a defined model-version
policy if their gaze choices feed several learner updates. With the light eye, 32 costs about
five minutes more at this frame budget. Denoiser/critic training in dreams and real simulation
time are independent of these watching-phase estimates.

Rerun with `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 .venv/bin/python -m`,
followed by any of:

- `artifacts.experiments.20261123_step_costs.streaming_perf --count 24`
- `artifacts.experiments.20261123_step_costs.split_perf --count 20`
- `artifacts.experiments.20261123_step_costs.recurrent_eye_perf`
- `artifacts.experiments.20261123_step_costs.training_perf`

For the selective checkpoint comparison, add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
and pass `--arms native retain_one retain_two --out artifacts/experiments/20261123_step_costs/training_selective.json`
to the last command.

## Second reading: dream throughput by batch (`dream_scaling.py`, `dream_scaling.json`)

| Streams | Dreamed frame (ms) | Per stream (ms) | Peak GPU memory (GiB) |
|---:|---:|---:|---:|
| 16 (first reading) | 117 | 7.3 | 0.3 |
| 64 | 202 | 3.15 | 0.56 |
| 128 | 372 | 2.90 | 0.97 |
| 256 | 715 | 2.79 | 1.85 |

- **A dreamed step becomes compute-bound by about 64 streams.** Its floor on this card, in eager PyTorch, is about
  2.8 ms per stream. That's roughly 1.5 times the 1.87 ms it costs to *train* on a real frame, backward included.
  - A dreamed frame runs the sampler's four evaluations, the crop's cascade and the context pass: about six denoiser
    passes.
  - Training runs one forward and one backward, batched along time.
- **For comparison:** a Craftax-Classic step and render costs 0.9 ms on one CPU core.
  - DayDreamer's quadruped acts at 20 Hz (50 ms per real step) and its arm at 2 Hz (500 ms; PDF, section 3).
  - A dreamed step is therefore about 3x the cost of a Craftax step, but 18x and 180x faster than those robots' real
    steps. Dreams also run 256 at once.

## Full-eye reuse and whole-world kernels (11-23)

The supplied fixation schedule and available denoiser inputs are matched in this study. The
sequential split is a frozen-model proxy for the unbuilt agent look and act tokens.

### Prepare gaze-independent work once

On P1.5 the record reads only the periphery. The benchmark prepares the periphery, lighting,
record scan and both record readouts for a chunk, then runs the same captured split denoiser and
Mamba sequentially on the chosen crop history. It compares the unprepared and prepared versions
on eight consecutive synthetic frames with identical fixations.

| Streams | Streaming graph, ms/step | Prepared graph, ms/step | Preparation, ms per chunk | Prepared total, ms/step |
|---:|---:|---:|---:|---:|
| 16 | 9.24 | 8.37 | 6.64 | 9.20 |
| 64 | 31.41 | 27.80 | 24.61 | 30.88 |

After charging the preparation pass, this buys only about 0.06 h over 25.6M frames at 64
streams. Rebatching bf16 projections also changes the checked trajectory outputs and belief,
with an aggregate max absolute difference of 0.156 on the short 64-stream sequence. The
mathematical dependency allows batch preparation. The proved gains and numerical equivalence
of this particular implementation remain limited.

Within each native dreamed frame, the record projection is identical across the four physical
calls, the full call at the start of the crop cascade and the context pass. Reusing it once
preserved the sampled output and committed cache exactly with the same random seed. The sampler
and context pass together took 114.34 → 112.67 ms at 16 streams and 179.52 → 178.45 ms at 64.
The benchmark excludes policy and value work and used eager dispatch in both arms.

### Fuse the fixed record scan

One Triton kernel computes the same per-cell complex recurrence and both its before- and
after-frame outputs, with episode resets. It retains the original channel and frequency state.

| Shape, streams × frames | Native ms | Fused ms | Max absolute state difference |
|---|---:|---:|---:|
| 8 × 72 | 15.49 | 1.31 | 3.88e-7 |
| 2 × 264 | 41.47 | 1.21 | 4.18e-7 |
| 64 × 8 | 13.09 | 1.20 | 1.79e-7 |
| 64 × 1 | 1.77 | 0.22 | 5.97e-8 |

These isolated timings include allocation and launch overhead. The world update below is
the appropriate scale for projecting training savings.

Reproduce with the environment settings above and modules
`artifacts.experiments.20261123_step_costs.reuse_perf` and
`artifacts.experiments.20261123_step_costs.record_scan_perf`.

### Combined real-update test

Used the original P1.5 model weights, real training streams, crop targets, loss, and
optimiser in scratch. Temporal FlexAttention evaluates the same causal local window and
physical/glimpse masks without materialising the same window's keys and values for each
query position. The fused record replaces the native fixed recurrence only. Both arms
use the expandable allocator and three timed updates after one warmup.

| Update shape | Native median | Flex + fused record median | Change | Peak reserved GPU, native → combined |
|---|---:|---:|---:|---:|
| 64 frames | 2.237 s | 2.068 s | −7.6% | 3.83 → 3.30 GiB |
| 256 frames | 8.438 s | 8.242 s | −2.3% | 3.93 → 3.19 GiB |

The three combined long updates were 8.24, 8.05 and 11.82 s. The timing panel is short
and remains sensitive to compiler warmup of different bootstrap sub-batch shapes and
execution variability. Same first-batch loss terms differ by up to 0.00035 and 0.00083 at
the two shapes. Relative L2 differences across sampled gradients are 0.39% and 2.72%.
The masks and recurrence are preserved mathematically and bf16 execution changes some
outputs. Native rollout and new gaze parity remain required before switching training.

Projecting the measured medians to the historical update counts gives about **0.76 h**
of world-loop savings. Applied to the previous captured 64-stream full-eye split
projection, that is **17.48 h at 25.6M frames**, before unbuilt choice work and extra
watching. The two timing runs still use separate graph and backward memory pools.
At this rate 20.48M frames project to roughly 14.0 h under proportional scaling. That
estimate says nothing about whether the new world or eye is sufficiently trained there.

Reproduce with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in addition to the
environment above, plus module
`artifacts.experiments.20261123_step_costs.training_perf --arms native record_fused_flex --reps 3 --out artifacts/experiments/20261123_step_costs/training_full_eye_optimizations.json`.
