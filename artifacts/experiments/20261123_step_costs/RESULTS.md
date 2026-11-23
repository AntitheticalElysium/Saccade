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
