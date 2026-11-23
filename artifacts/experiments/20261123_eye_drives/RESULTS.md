# Where do surprise and contingency send the eye, by day and by night? Results (2026-11-23)

Plan and code: `PLAN.md`, `eye_drives.py`, `foveal_surprise.py`. Outputs: `summary.json`, `drives.npz`,
`foveal_surprise.json`, `foveal_surprise.npz`.
- Frozen P1.5 checkpoint 10 and its fixation controller.
- 41,835 real DEV frames from 160 episodes.
- Night is light < 0.5.

"Lift" is a drive's mass on a window category divided by that category's share of windows on the same frames, so 1 is
uniform.

## First reading: the old chooser's raw surprise and contingency

| | Day | Night |
|---|---:|---:|
| Surprise S, creature windows | 4.32 [4.13, 4.53] | 2.04 [1.99, 2.10] |
| Surprise S, entity-free map windows | 0.54 | 0.77 |
| Surprise S, HUD windows | 1.04 | 1.00 |
| Contingency C, creature windows | 1.54 | 1.21 |
| Contingency C, entity-free map windows | 0.86 | 0.95 |
| The chooser's P on creature windows when one is in view (uniform) | 0.525 (0.204) | 0.351 (0.231) |

By day, raw surprise finds creatures well. At night its selectivity halves, and entity-free cells, which change 0.101
per cell at night against 0.0009 by day, draw its mass. Contingency is broad at any time.

## Second reading: the world's own foveal surprise

The trans-saccadic prediction of the crop from the predicted belief, the periphery and the lighting, scored against
the real crop at every window:

| | Day | Night |
|---|---:|---:|
| NLL lift, creature / entity-free map / HUD | 1.43 / 1.13 / 0.48 | 1.20 / 1.21 / 0.45 |
| MSE lift, creature / entity-free map / HUD | 1.60 / 0.70 / 1.38 | 1.29 / 0.96 / 0.97 |
| Argmax on a creature window when one is in view: NLL / MSE (uniform) | 0.548 / 0.383 (0.204) | 0.394 / 0.396 (0.231) |

- By day, the likelihood-scored foveal surprise picks creatures about as often as the old chooser did.
- At night, its lift is the same for creature and entity-free windows (1.20 against 1.21).
- The pixel decoder's scale reads only lighting and cell position, so its precision is blind to content. It can't
  know that static is noise and a creature isn't.

## What it means

- **Neither drive separates night creatures from night static.** That is a measured fact about this environment and
  these signals. Night is also where the fovea's one known information gain lies (creature recall +5 to 8 points).
- **The chooser's raw-surprise formula is not a sound innate drive here.** The world's own foveal surprise, under
  content-blind precision, isn't either.
- **Bayesian surprise in Itti & Baldi's sense needs content-aware precision:** a model that expects noise where noise
  is. The decoder doesn't have one.
- **What can make the eye find night creatures:**
  - task relevance, since zombies change returns: value of information through dreams and life;
  - or precision that depends on content.
- **Limits:** one checkpoint; real frames only; the readers are the S1 entity grids from labels; window lifts are
  frame-averaged.
