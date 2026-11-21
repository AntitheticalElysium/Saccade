# P1.6 results (checkpoint 10 against P1.5 checkpoint 10; one seed each)

Written 2026-11-21 while the rest of the evaluation queue is paused (trajectory, mining, sleep, sleep split,
entities and the comparison run after C15). Run-to-run variation on the paired readings has reached 0.12 (F6.1).

## 1. The evaluation queue, steps done

- **S1:** no change. The belief's zombie AP is up 0.01-0.02 at every staleness. One-step reader counts, reward and
  continuation calibration and the TSP NLL agree with P1.5 to the third decimal.
- **Native pairs** (exact-history pairs, the tick in one member only): the drop is drawn when the tick happens.

  | clock | P1.5 | P1.6 | drawn without the tick |
  |---|---|---|---|
  | hunger (24 pairs) | 0.20 | **0.66** | 0 in both |
  | zombie-hit cooldown (32 pairs) | 0.87 | 0.72 | 0.005 -> 0.020 |

- **Consequences** (the counter's next value drawn after the action): coal 0->1 0.73 -> 0.89, coal 1->2 .. 4->5
  roughly doubled (0.08-0.12 -> 0.15-0.28), iron 0->1 0.36 -> 0.67; stone mostly up, wood mixed. Waking beside a
  zombie unchanged (0.93 / 0.89 when ready).
- **Interface** (the reward head on generated next frames): collect coal 0.75 -> 0.93, collect iron 0.40 -> 0.72;
  defeat zombie 0.71 -> 0.66, defeat skeleton 0.66 -> 0.61, eat plant 0.50 -> 0.43, adjacent events 0.68 -> 0.61,
  collect stone 0.99 -> 0.95.
- **Recursion** (128 windows x 4, entity count): free-running at steps 1, 4, 8, 16: P1.5 1.00, 0.97, 0.81, 0.80;
  P1.6 1.01, 0.92, 0.77, 0.68 (truth 1.11, 1.22, 1.33, 1.27). Mobs fade slightly faster.
  - The real-replay identity passes on the same-precision reference (max 0.027). Against encode's full-precision
    belief it reaches 0.054, against P1.5's 0.016: P1.6's belief is more sensitive to fp16 token storage in its tail
    (run notes).
- **Belief probe:** food drop AP 0.048 -> 0.063 (CIs disjoint), recovery 0.81 -> 0.87, energy up asleep 0.62 -> 0.64.
- **Natural thirst** (one case): contrast 0.195 -> 0.133. **Movement:** identical.

## 2. Does the world use the foveal record? (reading 2, `readings/reliance.json`)

The G1 x-error on C2's 3,000 DEV frames (monitor noise and prior fixations), relative increase with the fovea group
zeroed in the denoiser's read, and in both reads:

| region | day | night | looked at t - 1 | unlooked 8 frames |
|---|---|---|---|---|
| mob | +0.5% | +1.2% | +1.4% day, +2.6% night | +0.1% day, +0.3% night |
| changed HUD | +1.3% | +0.2% | +4.1% day (55 cells) | +1.0% day |
| static HUD | -0.1% | +3.8% | +1.2% day, +5.9% night | 0.0% day, +2.6% night |
| entering terrain | +0.1% | +0.1% | | |
| terrain edits (177 cells) | +13.8% | +4.7% | | |
| static map | +0.4% | +0.4% | | |

- Zeroing Mamba's read as well adds nothing over the denoiser's.
- Prediction (2) holds in direction only: largest at night and at cells just looked at. The size is a few percent.
  The world barely uses the record it was given. A zeroed input usually overstates reliance; this is not hidden use.

## 3. Does a look protect a mob? (readings 4 and 6, `readings/calibration.json`)

C5's 2,079 units, 8 draws, each world's own S1 kind reader. The P1.5 control reproduces C5's keeps exactly (1.0).
"Looked" means the mob's cell was in frame t - 1's fixation window: the monitor's random prior, 15.6% of units.

| deletion | P1.5 looked | P1.5 unlooked | P1.6 looked | P1.6 unlooked |
|---|---|---|---|---|
| cows, day | 5.9 +- 0.8% | 6.0 +- 0.4% | 5.1 +- 0.7% | 6.1 +- 0.4% |
| cows, night | 26.6 +- 4.9% | 22.6 +- 2.1% | 26.9 +- 5.4% | 23.3 +- 2.2% |
| zombies, day | 0.9 +- 0.5% | 1.8 +- 0.3% | 2.9 +- 1.0% | 2.3 +- 0.4% |
| zombies, night | 7.4 +- 3.0% | 12.5 +- 1.8% | 10.1 +- 3.6% | 13.8 +- 1.9% |
| skeletons, day | 16.3 +- 3.8% | 14.4 +- 1.7% | 17.3 +- 3.9% | 18.8 +- 2.0% |

- Prediction (4) fails: looked-at night mobs are kept no better.
- The looked / unlooked gap is position, not protection (Codex 11-21, checked). The window centres are uniform over
  the 49 positions, so cells near the edge are looked at less: 13.4% of units at C5's edge ring against 17-19% inside.
  Edge mobs are deleted far more at night. Within each band the gap is gone:
  - night zombies, P1.5: edge 31.2% looked (6 units) against 32.5% unlooked (53); inner 2.8% (31) against 5.2% (145);
  - P1.6: edge 37.5% against 34.4%; inner 4.8% against 6.3%.

  This agrees with A5's paired on / away test on P1.5 (-0.1 +- 0.1 points). The denoiser's physical tokens never
  attend to glimpse tokens (the factorised mask), so the context glimpse cannot reach the drawn periphery directly.
- The day coin is unchanged (6.0 -> 6.1%), as predicted.
- Prediction (6) holds: spurious mobs per empty cell are 1.48% inside the looked window against 1.33% outside in
  P1.6, and 1.50% against 1.36% in P1.5. No excess from G1.

## 4. The added test: the counter gains with the record writing no glimpse (`readings_blank_fovea/`)

`P16_BLANK_FOVEA=1`: the fovea group exactly zero in both reads. The blank check (one DEV episode) passes: the fovea
group's maximum goes from 0.86 to 0, the periphery group is unchanged (0.0), and Mamba's read changes (0.21).

| | P1.5 | P1.6 | P1.6 blanked |
|---|---|---|---|
| hunger tick drawn (24 pairs) | 0.203 +- 0.019 | 0.661 +- 0.029 | 0.720 +- 0.023 |
| cooldown hit drawn (32 pairs) | 0.866 +- 0.041 | 0.715 +- 0.051 | 0.708 +- 0.050 |
| coal 0->1 / 1->2 / 2->3 | 0.725 / 0.117 / 0.102 | 0.893 / 0.280 / 0.184 | 0.893 / 0.293 / 0.180 |
| iron 0->1 | 0.363 | 0.668 | 0.673 |
| woken beside a ready zombie | 0.93 | 0.89 | 0.90 |

Every transition, wood and stone included, is within 0.03 of intact. **The counter gains do not come through reading
the foveal record.** What this does not exclude is G1's presence during training changing what the rest of the world
learned.

## 5. When the gains appear: the trajectory over the ten checkpoints (`readings/trajectory/`)

Native contrast (tick drawn minus no-tick drawn) at each checkpoint. The world-eye acts from pass 7, after
checkpoint 6.

| checkpoint | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| hunger P1.5 | 0.26 | 0.14 | 0.02 | 0.21 | 0.06 | 0.13 | 0.42 | 0.20 |
| hunger P1.6 | 0.11 | 0.19 | 0.20 | 0.70 | 0.23 | 0.44 | 0.52 | 0.66 |
| recovery P1.5 | 0.65 | 0.58 | 0.47 | 0.51 | 0.63 | 0.55 | 0.75 | 0.75 |
| recovery P1.6 | 0.48 | 0.54 | 0.76 | 0.84 | 0.74 | 0.88 | 0.87 | 0.86 |
| cooldown P1.5 | 0.30 | 0.53 | 0.57 | 0.77 | 0.80 | 0.82 | 0.80 | 0.86 |
| cooldown P1.6 | 0.26 | 0.51 | 0.50 | 0.55 | 0.64 | 0.63 | 0.70 | 0.70 |
| food-drop AP P1.5 | 0.043 | 0.045 | 0.047 | 0.046 | 0.046 | 0.046 | 0.047 | 0.048 |
| food-drop AP P1.6 | 0.044 | 0.044 | 0.043 | 0.050 | 0.053 | 0.056 | 0.059 | 0.063 |

- Hunger, recovery and cooldown part ways at checkpoints 4-6, before the world-eye acts. **They are not the
  world-eye's.** The food-drop AP alone climbs from checkpoint 6 on.
- Both runs swing hard between neighbouring checkpoints (P1.6 hunger 0.70 -> 0.23 -> 0.44). One checkpoint's pairs
  can mislead. P1.6's hunger lead nonetheless holds at all seven checkpoints from 4 on.
- The two runs share the recipe, schedule, world seed and initialisation (G1's columns start at zero). They differ by
  G1's presence in training and by GPU nondeterminism. Which of the two produced the divergence is not separable
  without a replicate.

## 6. The rest of the queue

- **Entities** (20,000 tuples, persisting and entering mobs): P1.6 equals P1.5 within 0.01-0.03 everywhere, both
  level with F4. Night skeletons are slightly lower (near 0.464 -> 0.431).
- **Sleep, sleep split:** within their CIs (early wake 0.023 -> 0.082, CIs overlapping; the phase test is closer to
  the simulator, 0.67 -> 0.53).
- **Mining** (201 coal pairs, the DO / NOOP factorial): the drawn coal increment doubles, 0.104 -> 0.218 (CI 0.18-0.26
  against 0.08-0.12); memory and denoiser sides are each sufficient as in P1.5. Wood unchanged (0.916 -> 0.903).
- **Reading 5, C7's zombie probe:** unchanged. Per-cell AP within two cells: belief 0.116 -> 0.118, predicted belief
  0.071 -> 0.072. Prediction (5) fails: the belief holds no more about which cell.
- **Reading 7, the eye in training** (C11b on the 400 labelled training episodes, chooser against paired uniform):
  - Passes 1-6 track P1.5's chooser within about 0.01.
  - From pass 7, under the world-eye: entity coverage up 2-3 points (any entity 0.47 against 0.45, cows 0.41 against
    0.37, entering terrain 0.46 against 0.43; zombies equal), changed-HUD coverage down 8 points (0.35 against 0.43).
  - Prediction (7) holds as stated. The eye looked more at mobs, and the world gained nothing on mobs from it.
- **Clocks (gradient budget, checkpoint 10):** the reward forecast's loss is 24% lower (0.066 -> 0.050). The reward
  heads' share of the belief's gradient is up (reward cosine to the total 0.50 -> 0.71), and the belief's total
  gradient norm is up 49%. That fits a better-timed hunger clock, since the reward forecast is the loss that asks for
  clocks (11-13). It is an association at one checkpoint, not a cause.

## What this says

- **G1's stated purpose fails.** The foveal record is barely read: a few percent at most, and blanking it changes no
  counter reading. It does not protect mobs, and creates no spurious ones.
- **P1.6 is better than P1.5 on counters.** Hunger timing, recovery, coal and iron increments, and the food-drop
  belief all beat the 0.12 run-to-run bar, at every late checkpoint where measured. They are not carried by the
  record, and the clock divergence is not the world-eye's. That leaves G1's effect on training, or this run's
  trajectory: one seed cannot separate them.
- **P1.6 is flat or slightly worse on mobs:** cooldown hits, free-running counts at step 16, zombie and skeleton
  rewards, night skeleton persistence.
- **The day coin is unchanged (6.0 -> 6.1%).** C15 on P1.5: interval guidance cuts it by a third (-2.14 +- 0.21) but
  takes the history-free prediction's error into the HUD (x37) and into spurious mobs (+28%). Restart does nothing.
  By the declared reading neither is a remedy. Details in `../20261120_switching_dissection/RESULTS.md`, C15.
