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
- The looked-at protection of night zombies is already in P1.5, whose record holds no glimpse. The glimpse in the
  context frames gives it; the foveal record adds nothing.
- The day coin is unchanged (6.0 -> 6.1%), as predicted.
- Prediction (6) holds: spurious mobs per empty cell are 1.48% inside the looked window against 1.33% outside in
  P1.6, and 1.50% against 1.36% in P1.5. No excess from G1.

## What this says so far

- G1's stated purpose fails. The foveal record is barely read (a few percent at most) and does not protect mobs.
- The counter gains (hunger timing, coal and iron increments) are real against the 0.12 run-to-run bar.
  - They do not come from reading the foveal record: changed-HUD reliance is +1.3%.
  - Two causes remain open: the world-eye's training distribution from pass 7, or run chance. Coal increments have
    jumped abruptly between checkpoints before (F4, checkpoints 4 -> 5; increment regression 11-19).
- The mob readings are flat or slightly worse: cooldown hits, free-running counts, zombie and skeleton rewards.
</content>
</invoke>
