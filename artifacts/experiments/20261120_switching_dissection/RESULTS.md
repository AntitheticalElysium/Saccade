# Switching dissection, stage A (events.py on S2's saved grids; `events.json`, `events.log`)

The data are 4,000 starts x 8 futures per world, read by the same reader with the same plans. The tables give cells per
1,000 transitions, pooled over day and night. SEs come from an episode bootstrap paired against P1.5.

Abbreviations:
- P1.5: the base world.
- R2k / R4k: the rerun `sw_rec2` at 2,000 / 4,000 updates.
- A1: the first, re-warmed arm `sw_rec` at 2,000.

## 1. The extra appearances are spawns, not jumps (candidate 5 refuted)

| Cows, per 1,000 transitions | Step | Simulator | P1.5 | R4k | A1 |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Interior spawn (no vanishing within 3 cells) | 1 | 6.3 | 7.0 | 13.4 | 19.6 |
| Interior spawn | 9-16 | 6.7 | 7.9 | 17.1 | 26.3 |
| Interior jump (a vanishing within 3 cells) | 1 | 1.3 | 1.5 | 2.1 | 2.8 |
| Interior jump | 9-16 | 1.3 | 0.8 | 1.8 | 2.6 |
| Edge arrival | 1 | 50.1 | 21.0 | 32.8 | 44.7 |
| Edge arrival | 9-16 | 55.2 | 22.6 | 33.9 | 49.8 |
| Unpaired vanishing | 1 | 3.4 | 37.0 | 37.5 | 48.7 |
| Unpaired vanishing | 9-16 | 3.4 | 16.0 | 22.4 | 31.9 |

- **R4k:** spawns are about 2x P1.5 (+6.4 at step 1, +8.5 to +9.2 deeper, SE 0.5-0.6). Jumps grow by only +0.5.
- **Zombies:** spawns +1.8 to +2.3 per 1,000 against P1.5's 3.5-4.8 (simulator 3.2-4.0). Jumps +0.2 to +0.5.
- **Skeletons:** spawns +0.9 to +1.1.

## 2. The change is present at step 1, from a real history

At step 1 the record and the context are built from real frames only. R4k already differs from P1.5 there:
- **Cows:** interior spawns 13.4 against 7.0, and edge arrivals 32.8 against 21.0.
- **Start-zombie hazard:** 0.084 against 0.070 (+0.014, SE 0.003).

| Hazard per step of the start's own mobs | Step 1 | 2-4 | 5-8 | 9-16 |
| :--- | ---: | ---: | ---: | ---: |
| Zombies: simulator | 0.014 | 0.008 | 0.005 | 0.003 |
| Zombies: P1.5 | 0.070 | 0.029 | 0.021 | 0.017 |
| Zombies: R4k | 0.084 | 0.046 | 0.037 | 0.030 |
| Cows: simulator | 0.012 | 0.006 | 0.005 | 0.005 |
| Cows: P1.5 | 0.095 | 0.065 | 0.060 | 0.052 |
| Cows: R4k | 0.098 | 0.070 | 0.064 | 0.059 |
| Skeletons: simulator | 0.025 | 0.013 | 0.010 | 0.018 |
| Skeletons: P1.5 | 0.228 | 0.147 | 0.107 | 0.129 |
| Skeletons: R4k | 0.264 | 0.177 | 0.148 | 0.197 |

- **R4k against P1.5, zombies:** +0.014 / +0.017 / +0.016 / +0.013 (SE 0.001-0.003), so 50-75% faster.
- **Cows:** +0.004 to +0.008 (SE about 0.002).
- **Skeletons:** +0.030 to +0.068 (SE 0.006-0.019).
- **R2k:** the zombie excess is +0.008 to +0.010, so it grows with training.

The excess hazard is roughly constant with depth: a one-step property, not a compounding one. Nothing shows the
intended effect, which would be a lower hazard at depth from robustness to self-fed records. Vanishings at depth rise
instead (cows +3.5 to +6.4 per 1,000).

## 3. What the base world does at step 1 (the fade itself)

From a real history, P1.5 already:
- vanishes cows at 11x the simulator's rate (37.0 against 3.4 per 1,000 transitions);
- draws 0.42x its edge arrivals;
- loses start cows at 0.095 per step against 0.012, and 5-6% per step after that (about 10x).

The route test (`20261119_s2/routes_readers.json`) shows that, with a real record, P1.5 redraws the mobs it lost:
- record_real holds 0.93-0.99 mobs per frame through step 16;
- free running falls to 0.55.

So the fade is a large one-step loss rate. The real record hides it by restoring the lost mobs; the record of its own
frames cannot.

## 4. Arrivals and spawns move together

| Cows, per 1,000 transitions | Simulator | P1.5 | R4k | A1 |
| :--- | ---: | ---: | ---: | ---: |
| Edge arrivals | 50-55 | 21-23 | 33-35 | 45-50 |
| Interior spawns | 6-7 | 7-9 | 13-18 | 20-26 |

- Both rise together across the three worlds, as if moved by one setting: how readily the denoiser draws a mob where
  the record holds no evidence of one.
- P1.5 under-draws at the edge (newly revealed cells) but is about right in the interior (cells seen empty).
- A1 nearly reaches the simulator's arrivals and triples interior spawns.
- A world must treat "not yet seen" differently from "seen empty"; one setting cannot do both.

## Reading against the candidates (PLAN.md)

- **(5) Motion: refuted** (section 1).
- **(2) Mismatch** does not explain the harm. The changes appear at step 1 with a real record (section 2). It may still
  explain the absence of the intended gain; that is open.
- **(1) Shrinkage:** all four predictions hold.
  - The change is there at step 1.
  - The extra appearances are spawns.
  - The start's own mobs die faster.
  - The conditioning-driven held-out loss rises (level 0 +2.5%, level 1 +1.6%).

  This is a description so far, not a measured mechanism. Stage B measures how much the denoiser relies on the record
  directly.
- **(3) Dose** and **(4) Recipe:** not addressed by these data.

## Stage A2: deleted mobs coming back (`resurrect.py`, `resurrect.json`, `resurrect.log`)

The table gives the share of the start's own mobs that, after the tracker loses them, have a same-kind mob within 1
cell of their last position at the next 1-2 steps. R4k's change against P1.5 is in brackets (SE).

| Recovery rate | Simulator | P1.5 | R2k | R4k |
| :--- | ---: | ---: | ---: | ---: |
| Cows, lost at step 1 | 0.358 | 0.118 | 0.162 | 0.166 (+0.048 ± 0.009) |
| Cows, lost at steps 2-14 | 0.442 | 0.150 | 0.210 | 0.207 (+0.056 ± 0.007) |
| Zombies, lost at step 1 | 0.429 | 0.097 | 0.121 | 0.123 (+0.025 ± 0.020) |
| Zombies, lost at steps 2-14 | 0.500 | 0.162 | 0.183 | 0.180 (+0.017 ± 0.015) |
| Skeletons, lost at steps 2-14 | 0.500 | 0.182 | 0.265 | 0.232 (+0.050 ± 0.028) |

- **The simulator's rate (0.36-0.50) is the instrument's.** Its "losses" are mostly reader misses of a mob that is
  still there.
- **The lesson landed partly, for cows.** Deleted cows come back about 40% more often (+5 to +6 points, 5-8 SE).
  - Limit: this cannot tell a redrawn cow from a spawn at a plausible nearby cell.
  - Spawns doubled, so part of the gain may be spawns.
- **For zombies it did not land measurably:** +2 points at about 1 SE, while zombie deaths rose 50-75%.
- **Net:**
  - Cows: more deletions, somewhat more recovery, and twice the spawns.
  - Zombies: more deletions, the same recovery.

  Switching taught a little of the intended recovery. The distrust it taught as well outweighs it.

# Stage B (user 11-20: plan and run both)

## A3: where P1.5's step-1 deletion lives (`deletions.py`; isolated inner start mobs; deletion = no same-kind mob within 1 cell)

| Step-1 deletion | Simulator through reader | P1.5 |
| :--- | ---: | ---: |
| Cows, all | 0.010 | 0.098 |
| Cows, day | 0.004 | 0.062 |
| Cows, night | 0.032 | 0.226 |
| Cows, day, inner | 0.006 | 0.052 |
| Cows, day, edge ring | 0.002 | 0.070 |
| Cows, night, edge ring | 0.040 | 0.291 |
| Zombies, all | 0.014 | 0.069 |
| Zombies, night, edge ring | 0.047 | 0.331 |
| Skeletons, all | 0.026 | 0.236 |

- **Night is the largest factor.** Craftax-Classic's renderer adds fresh per-pixel random static at night. Its intensity
  grows with darkness and with distance from the view's centre (`craftax_classic/renderer.py`, `night_noise_intensity_texture`).
  That is why the night and edge-ring effects come together.
- **In daylight a base loss remains.** Inner cows with the camera still are lost at 0.048 (floor 0.000), even when the
  simulator says they almost surely stay (0.042-0.057).
- **Motion uncertainty matters for zombies.** Zombies that certainly stay are lost at 0.014 (by day 0.000); those whose
  next cell is uncertain at 0.118-0.135.
- **Scroll matters little** once the reader's own error on scroll steps is subtracted (floor 0.033-0.036). Zombies are
  about 2x; cows show none.
- **R4k** is the same or slightly worse in every split.

## A4: a glimpse barely protects (`fovea.py`)

**Step 1 (a natural experiment).** The last real frame's glimpse was a seeded uniform draw. Deletion, glimpsed against
not, in P1.5:

| Step-1 deletion | Glimpsed | Not glimpsed | SE of the difference |
| :--- | ---: | ---: | ---: |
| Cows, day, inner | 0.058 | 0.048 | |
| Cows, day, ring | 0.049 | 0.065 | |
| Cows, night, inner | 0.125 | 0.131 | |
| Cows, night, ring | 0.219 | 0.311 | 0.064 |
| Zombies, day, inner | 0.014 | 0.017 | |
| Zombies, night, ring | 0.388 | 0.307 | |

- At the night ring, cows differ by -0.092 at 1.4 SE, while zombies at the night ring go the other way.
- The rerun shows the same pattern, but it is not independent: the same starts and the same glimpses.

**Later steps (observational: the chooser picks).**
- Cows: 0.047 against 0.055 by day, 0.084 against 0.108 at night.
- Zombies and skeletons: no difference.

**Reading.** At step 1 the world has full information: the cow is in the real tokens, the record and the belief. It
still loses inner daylight cows at about 5%. So the deletion is not missing information a glimpse could supply. The only
hint of protection is where the periphery token is corrupted (the night static at the edges). A5 tests that.

## K sweep: the sampler is a minor part (`ksweep_*.log`; first 1,000 starts; K = 4 is this process's reference)

| Step-1 hazard of start mobs | K = 1 | K = 2 | K = 4 | K = 8 | Simulator |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Cows | 0.091 | 0.082 | 0.077 | 0.065 | 0.008 |
| Zombies | 0.068 | 0.059 | 0.049 | 0.045 | 0.020 |
| Skeletons | 0.349 | 0.164 | 0.109 | 0.102 | 0.000 |

- **Cows at steps 9-16:** hazard 0.081 / 0.056 / 0.050 / 0.040. Unpaired vanishing at step 1 is 31.6 / 28.9 / 26.8 /
  22.6 per 1,000 transitions (simulator 2.0).
- **Spawns and arrivals do not change with K.** Cow edge arrivals are 18-23 per 1,000 against the simulator's 38-51.
- **The finest trained step (k_max = 8)** cuts cow deletion about 15% against K = 4, with no extra spawns. It still
  leaves 8x the simulator.
- **So most of the deletion is in what the denoiser predicts, not in how coarsely it is sampled.** The arrival deficit
  is not the sampler's at all.

## B2: switching made the denoiser rely less on the record, map-wide (`b2.py`, `b2.json`, `b2.log`)

The smoke reproduced the P1 monitor's G1 at every level within 0.29%. Below, reliance = the error with the record's
history removed minus with the real record, per cell, at low signal (levels 0-3), on the monitor's 2,000 windows.

| Region | P1.5 reliance | R4k reliance | Change (SE) |
| :--- | ---: | ---: | ---: |
| Mob cells | 0.1326 | 0.0971 | -27% (± 1%) |
| Next to a mob | 0.0670 | 0.0392 | -41% |
| Cells a mob just left | 0.0455 | 0.0212 | -53% |
| Other map cells | 0.0837 | 0.0462 | -45% |
| Food cell | 0.1270 | 0.1222 | -4% |
| Energy cell | 0.1233 | 0.1171 | -5% |
| Other HUD | 0.1426 | 0.1237 | -13% |
| Food cell at food drops | 0.0885 | 0.0810 | -8% |
| Energy cell at energy drops | 0.1336 | 0.1265 | -5% |

- **Attenuation is now measured, and it covers the whole map,** static terrain included, not just mobs.
  - With the real record, R4k's error is unchanged: mob cells 0.0482 → 0.0489, other map 0.00206 → 0.00209.
  - So R4k moved its reliance from the record to its other inputs.
- **The HUD barely changed.** This does not explain why hunger regressed while energy held; that stays open.
- **The dose was not small.** At P1.5 a switched record raises the periphery loss by 17%: 21% of that rise falls on mob
  cells, 69% on other map cells. R4k cut it to 12%: it learned its own task, by trusting the record less.
- **Mobs are not under-weighted in the loss.** Mob cells are about 1.4% of cells but carry 20% of the periphery loss
  (per-cell error about 15x the mean). Much of that error is probably irreducible (where a cow moves is random), and
  these numbers cannot split it.
  - So the "diluted supervision" explanation for small objects (OC-STORM, ConsisDrive) does not hold here in loss-share
    terms.
  - More loss weight at mobs would mostly add pressure on irreducible motion noise.

## A6: the learned gaze moved from mobs to the HUD across the arms (`gaze_dev.py`; `gaze/*/TRACE.json`, `gaze/summary.json`)

The reading is the executed training gaze on the same labelled training subset: 1.30M occurrences, 383 episodes. It
gives crop coverage when the target is present, learned (paired uniform). The F2 rerun reproduces the review's archive
exactly. F2.1 is excluded (another corpus).

| Covered when present | F2 | F4 | HUD-Change | F6.1 | P1.5 | Uniform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Any entity | 0.487 | 0.485 | 0.461 | 0.464 | 0.418 | 0.207 |
| Cows | 0.402 | 0.401 | 0.383 | 0.344 | 0.348 | 0.146 |
| Zombies | 0.353 | 0.350 | 0.328 | 0.381 | 0.308 | 0.167 |
| Skeletons | 0.368 | 0.365 | 0.354 | 0.351 | 0.285 | 0.125 |
| Entity, day | 0.550 | 0.547 | 0.518 | 0.518 | 0.472 | 0.197 |
| Entity, night | 0.383 | 0.383 | 0.366 | 0.375 | 0.328 | 0.223 |
| Changed HUD | 0.317 | 0.304 | 0.348 | 0.373 | 0.418 | 0.114 |
| Drink counter at an attempt | 0.500 | 0.452 | 0.527 | 0.539 | 0.610 | 0.121 |
| Coal counter at an attempt | 0.304 | 0.305 | 0.314 | 0.378 | 0.418 | 0.081 |
| Entering map | 0.358 | 0.357 | 0.378 | 0.366 | 0.402 | 0.217 |

- **P1.5's gaze covers entities least of all nine worlds**, 6.9 points below F2 and 4.6 below F6.1. The entity gain
  over uniform is +21.0 [20.5, 21.7] against F2's +28.0 [27.5, 28.6]. Zombies and skeletons lose most.
- **It covers HUD changes and counters most:** HUD gain +30.4 [29.9, 30.9] against F2's +20.3.
- **Within P1.5's training,** entity coverage rose from 0.33 (first tenth) to 0.45 (last), still below F6.1's 0.49.
- **The logged surprise score at the end** is 2.0 in P1.5 against 3.7-4.3 in every other world, with contingency
  similar (9.8). The policy normalises both by their means, so only their spatial shape matters. In the F2 review,
  surprise carried most of the entity attraction, which is consistent with P1.5's gaze leaving the mobs.
- **For precision-weighting the loss by gaze,** P1.5's gaze would now put that weight more on the HUD and less on mobs
  than any earlier world.

## A5: a glimpse on the mob does not protect it at the next step (`a5.py`, `a5.json`, `a5.log`)

The design is paired: the same starts and the same noise, with the last real frame's glimpse re-encoded either on the
target mob or on the farthest map window from it.
- **Instruments:** the orig arm reproduces S2's saved step-1 frames in 99.57% of futures (cross-process drift). The
  state-0 glimpse tokens change by 0.37 on average between arms.
- **The drawn step-1 periphery barely moves:** by 0.0011 over all cells and 0.0017 around the target (periphery token
  rms about 0.017).

| Step-1 deletion of the target | On | Away | On - away (SE) | Units |
| :--- | ---: | ---: | ---: | ---: |
| Cows, day, inner | 0.050 | 0.050 | +0.000 (0.001) | 334 |
| Cows, day, ring | 0.062 | 0.063 | -0.001 (0.001) | 438 |
| Cows, night, inner | 0.121 | 0.121 | +0.000 (0.003) | 89 |
| Cows, night, ring | 0.289 | 0.290 | -0.001 (0.001) | 107 |
| Zombies, day, inner | 0.017 | 0.020 | -0.003 (0.002) | 228 |
| Zombies, night, ring | 0.304 | 0.307 | -0.003 (0.003) | 47 |
| Skeletons, night, ring | 0.533 | 0.517 | +0.017 (0.009) | 16 |

- **The glimpse has no causal effect on next-step deletion,** night edge included (-0.1 ± 0.1 points). A4's -9 points
  for night-edge cows was noise.
- **The previous frame's glimpse barely reaches the next frame's periphery.** This fits 10-29 ("the belief keeps nothing
  of glimpses") and F1.6 (half retained, the global projection loses access).
- **For a gaze-precision-weighted loss,** the route it would act through is unused today. Such a loss would first have
  to create that route (as the brain's transsaccadic learning does), and the deletion it would target happens with full
  information in daylight.

# The eye's checks (user 11-20)

## C1: why P1.5's chooser predicts better (`c1.py`, `c1.json`; 150 DEV episodes; each world's own encoder and final head)

| Per cell | F6 | F6.1 | P1.5 |
| :--- | ---: | ---: | ---: |
| Head next-token error (whole frame) | 0.031 | 0.028 | 0.015 |
| Mob cells: token change | 11.6 | 11.0 | 5.7 |
| Mob cells: chooser surprise | 7.8 | 7.4 | 3.8 |
| Static map: token change | 1.87 | 1.68 | 1.53 |
| Static map: share explained by the head | 0.81 | 0.82 | 0.92 |
| HUD flip: token change | 14.5 | 14.6 | 13.1 |
| HUD flip: chooser surprise | 8.4 | 7.5 | 7.6 |
| Entering terrain: chooser surprise | 2.2 | 2.0 | 1.9 |

- **Mobs barely change P1.5's tokens.** At mob cells the tokens change half as much as in F6 / F6.1, and the head
  explains the same share (0.32-0.33). P1.5's encoder represents mobs with about half the contrast.
- **Static terrain moves more predictably** (0.92 explained against 0.81-0.82) at a similar change.
- **HUD flips and entering terrain are unchanged.** Their share of the chooser's surprise roughly doubles: 11.3% and
  27.9%, against 6.1-6.2% and 15.8-15.9%.
- **Coverage on these frames, P1.5 against F6.1:**
  - entities 0.447 against 0.486;
  - zombies 0.315 against 0.370 (night 0.251 against 0.318);
  - cows 0.384 against 0.379 (unchanged);
  - changed HUD 0.423 against 0.380 (night 0.431 against 0.337).
- **Cause, by P1.5's design:** two of its five changes decide what trains the encoder. Route by role means the reward,
  forecast and gaze heads no longer train it; the harmoniser's fixed point reweights G1 / G2 / G3. C1 cannot separate
  the two. Neither the chooser's inputs nor its recipe changed.

## C2's cost (`c2_cost.py`, `c2_cost.json`; measured)

| Item | Measured |
| :--- | ---: |
| Fresh world-eye, 64-step update (trainer 2.284 s) | 1.79 s (78%) |
| Fresh world-eye, 256-step update (trainer 8.581 s) | 6.94 s (81%) |
| Projected extra on the 15.4 h run | +12.2 h |
| Offline fill at 64 streams, per frame | 0.88 ms (in-loop 1.69-1.75 ms) |
| Cache read per update | 0.75 ms (64) / 2.47 ms (256) |
| Cache refreshed every 1 / 2 / 5 / 10 passes | +6.3 / +3.1 / +1.3 / +0.6 h |

- **The in-loop cost comes from the pass being sequential.** Each target frame needs one zero-signal evaluation plus
  its context pass, because in a single-stream diffusion-forcing pass a frame's noisy tokens are also later frames'
  context.
- **In imagination the same expectation is free:** it is the sampler's first evaluation.
- **The page-cache read was measured on a file that fits in RAM.** A full pass of periphery estimates is about 6.6 GB.

## C2's location part: the world-eye does not look at mobs more than the chooser (`c2_eye.py`, `c2_eye.json`)

The data are 3,000 DEV target frames from 150 episodes. Both eyes see the same frames: P1.5's zero-signal error under
the chooser's policy with contingency off, against P1.5's chooser.

| Covered when present | World-eye | Chooser | Uniform |
| :--- | ---: | ---: | ---: |
| Any entity | 0.484 | 0.484 | 0.204 |
| Any entity, night | 0.359 | 0.369 | 0.234 |
| Cows | 0.448 | 0.418 | 0.149 |
| Zombies | 0.296 | 0.352 | 0.166 |
| Changed HUD | 0.224 | 0.409 | 0.109 |
| Entering terrain | 0.511 | 0.457 | 0.216 |

- **Correction.** Earlier I compared B2's monitor-window read (mobs 57%) with the chooser's training-trace coverage
  (42%). Those were different frames. On the same frames the two eyes cover entities equally.
- **The world-eye shifts from HUD flips to newly revealed terrain,** and from zombies to cows.
- **The world's surprise by region:** mobs 30%, entering terrain 31%, static map 32%, HUD 6.5%. At night the static
  map is 60%: the night static, irreducible. The noisy-TV risk is real.
- **Its surprise does not mark the cows the world deletes.** Over 2,079 S2 start mobs, the zero-signal surprise around
  the mob in the recorded next frame against P1.5's step-1 deletion rate:
  - by surprise quartile, deletion is 0.13 / 0.20 / 0.17 / 0.09 (correlation -0.12);
  - deleted mobs show surprise 4.13, never-deleted mobs 4.40.

  So the world's mean prediction keeps the cow, and the deletion happens in sampling from noise to a sample. This
  refines the stage-B reading ("in the denoiser's prediction"): the conditional mean is right, and the flow to a
  sample drops the cow.
- **Staleness (cache quality)** against the fresh eye, whose distance to the chooser is a TV of 0.237:

  | Cache age | TV to fresh | Argmax agreement | Entity coverage (fresh 0.484) |
  | :--- | ---: | ---: | ---: |
  | 1 pass stale | 0.038 | 0.76 | 0.472 |
  | 2 passes stale | 0.065 | 0.65 | 0.452 |
  | 5 passes stale | 0.250 | 0.43 | 0.302 |

  A cache refreshed every 1-2 passes is close to fresh; 5 passes is not. All of these are late-training checkpoints.

## C3: the cheap memory read, inconclusive at this budget (`c3.py`, `c3.json`, `c3.log`)

The setup: P1.5's final chooser head, fine-tuned on 320 DEV episodes (120 frames each) and read on 80 held-out
episodes. Both arms continue the chooser's own AdamW at its final rate 1e-4. The record arm adds a zero-initialised
per-cell read of the record (Linear 1,024 -> 64) in its own group. 1,500 updates, batch 256.

| Held-out | Plain | + record read |
| :--- | ---: | ---: |
| Training loss, last log | 0.01402 | 0.01385 |
| HUD-flip surprise per cell | 7.95 | 7.97 |
| Mob surprise per cell | 4.41 | 4.40 |
| Entity coverage | 0.476 | 0.476 |
| HUD-flip coverage | 0.432 | 0.430 |

- **The record arm is indistinguishable from the plain one;** its loss is 1.2% lower.
- **This is not evidence that the record cannot help the eye.** A zero-initialised read at the head's final 1e-4 for
  1,500 updates has little room to form. The information is there: P1.5's denoiser times the clocks from this record,
  and on 11-16 elapsed time was linearly readable from rotating bases (0.90).
- **A decisive version:**
  - First, a CPU readability probe: does a linear read of the HUD cells' record predict the next flip?
  - Then, if it does, the new read's own group at the chooser's peak 1e-3 (fresh parameters, so no re-warm of trained
    ones), with the trained head kept at 1e-4, for about 6,000 updates.

## C3, decisive version: the record anticipates flips, but a memory-equipped chooser still looks the same

- **Probe (`c3_probe.py`, `c3_probe.json`).** Held-out AP for "this digit flips at t", from the state through t - 1:

  | Feature set | Food | Drink | Energy |
  | :--- | ---: | ---: | ---: |
  | Record | 0.275 | 0.260 | 0.455 |
  | Chooser's last 8 frames | 0.041 | 0.072 | 0.048 |
  | Base rate | 0.035 | 0.048 | 0.034 |

  The record carries the anticipation linearly; the chooser's own view does not.
- **Rerun (`c3.py 4000 1e-3`, `c3_lr0.001_4000.json`).** The new read at the chooser's peak 1e-3 (warm-up 100), the
  trained head at its final 1e-4, 4,000 updates.

  | Held-out | Plain | + record read |
  | :--- | ---: | ---: |
  | Training loss, last log | 0.01395 | 0.01357 (-2.7%) |
  | HUD-flip surprise per cell | 7.75 | 7.60 (-2%) |
  | Static map surprise per cell | 0.099 | 0.106 |
  | HUD-flip coverage | 0.435 | 0.432 |
  | Entity coverage | 0.476 | 0.476 |

- **Reading.** The eye's surprise is the squared error of its predicted next token. A flip anticipated at an AP of
  0.26-0.45 stays mostly unpredictable at one step: the squared-error-optimal prediction is the expected change, so the
  surprise at the flip barely falls. A cheap memory read therefore does not take the eye off the HUD flips.
- **The world-eye does** (HUD-flip coverage 0.224). The world's own prediction times the flips sharply, from the same
  record through the denoiser's read, trained by the whole of P1.

## C4: the sample-eye fails both criteria (`c4_sample.py`, `c4_sample.json`)

The data are C2's 3,000 DEV frames and 2,079 S2 start mobs. Surprise is measured against the world's own 4-step sample,
two draws per frame.

| Covered when present | Sample-eye | World-eye (C2) | Chooser | Uniform |
| :--- | ---: | ---: | ---: | ---: |
| Any entity | 0.435 | 0.484 | 0.484 | 0.204 |
| Cows | 0.402 | 0.448 | 0.418 | 0.149 |
| Zombies | 0.266 | 0.296 | 0.352 | 0.166 |
| Changed HUD | 0.206 | 0.224 | 0.409 | 0.109 |
| Entering terrain | 0.511 | 0.511 | 0.457 | 0.216 |

- **Where its surprise lies:** static map 36% (night 60%), entering terrain 34%, mobs 24%. The two draws differ from
  each other by 78% of their mean error.
- **Deletion link:** when a draw deleted the mob, the eye covers it 0.483; when the draw kept it, 0.489 (596 / 3,562
  draws). S2's deletion rate by surprise quartile is 0.07 / 0.34 / 0.10 / 0.08 (correlation -0.15).
- **Cost (measured):** 4.08 s per 64-step update (179%) and 16.0 s per 256-step update (187%), so +28.1 h on the
  15.4 h run if fresh.
- **Verdict by the declared rule:** fails. Mob coverage is not higher (it is lower), and deleted mobs are not covered
  more than kept ones.
- **Why.** A sample differs from reality wherever the world is stochastic: textures, night static, terrain entering
  the view. A single cell where the draw dropped the cow is one high-error cell among many. The two draws' disagreement
  (0.78) shows how large that sampling noise is.
- **What it implies for the deletion.** Reality shows the cow, and so does the world's mean prediction. Only a fraction
  of its samples drop it. The deletion is a property of the world's sampling distribution (its calibration at mob
  cells), and no single-frame comparison between reality and the world singles it out. An eye cannot find it in
  training; a score over many of the world's samples can.

## C5: the deletion is the denoiser's sampling with the information present, decided at its first step; Mamba's belief barely holds mob positions (`c5_calibration.py`, `c5_calibration.json`, `c5_units.npz`)

The data are C4's 2,079 S2 start mobs, with 8 draws each of the default sampler (16,632 draws). The draws share the
context, record and belief; only the target's noise differs.

**Calibration.**

| | Real frame keeps (reader) | World keeps (C5) | S2 step-1 deletion |
| :--- | ---: | ---: | ---: |
| Cows, day | 1.000 | 0.940 | 0.062 |
| Cows, night | 0.977 | 0.767 | 0.227 |
| Zombies, day | 1.000 | 0.983 | 0.023 |
| Zombies, night | 0.983 | 0.883 | 0.147 |
| Skeletons, day | 1.000 | 0.853 | 0.153 |
| Skeletons, night | 0.895 | 0.530 | 0.447 |

- The window form reproduces S2's streaming step-1 deletion (cows 6.0% vs 6.2% by day, 23.3% vs 22.7% at night).
- Arrows are left out: under the 3 x 3 rule the real frame keeps only 0.795 of them (they fly faster), so the
  instrument does not hold.

**Concentration.**
- **By day, close to a binomial.**
  - Cows: dispersion 1.38. 64.9% are never deleted (binomial 61.0%), units deleted in at least 4 of 8 draws hold 6.7%
    of the deletions, and none is deleted in all 8.
  - Zombies: dispersion 1.57. Skeletons: 1.50.
  - Neither the ring (cows: inner 5.2%, edge 6.6%) nor the camera (still 6.1%, moved 5.1%) changes it much.
  - Every visible mob by day carries about the same coin per draw.
- **At night, concentrated (dispersion 4.2-4.8) and at the edge ring** (post hoc split; edge was predicted).
  - Cows: 40.8% are never deleted (binomial 11.9%), the 19.1% deleted in at least 4 of 8 draws hold 65.6% of the
    deletions, and 16 are deleted in all 8.
  - Edge ring against inner cells: cows 29.7% vs 14.9%, zombies 32.4% vs 4.8%.
  - 27 of the 29 mobs deleted in all 8 draws sit on the edge ring, where the night static is strongest (Stage B).

**Where on the ladder:** the reader's P(kind), max over the neighbourhood, on each step's clean estimate.

| Step | 1 | 2 | 3 | 4 (= the sample) |
| :--- | ---: | ---: | ---: | ---: |
| Deleted draws | 0.31 | 0.20 | 0.08 | 0.01 |
| Kept draws | 0.86 | 0.90 | 0.98 | 0.998 |
| AUC for the final outcome | 0.84 | 0.92 | 0.99 | 1.00 |

- The first clean estimate already separates the draws, so the target's noise decides at the first step and the later
  steps commit to it.
- The same holds by day (AUC 0.77 at step 1) and at night (0.88).

**The deleted cell's form:** a well-formed absence, not a blend.
- In 94% of deleted draws P(kind) is below 0.05 over the whole neighbourhood.
- The mob's own cell reads "no entity" in 95% of them.

**Does Mamba know where the mobs are?** Linear probe per map cell, held-out AP; 611 / 153 unit-free DEV episodes.

| Target (base rate) | Belief | Previous frame, fitted | Copy, unfitted | Both |
| :--- | ---: | ---: | ---: | ---: |
| Any entity (0.017) | 0.048 | 0.255 | 0.195 | 0.249 |
| Cows (0.0075) | 0.012 | 0.230 | 0.170 | 0.212 |

- The belief holds little of where mobs are, and adds nothing to the previous frame.
- **On the units, neither probe marks the deleted mobs.**
  - By the belief probe's quartile, deletion is 0.095 / 0.158 / 0.235 / 0.076 (correlation 0.01).
  - By the previous-frame probe's quartile, it is 0.154 / 0.102 / 0.087 / 0.221 (-0.08).
- A linear probe is a lower bound on what the denoiser's attention could extract from the memory tokens.

**Predictions against outcome.**
- Over-dispersed, mostly night and edge: **right at night** (edge), **wrong by day** (near binomial).
- Every step's estimate keeps the mob at first, and the draws part at steps 2-3: **wrong**. They part at the first step.
- The belief probe well below the copy and not marking deleted units: **right**.

**What C5 establishes.**
- The deletion happens in the denoiser's sampling, with the information present.
  - The real past shows the mob, and most draws keep it.
  - The target's noise decides at the first step which draws lose it, and they end in clean terrain.
- By day it is one small, uniform coin per visible mob (cows 6% per step against a real-frame floor of 0-0.5%). The
  conditional is under-confident about persistence everywhere, not in particular contexts.
- At night, concentrated losses at the edge ring come on top: some mobs there are lost in nearly every draw.
- Mamba's predicted belief barely holds where mobs are. That does not decide which mobs are lost, but nothing in the
  belief could restore a mob once a generated frame has dropped it.

**What it does not establish:** why the learned conditional gives a visible mob a 6% absence by day.
- The data do not teach it: the reader on the real next frame keeps 100% by day.
- K = 8, the finest trained step, removes only about 15% (Stage B), so the cause lies more in the learned conditional
  than in the coarse steps.

## C6: learned fast, then flat while the learning rate decays (`c5_calibration_ckpt{02,05,08}.json`)

C5's draws and readings at P1.5's checkpoints, each read with its own S1 reader (real-frame floor 0.95-1.00 at every
checkpoint).

| World keeps | Ckpt 2 (5.1M) | Ckpt 5 (12.8M) | Ckpt 8 (20.5M) | Ckpt 10 (25.6M) |
| :--- | ---: | ---: | ---: | ---: |
| Cows, day | 0.252 | 0.881 | 0.938 | 0.940 |
| Cows, night | 0.013 | 0.703 | 0.763 | 0.767 |
| Zombies, day | 0.262 | 0.951 | 0.981 | 0.983 |
| Zombies, night | 0.018 | 0.840 | 0.873 | 0.883 |
| Skeletons, day | 0.032 | 0.751 | 0.825 | 0.853 |
| World learning rate | 9.6e-5 | 5.8e-5 | 2.4e-5 | 1.0e-5 |

- Day cows' dispersion is 1.41 / 1.33 / 1.38 from checkpoint 5 on; at checkpoint 2 it is 3.71.
- The step-1 AUC (day) is 0.77 / 0.80 / 0.77 from checkpoint 5 on.
- **By the declared rule this is mixed.**
  - The coin still falls from 5 to 8 (cows +5.7 points), so it is not flat from 5 on. My prediction was wrong.
  - It is flat from 8 to 10 (cows +0.2, within noise), while skeletons still rise (+2.8).
- **The plateau coincides with the cosine schedule's tail.** The rate falls from 2.4e-5 to 1.0e-5 between 8 and 10, so
  "converged" and "too small a rate to keep learning" cannot be told apart here.
- **What it does establish:** the persistence coin is learned (from 75% deletion at checkpoint 2 to 6%), with
  diminishing returns. It is not a fixed property of the sampler, and the form (near-binomial by day, decided at the
  first step) is stable from checkpoint 5 on.

## C7: the belief knows a zombie is next to the player, not which cell it is in (`c7_zombies.py`, `c7_zombies.json`)

The data are all DEV episodes, 807 to fit and 202 held out (22,113 held-out frames). The probe is linear, per
target.

| Held-out AP | Adjacent | Within 2 | On screen | Per cell, within 2 |
| :--- | ---: | ---: | ---: | ---: |
| Base rate | 0.130 | 0.197 | 0.286 | 0.0093 |
| Corrected belief b_t (the actor's input) | **0.921** | 0.832 | 0.788 | 0.116 |
| Predicted belief bp_t (the denoiser's memory) | 0.832 | 0.792 | 0.769 | 0.071 |
| Reader on frame t, fitted / unfitted | 1.000 / 1.000 | 1.000 / 1.000 | 0.997 / 0.999 | 0.989 / 0.999 |
| Reader on frame t - 1, fitted / unfitted | 0.885 / 0.802 | 0.945 / 0.887 | 0.961 / 0.932 | 0.343 / 0.329 |

- **Presence, yes.** The actor's input knows whether a zombie is adjacent (0.921 against the frame's 1.000) and near
  (0.832).
- **Timing, yes.** P1.5's evaluation already showed the cooldown: given an adjacent zombie, bp anticipates a hit at
  AP 0.952.
- **Position, no.** Which cell it is in is weak: 0.116 against the current frame's 0.99. That covers the direction to
  face to attack or flee.
- **For the actor (P3):** b_t says "a zombie is next to me, and it can hit now", but not where. An actor reading b_t
  alone would need the current percept for direction. Not an issue for P1.6.

## C8: belief change fails; it moves the eye onto entering terrain, not mobs (`c8_belief_change.py`, `c8_belief_change.json`)

The data are C2's frames with t + 1 <= T: 2,983 windows. The Mamba re-run reproduces C2's belief within 0.041 (bf16),
and both arms share that path.

| Covered when present | Belief change | World-eye surprise | Chooser | Uniform |
| :--- | ---: | ---: | ---: | ---: |
| Any entity | 0.423 | 0.484 | 0.485 | 0.203 |
| Any entity, day | 0.436 | 0.530 | 0.527 | 0.193 |
| Any entity, night | 0.385 | 0.357 | 0.369 | 0.233 |
| Zombies | 0.263 | 0.297 | 0.353 | 0.165 |
| Changed HUD | 0.242 | 0.227 | 0.414 | 0.110 |
| Entering terrain | 0.561 | 0.511 | 0.457 | 0.216 |

| Share of the score | Mobs | Entering terrain | Static map | Changed HUD |
| :--- | ---: | ---: | ---: | ---: |
| Belief change, all / night | 0.10 / 0.11 | 0.43 / 0.26 | 0.39 / 0.49 | 0.05 / 0.08 |
| World-eye surprise, all / night | 0.31 / 0.17 | 0.33 / 0.17 | 0.30 / 0.59 | 0.04 / 0.05 |

**By the declared rule, belief change fails.**
- Entity coverage falls (0.423 against 0.484). Only night rises a little (0.385 against 0.357).
- Night static falls from 59% to 49% of the score, but stays the largest share.
- Changed HUD stays near the world-eye's (0.242 against 0.227).

**Why.**
- **A mob's next position is uncertain, so its information washes out.** The world's zero-signal prediction of frame
  t + 1 hedges where a mob will be. Knowing exactly where it is at t moves that blurred prediction little: mobs hold
  10% of the score.
- **Entering terrain persists and is new,** so it moves the prediction most (43%).
- **Frame t's static still moves the prediction of t + 1** (49% at night), although the renderer redraws it every
  frame. The world carries part of its context's noise forward.

**Predictions:** right that night static falls and that HUD flips stay near the world-eye's. Wrong about "well below
60%" and about entities.
