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
