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

  This surprise reading suggested that the world's mean prediction keeps the cow. C13 (below) corrects it: read
  directly, the noise-free estimate omits 5.4% of day cows fully and 8.7% partly, and 22.9% of night cows. By day,
  most deletions (85%) still arise along the sampler's path from an estimate that shows the cow.
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

## C9: a score cache works and an estimate cache does not; the cheaper route halves the cost but no more (`c9_eye_cache.py`, `c9_early.py`, their `.json`, `c9_fixations.png`)

The data are C2's 3,000 DEV frames, and all eyes see the same frames. "Deployed" is the form a world-eye would take in
training: the chooser's own contingency C with the world's S. Coverage when present:

| Deployed | Chooser | Fresh world-eye | Score cache, 2 stale | Score cache, 5 stale | Estimate cache, 5 stale | Uniform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Any entity | 0.484 | 0.514 | 0.514 | 0.490 | 0.357 | 0.204 |
| Cows | 0.418 | 0.462 | 0.462 | 0.433 | 0.283 | 0.149 |
| Zombies | 0.352 | 0.347 | 0.348 | 0.345 | 0.293 | 0.166 |
| Changed HUD | 0.409 | 0.299 | 0.307 | 0.338 | 0.282 | 0.109 |
| Entering terrain | 0.457 | 0.485 | 0.485 | 0.475 | 0.318 | 0.216 |
| TV to fresh | 0.130 | 0 | 0.031 | 0.086 | 0.232 | |
| Argmax agreement with fresh | 0.67 | 1 | 0.87 | 0.76 | 0.58 | |

- **The deployed world-eye beats the chooser modestly:**
  - entities +3.0 points and cows +4.4;
  - zombies tie (-0.5);
  - changed-HUD looks fall by 11 points.
  - With contingency off (C2's form) it had tied on entities and lost 5.6 points on zombies. Contingency restores the
    zombies.
- **The estimate cache chases encoder drift.**
  - Its 5-stale choices are nearer the drift-only eye than the fresh eye (TV 0.133 against 0.232).
  - Its looks are halfway to uniform: mob share 0.031 of looked cells, against 0.041 for the chooser and 0.017 for
    uniform.
  - Its look mass on windows the current world already predicts well is 0.368, against fresh 0.254, the chooser 0.270
    and drift only 0.449.
  - So it isn't confirming what the world knows; it is looking where the encoder's tokens moved.
- **The score cache** stores each refresh's error in that checkpoint's own tokens, which removes the drift. 1, 2 and 5
  passes stale give TV 0.028, 0.031 and 0.086. At 5 passes stale it still matches or beats the chooser on everything
  except zombies (-0.7).
- **Early training drifts faster.** One pass stale, deployed with each checkpoint's own chooser head:

  | At checkpoint | TV to fresh | Entities: stale / fresh / chooser | Zombies: stale / fresh / chooser | Changed HUD: stale / chooser |
  | :--- | ---: | :--- | :--- | :--- |
  | 2 | 0.172 | 0.383 / 0.452 / 0.454 | 0.296 / 0.342 / 0.343 | 0.264 / 0.367 |
  | 3 | 0.117 | 0.451 / 0.473 / 0.459 | 0.345 / 0.348 / 0.342 | 0.319 / 0.402 |
  | 4 | 0.099 | 0.470 / 0.468 / 0.466 | 0.354 / 0.336 / 0.351 | 0.368 / 0.416 |
  | 5 | 0.088 | 0.468 / 0.488 / 0.473 | 0.333 / 0.340 / 0.349 | 0.327 / 0.422 |

  Three passes stale early (checkpoint 2 read at 5, contingency off): TV 0.153, entities 0.411 against fresh 0.458.
  A cache refreshed before checkpoint 3 is worse than the chooser.
- **The chooser-proxy belief barely matters.** Building Mamba's belief from the chooser's sampled looks instead of the
  monitor's prior moves the eye by TV 0.006-0.007 (argmax agreement 0.95-0.97). In P1.5 the eye's own past looks
  hardly change its estimate, so the sequential dependence can be cut. G1 changes this by design, and C9 cannot
  measure it.
- **The cheaper route costs +6.6 h,** about 41-45% of each update. Two chunked denoiser forwards are 80-87% of that;
  the perception, record and belief pass is the rest. It halves the fresh in-loop eye (+12.2 h) and costs about
  the same as a cache refreshed every pass (+6.3 h). The denoiser's evaluations are the cost, whichever way they are
  arranged.
- **The figure** (`c9_fixations.png`): six frames, each eye's per-cell look probability and its most likely window.
  The 5-stale estimate cache spreads its looks; the score cache follows the fresh eye.

**Predictions:**
- right that the score cache is far closer to fresh than the estimate cache (0.086 against 0.232), and that the
  estimate cache chases drift;
- right that the proxy belief is within the 1-pass staleness (0.007 against 0.038);
- wrong that the cheaper route costs under a third of the fresh eye: it costs 54%.

## C10: history guidance does not cleanly fix the coin (`c10_history_guidance.py`, `c10_history_guidance.json`)

The data are C5's 2,079 start mobs, with 8 draws per arm and C5's noise seeds. The unguided arm reproduces C5's
outcomes exactly (match 1.000).

| World keeps (real frame) | Unguided | Frames masked, w 1.5 | Frames masked, w 3 | All history, w 1.5 |
| :--- | ---: | ---: | ---: | ---: |
| Cows, day (1.000) | 0.940 | 0.942 | 0.941 | **0.976** |
| Cows, night (0.977) | 0.767 | 0.797 | **0.848** | 0.819 |
| Zombies, day (1.000) | 0.983 | 0.983 | 0.982 | 0.996 |
| Zombies, night (0.983) | 0.883 | 0.898 | 0.916 | 0.918 |

| Side effects | Unguided | Frames, 1.5 | Frames, 3 | All, 1.5 |
| :--- | ---: | ---: | ---: | ---: |
| Cows staying put, share of kept draws (real 0.603) | 0.686 | 0.695 | 0.712 | 0.762 |
| Zombies staying put (real 0.599) | 0.671 | 0.664 | 0.656 | 0.694 |
| Spurious mobs per frame | 0.841 | 0.852 | 0.886 | 0.872 |
| Token error: static HUD | 0.039 | 0.040 | 0.043 | **0.388** |
| Token error: changed HUD | 5.42 | 5.46 | 5.61 | 6.96 |
| Token error: static map | 0.246 | 0.249 | 0.258 | 0.294 |
| Token error: mob cells | 5.49 | 5.47 | 5.45 | 5.00 |
| Token error: entering terrain | 3.73 | 3.73 | 3.78 | 3.51 |
| Spread over draws: mob cells | 1.56 | 1.56 | 1.56 | 1.35 |
| Spread over draws: entering terrain | 1.63 | 1.63 | 1.66 | 1.29 |

- **Masking only the frames (DFoT's form) leaves the day coin untouched** (0.940 -> 0.942 -> 0.941), even at w = 3.
  It helps at night: night cows go from 0.767 to 0.848 at w = 3, with little damage elsewhere.
- **Masking the whole history cuts the day deletion by 60%** (cows 6.0% -> 2.4%), but it damages the rest:
  - the static HUD error rises tenfold, the changed HUD by 29% and the static map by 20%;
  - mobs stay put more often (cows 0.762, against 0.686 unguided and 0.603 real);
  - diversity falls (mob spread -14%, entering terrain -21%).
- **Reading.** Guidance acts only where the conditional and unconditional predictions differ.
  - With only the frames masked, the record and belief still carry the day cow, so both predictions keep it and
    nothing changes. At night the frames add more.
  - With everything masked, the unconditional is an episode's first frame. Wherever the history is decisive (the HUD
    digits, the static map), w > 1 extrapolates past the conditional prediction. This is classifier-free guidance's
    known overshoot, and it is also DFoT's static-video failure in mild form.
  - That the record rather than the belief carries the day cow is likely (C5: the belief barely holds positions), but
    not measured.
- **The world already under-moves its mobs** (0.686 against 0.603 real) before any guidance.
- **Predictions:**
  - wrong that frames at 1.5 halves the day coin (no effect);
  - right that "all" does more at a cost in fidelity;
  - only weakly right that w = 3 raises staying put (cows +2.6 points, zombies -1.5).
- **Night, by C5's edge ring** (`night` rerun, 580 night units, `c10_history_guidance_night.json`; the full run's ring
  definition held no units, my error). World keeps, with the real frame in brackets:

  | | Unguided | Frames, 1.5 | Frames, 3 | All, 1.5 |
  | :--- | ---: | ---: | ---: | ---: |
  | Cows, edge (0.966) | 0.704 | 0.724 | 0.777 | 0.763 |
  | Cows, inner (0.991) | 0.851 | 0.889 | 0.937 | 0.894 |
  | Zombies, edge (0.966) | 0.676 | 0.699 | 0.725 | 0.737 |
  | Zombies, inner (0.989) | 0.951 | 0.965 | 0.978 | 0.979 |

  - Frame guidance at w = 3 cuts night deletions inside the ring by more than half (cows 14.9% -> 6.3%), and at the
    edge only by a quarter (29.6% -> 22.3%). The night static at the edge stays the main loss.
  - The rerun matches the full run within one draw in 1,880 for zombies (batch composition changes bf16 kernels).

## C11: a cache is as good as the world that filled it; fill late, not often (`c11_schedule.py`, `c11_schedule.json`)

Entity coverage on C2's 3,000 DEV frames, deployed (checkpoint j's chooser C). Rows: the cache filled at checkpoint
i. Columns: read at checkpoint j.

| | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cache from 1 | 0.383 | 0.383 | 0.381 | 0.382 | 0.384 | 0.386 | 0.381 | 0.380 | 0.384 |
| Cache from 2 | | 0.451 | 0.448 | 0.448 | 0.450 | 0.452 | 0.448 | 0.447 | 0.450 |
| Cache from 3 | | | 0.470 | 0.469 | 0.472 | 0.474 | 0.471 | 0.469 | 0.472 |
| Cache from 5 | | | | | 0.490 | 0.492 | 0.489 | 0.488 | 0.490 |
| Cache from 6 | | | | | | 0.514 | 0.511 | 0.510 | 0.512 |
| Fresh world-eye at j | 0.452 | 0.473 | 0.468 | 0.488 | 0.512 | 0.514 | 0.512 | 0.509 | 0.514 |
| Chooser at j | 0.454 | 0.459 | 0.466 | 0.473 | 0.483 | 0.480 | 0.479 | 0.481 | 0.484 |

(Fresh at checkpoint 1: 0.361; chooser at 1: 0.402.)

- **A cache's coverage is set by the checkpoint that filled it, not by its age.** Each row is flat to within 0.006.
  C9's "staleness" was the gap between the filler's world and today's, not decay.
- **Early worlds make poor eyes.**
  - At checkpoint 1 even the fresh world-eye is worse than the chooser (0.361 against 0.402); at 2 they tie.
  - A cache filled at checkpoint 1 stays at 0.383, below the chooser at every later checkpoint.
- **The world-eye jumps at checkpoint 6** (0.488 -> 0.512), the first pass with 256-step chunks, and then plateaus.
  After that, refreshing adds nothing measurable.
- **Zombies:** fresh and chooser are within 0.015 of each other at every checkpoint. **Changed HUD:** the world-eye
  looks 7-13 points less at every checkpoint.
- **Schedules,** each pass read at its end (mean over ten passes):

  | Schedule | Entities | Zombies | Cows | Changed HUD | Fills |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | Chooser throughout | 0.466 | 0.341 | 0.403 | 0.407 | 0 |
  | Refresh after 1 and 5 (early and stale) | 0.438 | 0.320 | 0.385 | 0.343 | 2 |
  | Refresh after 2 and 5 | 0.465 | 0.338 | 0.406 | 0.358 | 2 |
  | Refresh after 6 | 0.478 | 0.342 | 0.420 | 0.366 | 1 |
  | **Refresh after 5 and 6 (chosen by the rule)** | **0.479** | **0.341** | **0.422** | **0.359** | 2 |
  | Fresh throughout | 0.481 | 0.337 | 0.426 | 0.305 | in-loop (+12.2 h) |

  The chosen schedule recovers 90% of the fresh eye's entity gain over the chooser at two fills. An early, stale
  world-eye is worse than the chooser.

## C11b: the chooser beats uniform from its first updates (`c11_trace.py`, `c11_trace.json`)

P1.5's own training looks (the 400 labelled training episodes) against the paired uniform window:

| Updates | Entities: chooser / uniform | Zombies | Cows | Changed HUD | Entering terrain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0-250 | 0.296 / 0.218 | 0.213 / 0.164 | 0.228 / 0.148 | 0.284 / 0.117 | 0.226 / 0.215 |
| 1,000-1,250 | 0.317 / 0.194 | 0.231 / 0.166 | 0.288 / 0.138 | 0.322 / 0.108 | 0.227 / 0.226 |
| 2,250-2,500 | 0.335 / 0.203 | 0.232 / 0.162 | 0.300 / 0.136 | 0.430 / 0.114 | 0.329 / 0.210 |
| 10,000-12,500 | 0.431 / 0.209 | 0.329 / 0.170 | 0.351 / 0.146 | 0.447 / 0.119 | 0.424 / 0.219 |

The chooser covers entities, zombies and cows above uniform from the first 250 updates. Uniform would only remove its
HUD over-looking. This is coverage, not what the world learns from it.

## C12: the day coin is not a hedge over where the mob goes next (`c12_hedging.py`, `c12_hedging.json`)

C5's 2,079 units and 8 draws, by the mob's free 4-neighbours at t0 under Craftax-Classic's move rule. A cow moves to a
random choice of 4 moves and 4 stays, and a blocked move becomes a stay; so a cow with one free neighbour stays put at
least 87.5% of the time, and one with four free neighbours 50%.

| Day cows | 1 free | 2 free | 3 free | 4 free |
| :--- | ---: | ---: | ---: | ---: |
| Units | 43 | 117 | 310 | 459 |
| Deletion | 0.044 +- 0.013 | 0.065 +- 0.011 | 0.052 +- 0.005 | 0.066 +- 0.005 |

- **Ratio of 4 free to at most 1 free: 1.51, bootstrap 95% interval 0.90-3.23;** correlation over units 0.04. By the
  declared reading the interval covers 1, so the hedge is not supported. A hedge strong enough to be the cause would
  make nearly boxed-in cows far safer than free ones. A small contribution cannot be excluded; no day cow was fully
  boxed in.
- **By day zombies show no trend either** (0.9-3.7%). At night cows rise from 0.146 (1 free) to 0.271 (4 free), but
  night is confounded with the edge ring and static (C5).
- **The coin is a near-uniform under-confidence about persistence**, as C5's dispersion (1.38) suggested.
- **Ruled out so far:**
  - missing evidence (C5);
  - the coarse sampler (K = 8, the finest trained step, removes 15%);
  - the shortcut's bootstrap (K = 8 uses only the flow term);
  - a hedge over the next cell (C12);
  - sampling-time history guidance as a clean fix (C10).
- **Not yet tested:**
  - whether more training at a useful learning rate keeps lowering it (C6 flat while the rate decays to 1e-5);
  - the objective's emphasis at the low-signal levels where presence is decided (ramp w = 0.9 tau + 0.1: weight 0.1
    at tau = 0 against 0.89 at 7/8);
  - a mob being 1 cell of 81 in the per-frame mean.

## C13: by day, early under-confidence carried forward and committed by the sampler; at night, a context that doesn't establish the mob (`c13_levels.py`, `c13_levels.json`, `c13_units.npz`)

C5's 2,079 units, P1.5 checkpoint 10, S2's readers. (a) is the real frame noised to each finest level, the flow
term's training form; (b) is the K = 8 sampler's own path. Both start from pure noise at level 0.

**Day cows (929 units; real-frame floor 1.000):**

| Signal level (/8) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | Final |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) real path keeps | 0.904 | 0.903 | 0.903 | 0.936 | 0.974 | 0.994 | 0.999 | 1.000 | |
| (b) sampler keeps | 0.904 | 0.898 | 0.900 | 0.921 | 0.942 | 0.948 | 0.949 | 0.949 | 0.949 |

- **The noise-free estimate is nearly independent of its noise** (spread over draws 0.014 against 2.1 from the real
  token, day cows that stayed). By day it omits the cow on every draw for 5.4% of units, keeps it on every draw for
  85.9%, and varies for 8.7%.
- **The two components:**

  | | Estimate omits the mob | Their final deletion | Estimate keeps it | Their final deletion | Deletions from the omitted |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | Cows, day | 5.4% | 0.142 | 85.9% | 0.042 | 15% |
  | Cows, night | 22.9% | 0.660 | 61.5% | 0.066 | 70% |
  | Zombies, day | 2.4% | 0.014 | 93.3% | 0.011 | 3% |
  | Zombies, night | 14.5% | 0.467 | 73.6% | 0.032 | 63% |

- **At night, most deletions are mobs the noise-free estimate already omits:** the context does not establish them
  (static, darkness). This is what G1's fine evidence targets.
- **By day, the coin lives where the estimate shows the cow** (798 units):

  | Signal level (/8) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | Final |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | (a) real path keeps | 1.000 | 0.981 | 0.968 | 0.973 | 0.990 | 0.999 | 1.000 | 1.000 | |
  | (b) sampler keeps | 0.998 | 0.981 | 0.970 | 0.969 | 0.966 | 0.961 | 0.958 | 0.958 | 0.958 |

  - At levels 1-3 the conditional is under-confident. With the real frame's weak signal and a context that shows the
    cow, noise removes it in 2-3% of draws.
  - At levels 4-6 the real path recovers as real signal grows; the sampler keeps losing, because its input is built
    from its own weakened estimates.
  - Among draws that end without the cow, the first loss is at steps 2-8 in 21 / 36 / 45 / 71 / 60 / 31 / 3 draws:
    mostly steps 4-6.
  - Of 498 draws that lose the cow at some step, 229 regain it by the end.
  - This is exposure bias in its literature sense (Ning et al. 2023, 2024): the sampler's inputs drift from training's
    as its own errors accumulate.
- **Not a hedge over movement:** cows that stayed and cows that moved are omitted alike at level 0 (9.0% against
  10.7%), and deleted alike (5.2% against 4.8%), as in C12.
- **Not rarity in the objective:**
  - entity cells carry 40-42% of the target's periphery error at levels 0-2 and 30% of the ramp-weighted objective;
  - levels 1-3 together are 57% of that objective.
  - The low levels are not starved and mobs are not a negligible share.
  - Correction (11-21, Codex): those shares are on C5's unit frames, chosen because a mob is present. On 3,000 random
    DEV frames (reading 2's, P1.6 checkpoint 10), mob cells are 1.4% of cells (1.16 of 81 per frame) and carry 23.7%
    of the ramp-weighted error: 30-31% at levels 0-2, 26% at 3, 5% at 7. The share is not negligible, but a share of
    error does not say how much of it asks the world to keep a mob rather than to place a moving one.
- **Skeletons:** the noise-free estimate shows them for only 15% of units by day. They appear only once the real
  signal is present (level 3 and above) or along the sampler's path (final 0.87).
- **Predictions:**
  - right that the conditional loses mobs at low levels (levels 1-3 for the estimate-keeps units);
  - wrong that the losses would sit only there: the sampler commits most of them at steps 4-6;
  - wrong that mobs are a small share of the objective (30%).

## C14: the day coin is not the learning rate running out (`c14_lr.py`, `c14_readings.json`, `c14_calib_ckpt{10,11}.npz`)

P1.5 was continued from checkpoint 10 for one checkpoint interval (625 updates of 256 steps, 1.6 h) at checkpoint 5's
learning-rate factor (0.579, a rate of 5.8e-5), held constant. Checkpoints 10 and 11 were each read with their own S1
reader, on the same 2,079 units with the same 8 draws per unit.

| | Units | Deletion, checkpoint 10 | Checkpoint 11 | Paired change (95%) | Real-frame floor 10 / 11 |
| :--- | ---: | ---: | ---: | :--- | :--- |
| Cows, day | 929 | 0.0600 | 0.0580 | -0.0020 (-0.0063, +0.0022) | 1.000 / 1.000 |
| Cows, night | 262 | 0.2333 | 0.2090 | -0.0243 (-0.0339, -0.0153) | 0.977 / 0.973 |
| Zombies, day | 372 | 0.0168 | 0.0144 | -0.0024 (-0.0060, +0.0013) | 1.000 / 1.000 |
| Zombies, night | 235 | 0.1170 | 0.0989 | -0.0181 (-0.0266, -0.0101) | 0.983 / 0.979 |
| Skeletons, day | 97 | 0.1469 | 0.1972 | +0.0503 (+0.0296, +0.0722) | 1.000 / 1.000 |
| Skeletons, night | 38 | 0.4704 | 0.4704 | 0.0000 (-0.023, +0.023) | 0.895 / 0.921 |

- **The day coin does not move:** -0.2 points, the interval covering zero. By the declared reading it is a floor
  under this objective, not the rate running out.
- **Night deletions fall by about a tenth** for both cows and zombies: night is still being learned. This fits C13,
  where night is an evidence problem (the noise-free estimate omits the mob).
- **Day skeletons worsen by 5 points.**
- **The monitor's G1 is 4.3% worse at checkpoint 11** (0.001828 against 0.001753), at every level: the rate was
  raised and never annealed again.
- **Limit:** checkpoint 11 is not annealed, so its noise could mask a small gain. The day coin did not move even
  while night improved, and 8 -> 10 was already flat at a low rate.
- **Prediction:** "a small fall (about a point)". Right for night (2.4 points), wrong for day (0.2).
- **For P1.6:** neither C13 nor C14 points to a change in its training.
  - Night is an evidence problem, which G1 targets.
  - The day coin is the sampler's path (exposure bias), whose candidate remedies are sampling-time and need no
    retraining: guidance limited to the mid-chain interval (Kynkaanniemi et al. 2024), restart sampling (Xu et al.
    2023) and epsilon scaling (Ning et al. 2024).
  - Training-time input perturbation (Ning et al. 2023) is unproven here and stays out.
- **C13's levels at checkpoint 11** (`c13_levels_ckpt_11.json`, own reader): the final samples agree with the
  calibration (day cows 0.949 -> 0.953 kept at K = 8; night cows 0.782 -> 0.809).
  - At level 0 the noise-free estimate shows fewer mobs at 11 (day cows 0.90 -> 0.88, night 0.70 -> 0.46), while the
    sampler recovers more.
  - Level 0 is a blend read near the reader's threshold, and checkpoint 10 was read with S2's reader while 11 used its
    own. Those level-0 differences are therefore not interpretable as a change in mechanism. The un-annealed
    checkpoint may also simply blur its zero-signal estimate.

## C15: amplified history at the commit steps cuts the day coin by a third but corrupts the HUD; restart does nothing (`c15_remedies.py`, `c15_remedies.json`, `c15_units.npz`)

C5's 2,079 units, 8 draws, P1.5 checkpoint 10, the default K = 4 sampler. A0 reproduces C5's keeps exactly (1.0).
The smoke checks are exact: A0 equals the sampler, and guidance at w = 1 equals A0 (both 0.0).

Deletion, with the paired change against A0 (same units, seeds and noise):

| | A0 unguided | A1 guidance w 2, steps 3-4 | A2 restart K 2 |
|---|---|---|---|
| cows, day (929) | 6.0% | **3.9%** (-2.14 +- 0.21) | 5.7% (-0.34 +- 0.37) |
| cows, night (262) | 23.3% | 19.4% (-3.96 +- 0.85) | 22.2% (-1.15 +- 1.02) |
| zombies, day (372) | 1.7% | 1.5% (-0.20 +- 0.21) | 1.6% (-0.03 +- 0.37) |
| zombies, night (235) | 11.7% | 10.1% (-1.65 +- 0.70) | 12.0% (+0.32 +- 0.67) |
| skeletons, day (97) | 14.7% | 10.4% (-4.25 +- 1.07) | 11.1% (-3.61 +- 1.62) |
| skeletons, night (38) | 47.0% | 36.8% (-10.2 +- 2.4) | 44.4% (-2.6 +- 3.6) |

The side conditions:

| | A0 | A1 | A2 |
|---|---|---|---|
| staying put among kept draws (cows; real 0.603) | 0.686 | 0.691 | 0.679 |
| spurious mobs per frame | 0.841 | 1.080 (**+28%**) | 0.856 (+2%) |
| token error, static HUD | 0.040 | 1.478 (**x37**) | 0.036 |
| token error, changed HUD | 5.41 | 11.20 (x2.1) | 5.55 |
| token error, mob / entering terrain | 5.49 / 3.73 | 5.61 / 3.88 | 5.52 / 3.77 |
| spread, mob / entering terrain | 1.56 / 1.63 | 1.70 / 1.71 (up) | 1.59 / 1.62 |

**By the declared reading neither arm is a candidate remedy.**
- A1 meets the primary bar: day cows 6.0 -> 3.9%, under 4.0%. It fails two side conditions: spurious mobs +28%
  against at most 10%, and static-HUD error x37 against at most x1.5.
- A2 fails the primary bar.
- The rule's consequence is that the coin points back to the conditional at levels 1-3, a training-side problem.

What the arms separate, beyond the rule:
- **Drift (i) is not supported.** Restart contracts accumulated sampler error and leaves the coin where it was, for
  cows and zombies, day and night. Only day skeletons move (-3.6 +- 1.6, 2 SE).
- **Context weight (ii) is supported as a direction.** Amplifying the history's contribution at the two commit steps
  (sigma 1.0 and 0.33) cuts deletions for every kind and light, by 10 SE for day cows.
- **The collateral is the unconditional's own error.** Where the conditional is right and the history-free prediction
  is wrong, x_u + 2 (x_c - x_u) - x = x - x_u: the guided draw takes on the unconditional's error. That happens at the
  HUD, which the history-free prediction cannot know (x37), and in spurious mobs. It is guidance's known failure with
  a weak unconditional, not a property of the coin.
- What is not separated: whether the conditional under-weights history at the commit steps (a training-side
  question), or whether amplification moves the coin through a route the conditional doesn't use. C15 tests remedies
  at sampling time, not the training cause. No follow-up test is run here; it goes to the user first.

## The day coin's error form: a mid-chain conditional error, half repaired, and a content swap, not noise (`coin_form.py`, `coin_form.json`)

P1.5 checkpoint 10, C5's 1,499 day units, the native K = 4 sampler, 8 draws, S2's readers.
- **Instruments:** the native path equals C4.samples exactly (0.0). The native final reproduces C5's keeps in 11,977
  of 11,992 draws (99.87%). The smoke had 216 of 216; the rest is batch-dependent bf16 kernels, as in C10.

**(1) Repair: start the sampler from the real frame noised to each level with the draw's own noise (day cows, 929).**

| start (tau) | 0 (native) | 2/8 | 4/8 | 6/8 | real frame |
|---|---|---|---|---|---|
| final deletion | 6.10% | 5.33% | 1.99% | 0.16% | 0 |

| step taken from a perfect input at | 0 | 2/8 | 4/8 | 6/8 |
|---|---|---|---|---|
| its estimate lacks the cow (drop) | 9.23% | 8.50% | 2.58% | 0.16% |
| final deletion it causes (delta) | 0.77 +- 0.30 | **3.34 +- 0.28** | **1.83 +- 0.18** | 0.16 +- 0.08 |
| share of its drops later steps repair | 0.92 | 0.61 | 0.29 | 0 (last step) |

- **85% of the coin (5.17 of 6.10 points) is made by the steps from 2/8 and 4/8,** even when their input is the real
  frame's own noised form. These inputs are in distribution, so the drop itself is the conditional's error at mid
  signal, not exposure bias: with a quarter of the signal and a history that shows the cow, the estimate omits it in
  8.5% of noise draws.
- **The non-repair is the exposure-bias part.** Later steps, fed the state built from that estimate, put the cow back
  in 61% (2/8) and 29% (4/8) of cases. On the real path, fresh signal puts it back by construction.
- **Step 0, from pure noise (always in distribution), drops the cow in 9.2% of estimates but 92% are repaired:** the
  noise-free estimate's omissions are not the coin (consistent with C13: 15% of day deletions).
- **By the declared reading: both.** Step 4/8 is below the 0.5 repair bar, step 2/8 between. Either lever alone would
  remove the same 5.2 points: no drops at 2/8 and 4/8, or full repair of them (the coin would be 0.93 points).
- Zombies (372): the same shape, smaller (1.65% native; delta 1.04 and 0.67 at 2/8 and 4/8; repair 0.75 and 0.51).
  Skeletons (97): 14.6% native; the estimate from 2/8 lacks them in 48% (repair 0.85), and from 4/8 in 7.1%
  (repair 0.38).

**(2) What the drifted state is: the native state's deviation from the same-noise real path at the cow's own cell**
(cows staying on their cell; 284 deleted and 4,196 kept draws).

| deleted draws | tau 2/8 | 4/8 | 6/8 |
|---|---|---|---|
| share of the deviation's energy toward the model's absence token (orthogonal to the cow) | 0.79 | 0.80 | 0.80 |
| share along the cow token (less cow) | 0.19 | 0.19 | 0.19 |
| share along the draw's own noise | 0.002 | 0.001 | 0.001 |
| deviation energy / the level's noise energy | 0.016 | 0.25 | 3.75 |
| frame-wide state energy / the real path's | 1.000 | 0.998 | 0.998 |

- **It is a content swap, not noise.**
  - The cow's cell moves toward the terrain content the model ends with (80%), with a smaller loss of the cow itself
    (19%), and almost nothing along its own noise.
  - That rules out a state that lags its label: the noise coefficient is ~0, not opposite the cow's. This is the
    premise of input perturbation and of MixFlow's slowed interpolation.
  - The frame's energy matches the real path within 0.2%: no frame-wide variance excess, the premise of epsilon
    scaling and time shift.
- **Kept draws drift the same way, less.** Absence share 0.70-0.77, cow coefficient -0.12 against -0.23 at 6/8,
  deviation 1.9 against 3.7 noise energies. This check is not circular: the absence token is the mean of the unit's
  deleted draws' finals, and kept draws do not enter it. For deleted draws their own final enters it, so their share
  is partly circular.
- **By the declared reading: a content swap.** Gaussian proxies do not imitate it. If the fix is recovery, the
  matching family trains on the model's own steps (DEFAR's simulated step, Codex's design).

**What this settles, and what it doesn't.**
- The coin needs a mid-chain error on in-distribution inputs (2/8 and 4/8) and its incomplete repair from the drifted
  state. The drift is structured (toward terrain), not noise.
- It doesn't say which lever is achievable. C14 says more training at a higher rate does not lower the coin, so the
  mid-chain error is a floor under the current objective. C15 says amplified history at those steps cuts it, so the
  history is under-weighted there.
- Limits: one checkpoint, one seed, day units only.
