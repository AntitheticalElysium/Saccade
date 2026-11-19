# S2 on P1.5, stage 1: results (2026-11-19)

PLAN.md's protocol, `gates.json` (all numbers below), one world seed. 4,000 terminal-inclusive DEV starts from 257
episodes; the world's 8 futures per start against the simulator's 8 (oracle B), both through the same frozen readers.
"Rejects" = beyond 3 episode-bootstrap SE (the existing rule). Stage 2 (BC, L3's decision test) is not built.

## 1. The instrument passes a correct world

- **Readers on simulator frames:** HUD exact 0.979 (health) to 1.000; asleep 0.988; entity cells recall 0.92-0.97,
  precision >= 0.987; camera position exact after 16 steps in 80% of futures (the shift reader's 3.5% per-step error).
- **Null (oracle A against B, through the readers): nothing rejects.** The label-level null reproduces `oracle_l1.json`.
- **Sanity (the recorded future through the readers against B): nothing rejects.**
- **Power through the readers:** fake consequences, immortal and sudden death are each rejected (the seven
  record-level corruptions were shown on labels on 11-14 and cannot be applied after rendering).
- The prefix beliefs agree with `encode`'s at Mamba-3's bf16 floor (median RMS 0.006, max 0.036; PLAN's correction).

## 2. What passes

- **N5, the belief on its own generated history: holds.** The HUD read from the generated belief matches the
  trajectory's own read HUD as well as the real belief matches the recording, at every depth (differences <= 0.013;
  health 0.98, energy 0.87, inventory 0.97 in both). Food and drink sit at 0.56-0.63 in both: the belief's known
  limit (P1.5 RESULTS 5), not an imagination failure.
- **Internal consistency: holds.** Health read 0 while alive: 0 in both. Reward against the read health change: 0.0036
  inconsistent (simulator 0.0034). Inventory increments and decrements per item at the simulator's rates, except
  sapling collection (0.0016 against 0.0028 per step: a 10% chance event, under-drawn).
- **Energy clock on its own history: exact** in every stratum (overdue starts: 0.454 by step 16, simulator 0.444).
- **Food and drink clocks with no drop yet in the prefix: exact** (food 0.576 / 0.575, drink 0.710 / 0.708 by 16).
- **First-step consequences:** damage, death and unlocks at step 1 match the simulator (damage beside a zombie 0.170 /
  0.169; death 0.0069 / 0.0069; unlock 0.017 / 0.017).
- **Waking:** close, slightly late (awake by step 8 0.570 against 0.628; by 16 0.881 against 0.921).
- **Return per future:** 0.146 against 0.156, within 3 SE.

## 3. What fails

**1. Mobs fade (TEST_SPEC S3's bookkeeping blocker fires for cows, skeletons and arrows).**

| Kind | Count, model / simulator (relative) | 4-step persistence, model / simulator | Blocker |
| :--- | :--- | :--- | :--- |
| Zombies | 0.203 / 0.269 (-25%) | 0.911 / 0.980 | no (both below its bounds) |
| Cows | 0.466 / 0.738 (-37%) | 0.806 / 0.959 | fires |
| Skeletons | 0.049 / 0.082 (-40%) | 0.586 / 0.852 | fires |
| Arrows (count only) | 0.033 / 0.048 (-32%) | 0.129 / 0.346 | fires |

- Partly one-step, mostly compounding: cows read 0.71 at the real start, 0.63 at step 1 (-11%), 0.54 at step 4, 0.36 at
  step 16, while the simulator's stay at 0.71-0.77; zombies 0.224 at step 1 (-7%) to 0.169 (simulator 0.280).
- Two mechanisms in the event gates: mobs vanish in view (cows by day 0.016 per step against 0.001; at night 0.041 /
  0.019) and fewer enter at the edges (cows by day 0.022 / 0.045; zombies 0.006 / 0.011).
- Arrows: drawn after a skeleton in range at 0.127 against 0.204, and without one at 0.017 against 0.006.
- Limit: a reader reads generated frames off its domain (TEST_SPEC); part of a deficit could be generated mobs the
  reader misses. The monotone decline, the in-view vanishing and the step-1 level argue for real loss.

**2. The zombie cooldown is wrong at depth, both ways.** Damage given an adjacent zombie, by steps since the last hit:
3 steps 0.057 against 0.032, 5 steps 0.118 against 0.033 (3.6x false hits while cooling), 6 or more 0.595 against 0.694
(too few when ready). Death beside a zombie by step 16: 0.647 against 0.706; all terminations 0.095 against 0.107.
The same split appears one step from real histories in P1.5's queue, so it starts in P1.

**3. Overdue food and drink drops are under-drawn.** With 20 or more steps since the last drop in the real prefix,
food drops by step 16 in 0.763 of futures against 0.928, drink 0.701 against 0.802; drink also drops early in 0.007
of futures by step 4 within 9 steps of the last drop (simulator 0). Check 15 found the food drop drawn at 0.49 one step
from real histories, so this also starts in P1.

**4. Smaller:** reward-bearing unlocks by step 16 0.233 against 0.255; damage away from zombies 0.107 against 0.157
(fewer mobs, fewer hits).

## 4. What this says for P2 (the user's rule: P2's objective is chosen from where S2 fails)

- Failure 1 is P2's designed target: compounding loss under self-feeding. P2's score (LOSSES G4) is an energy score on
  the latent tokens (patched + global), not on these reader counts (Codex): whether it penalises a disappearing mob
  has to be shown on simulated futures before a run (RESEARCH.md, the objective's power check). The event gates and
  the blocker are the reading after P2.
- Coverage of this reading: every start comes from a pure-random episode (epsilon 1.0; median 1 achievement at the
  start, none holding stone, coal or iron; the simulator's own futures contain no coal or iron increment). The states
  a competent actor visits are not measured here.
- Failures 2 and 3 start one step from real histories (P1). P2's outcome credit (RESEARCH.md 1) could calibrate the
  death probability at depth; it cannot give the world the cooldown or the overdue clock.
- Not yet measured: whether imagined decisions go the right way (L3, stage 2).

## 5. Stage 1b (PLAN.md "Stage 1b"; `gates_1b.json`, scored after Codex's selector fix)

**Where the mob loss comes from (route localisation; stage 1's bank, 1,000 starts x 4 futures, the recording's own
fixation schedule and the world's own outcomes in every arm).** Mobs per frame at steps 1 / 4 / 8 / 16:

| Arm | Mobs | Paired change against free running at 4 / 8 / 16 | Alive at 16 |
| :--- | :--- | :--- | :--- |
| Simulator | 1.03 / 1.04 / 1.07 / 1.14 | | |
| Free running | 0.94 / 0.83 / 0.71 / 0.55 | | 0.897 |
| Context fed the recorded frames | 0.94 / 0.82 / 0.69 / 0.55 | -0.010 / -0.015 / +0.000 (SE <= 0.014) | 0.898 |
| Mamba and the record fed them | 0.94 / 0.94 / 0.94 / 0.99 | +0.11 / +0.23 / +0.44 (SE <= 0.045) | 0.884 |
| Both (teacher) | 0.94 / 0.95 / 0.95 / 1.01 | +0.12 / +0.24 / +0.46 | 0.884 |

The loss comes from Mamba and the record reading the world's own frames: feeding them recorded frames recovers 96% of
the step-16 deficit, while the denoiser's context reading its own frames costs nothing (P1's diffusion-forcing context
corruption already covers it). The remaining step-1 gap (0.94 against 1.03) is one-step error. Mamba and the record were
one arm; which of the two carries it is not separated.

**Competent starts (1,000 random-plan starts; the null passes on every gate).**
- Mobs fade as in random play: count 0.78 against 1.30 (-40%), 4-step persistence 0.75 against 0.88.
  - The blocker fires for cows (0.27 against 0.48; persistence 0.72 against 0.93), skeletons and arrows.
  - Zombies: 0.39 against 0.54, persistence 0.877 against 0.971 (0.094 below, just inside the 0.10 bound).
- By step 16: damage 0.27 against 0.34, death 0.10 against 0.15, food drops 0.41 against 0.47, drink drops 0.45
  against 0.54; energy drops (0.32 / 0.31) and unlocks (0.161 / 0.161) match.
- Bookkeeping: inventory changes at the simulator's rates except sapling collection (0.0004 against 0.0017 per step),
  wood sword crafting (0.0017 / 0.0031) and wood increments (0.0039 / 0.0025, too many).

**The action's effect, DO minus NOOP (764 hold pairs, through the readers; the simulator's label effect in brackets).**

| Faced block | Step 1, world / simulator | Step 16, world / simulator |
| :--- | :--- | :--- |
| Tree (wood) | 0.78 / 0.87 [1.00] | 0.47 / 0.46 [0.53] |
| Stone | 0.77 / 0.92 [1.00] (rejects) | 0.51 / 0.53 [0.57] |
| Coal | 0.34 / 0.88 [1.00] (rejects) | 0.24 / 0.66 [0.79] (rejects) |
| Iron | 0.22 / 0.92 [1.00] (rejects) | 0.20 / 0.63 [0.73] (rejects) |
| Diamond (14 pairs) | 0.00 / 0.36 [1.00] | 0.00 / 0.21 [0.86] |
| Water (drink) | 0.67 / 0.93 [1.02] (rejects) | 0.70 / 1.11 [1.25] (rejects) |

Wood and stone follow the action at depth; coal is drawn about 40% of the time and iron about 25%, from the first step
on: the P1 increment regression, measured where it matters (Codex is on it). Drinking is under-drawn too.

**Spread (randomised PIT of the simulator's futures among the world's 8; null uniform in both banks).** Calibrated for
displacement, health drops, food / drink / energy drops and death step; biased for mobs at the end (53% of the simulator's
futures above every world future on competent starts, 48% on stage 1's bank), and on competent starts for return (33%
below every world future: the world pays more reward there).

**Revisits (TEST_SPEC S2.2, competent starts).** A cell re-entering view agrees with its last clear sighting in 0.68 of
cases within 16 steps and 0.60 at 17-64, against 0.94 / 0.88 for the simulator through the same readers: the world
redraws terrain it has seen. (The camera-shift reader is shared; stage 1's self-motion gates did not reject.)

**Readers on competent simulator frames:** HUD 0.92-0.99 per quantity, entity recall 0.92-0.94 (random play: about 1.0
and 0.95-0.97); the hold table's label column shows the attenuation.

**Mamba against the record (`routes_split.json`; the same starts, seeds and fixation schedule, the first generated
frame identical in every arm).** Paired change in mobs per frame against free running, at steps 4 / 8 / 16:

| Arm | All mobs | Cows | Zombies | Skeletons | Alive at 16 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| The record reads the recorded frames | +0.11 / +0.23 / +0.43 | +0.29 at 16 | +0.09 | +0.045 | 0.880 |
| Mamba's percept input reads them | +0.003 / +0.014 / +0.010 (SE 0.008) | +0.004 | +0.004 | +0.0005 | 0.893 |
| Both | +0.11 / +0.23 / +0.44 | +0.29 | +0.10 | +0.047 | 0.884 |
| Teacher (all routes) | +0.12 / +0.24 / +0.46 | +0.30 | +0.10 | +0.049 | 0.884 |

The spatial record carries 94% of the self-feeding mob loss; Mamba's own input carries about 2%. In P1 the record is
built from clean recorded tokens, while every frame the denoiser conditions on is noised (diffusion forcing): it is the
only conditioning route never trained on anything but perfect inputs, and in free running it integrates the world's
own frames. Its read (`record_read`) is denoiser parameters, so a denoiser-only fine-tune can change it.
