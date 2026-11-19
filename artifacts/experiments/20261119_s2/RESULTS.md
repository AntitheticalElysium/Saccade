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
