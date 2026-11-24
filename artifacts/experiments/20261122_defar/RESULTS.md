# The drift-row fine-tune: results

P1.5 checkpoint 10, continued for 625 updates (one checkpoint interval) at checkpoint 5's learning rate (5.8e-5), with
4 drift rows per update (PLAN.md). Training ran cleanly: 11.5 s per update, peak 3.40 GiB. The monitor's flow term
ended at 0.001832, against the higher-learning-rate continuation's 0.001828 (same recipe without drift rows) and
checkpoint 10's 0.001753.

## Primary: native 4-step deletion. It got worse; by the declared reading, a failure

The day-coin panel: 2,079 mobs, 8 draws, each checkpoint read with its own S1 reader. Pairing is exact: the new
outputs' unit identities equal the rebuilt panel. Intervals resample the 166 paired episodes.

| | Drift rows | Checkpoint 10 | Change | Continuation | Change |
|---|---|---|---|---|---|
| Cows, day (929) | 8.95% | 6.00% | **+2.95 [+2.47, +3.40]** | 5.80% | **+3.15 [+2.71, +3.60]** |
| Cows, night (262) | 26.81% | 23.33% | +3.48 [+2.21, +4.85] | 20.90% | +5.92 [+4.85, +7.01] |
| Zombies, day (372) | 1.51% | 1.68% | -0.17 [-0.53, +0.20] | 1.44% | +0.07 [-0.29, +0.47] |
| Zombies, night (235) | 11.54% | 11.70% | -0.16 [-0.91, +0.62] | 9.89% | +1.65 [+0.77, +2.67] |
| Skeletons, day (97) | 21.13% | 14.69% | +6.44 [+4.09, +8.81] | 19.72% | +1.42 [-0.36, +3.16] |
| Skeletons, night (38) | 43.09% | 47.04% | -3.95 [-7.24, +0.57] | 47.04% | -3.95 [-7.99, +0.43] |

- **Declared success** (a fall of 1 point or more against both references): no.
- **Declared null** (within +-0.5 of the continuation): no.
- Zombies are unchanged against checkpoint 10, and night zombies are worse than the continuation.
- **Corrected the same day: the +2.95 is mostly faint cows, not more vanished ones.** Read in token space on the
  560 cows of the token re-read, independent of the reader (see the explanatory readings below):
  - finals with cow mass below 0.3 (clean deletions) are unchanged, 6.14% -> 5.92%;
  - mass below 0.5 rises by +1.21 points (se 0.29), 6.79% -> 7.99%;
  - half-drawn cows (mass 0.2-0.6) nearly quadruple, 1.23% -> 4.51%.

  The new checkpoint's refitted reader calls a cow only from a blend weight of 0.40, against 0.30 for checkpoint
  10's. It counts 3.4% of draws as deleted that hold half a cow or more, against 1.4%.
- The real-frame floor stays at 1.000 for day cows: real cows still read correctly. What changed is that the
  world's final frames commit less.

**Guards (against checkpoint 10), both held:**
- spurious mobs per frame 0.841 -> 0.857 (x1.02, bar x1.10);
- static-HUD token error 0.039 -> 0.042 (x1.08, bar x1.5).

Reported without a bar:
- mob token error 5.49 -> 5.48;
- changed HUD 5.42 -> 5.46;
- entering terrain 3.73 -> 3.46;
- terrain edits 2.37 -> 0.95;
- staying put among kept cows 0.686 -> 0.676 (real 0.603).

## The repair gap: it learned to repair drifted states by hedging, and hedges on real inputs too

`repair_gap.py`, day cows' 3 x 3 neighbourhoods, 4 draws, each checkpoint on its own tokens. Errors are summed squared
token error. "Clean" is the finest output on the real frame noised to the destination level. "Drift" is the output
after one deployed quarter-step from the real frame at the source level. "Source" is the output at the source level.

| | Checkpoint 10 | Continuation | Drift rows |
|---|---|---|---|
| 4/8: clean / drift / source error | 1,696 / 27,745 / 20,953 | 1,797 / 28,383 / 21,158 | **3,588** / **22,328** / 21,143 |
| 4/8: cow mass, clean / drift / source | 1.005 / 0.990 / 0.993 | 0.981 / 0.967 / 0.958 | 0.960 / 0.947 / 0.958 |
| 6/8: clean / drift / source error | 118 / 2,068 / 1,696 | 188 / 2,208 / 1,797 | 170 / 1,821 / 3,588 |
| 6/8: cow mass, clean / drift / source | 1.007 / 1.002 / 1.005 | 0.978 / 0.974 / 0.981 | 0.985 / 0.975 / 0.960 |

- **The drift term did train.**
  - Error on drifted states fell 20% at 4/8 (27,745 -> 22,328) and 12% at 6/8, against the continuation's +2% / +7%.
  - The in-run drift-to-flow ratio moved little: -1.3% at 4/8 and -4.3% at 6/8, first fifth to last. That is a
    training diagnostic, read on a moving model.
- **The same change doubled the error on real inputs at 4/8** (1,696 -> 3,588; the continuation: 1,797).
  - The cow mass in that estimate fell from 1.005 to 0.960, and on drifted states from 0.990 to 0.947.
  - So the model lowered its drift error by hedging, moving its estimate toward a blend of cow and terrain. At 4/8 a
    drifted state and a real one look alike, so it hedges on both.
  - Hedged estimates are exactly what the chain drains (`midsignal_tokens`), so deletion rises.
- The 6/8 "gap" is negative only because its floor (the 4/8 clean error) doubled; Codex's warning about a learned
  floor applies.

**Reading.**
- Under x-space MSE, the best estimate from an ambiguous drifted state is its posterior mean. That is a blend, and a
  blend is what the chain fails to resolve.
- So the loss our translation used rewards the behaviour behind the coin. That is a deviation from DEFAR: its ADR
  term is a direction-only loss toward the data, and it backpropagates through the first step.
- **Whether DEFAR's own form avoids this is untested.** This is an interpretation of the gap, not a measured cause.
  The token-space chain on this checkpoint tests the hedging part.

## Explanatory readings: the world draws cows faint, and its history-only posterior hedges too

**8-step deletion** (the finest output throughout, the one the drift term trained), read with the same reader:

| | Drift rows | Checkpoint 10 | Change | Continuation | Change |
|---|---|---|---|---|---|
| Cows, day | 7.95% | 5.06% | +2.89 [+2.44, +3.38] | 4.74% | +3.22 |
| Cows, night | 23.62% | 21.76% | +1.86 | 19.13% | +4.48 |
| Zombies, day / night | 1.51% / 10.43% | 1.21% / 10.74% | +0.30 / -0.32 | 1.08% / 9.31% | +0.44 / +1.12 |

The trained output fails the same way, so this is not the shortcut's quarter-step failing to inherit a gain. These are
reader-based numbers; the reader caveat above applies.

**The token-space chain** (`midsignal_tokens` on the new checkpoint, the same 560 cows):

| | Checkpoint 10 | Drift rows |
|---|---|---|
| History-only posterior mass M_0, same-terrain / all neighbours | 0.978 / 0.996 | **0.938 / 0.951** |
| Own-cell weight at tau = 0 (data's staying share 0.603) | 0.623 | 0.591 |
| Final cow mass, mean | 0.965 | 0.920 |
| Deleted draws' mass at estimates 0-3 | 0.90, 0.75, 0.44, 0.23 | 0.88, 0.75, 0.64, 0.38 |
| Reader drops at 2/8 (finest): rate; their median mass | 7.5%; 0.92 | 12.8%; 0.90 |

- **The hedging reaches the history-only posterior.** The drift term trained only at 4/8 and 6/8, yet M_0 fell 4
  points: given history alone, the model now expects about 5% of visible day cows to be gone.
- **The chain drains more slowly, but ends on blends.** In deleted draws the mass falls more gently (0.64 against 0.44
  at the third estimate) and stops at 0.38. The sampler's last step lands on its estimate at 6/8, a posterior mean,
  so hedged estimates become faint final cows.
- **So the drift rows didn't teach repair.** They taught every mid-level estimate to blend, and the world's samples
  now commit less.

**What this settles, and what it doesn't.**
- **Settles:**
  - training on the model's own drifted states with an x-space squared error toward the real frame makes the world
    hedge, and that raises reader-counted deletion (+2.9 at 4 and at 8 steps);
  - the dose moved the model, so this is not under-training;
  - the guards held.
- **Doesn't settle:**
  - whether DEFAR's direction-only loss, or a gradient through the simulated step, would avoid the hedging;
  - whether a faint cow matters to an actor.
- One seed, one checkpoint interval, day cows for the token readings.
- Reader lesson: a change that can blur outputs must be read in token space. The refitted reader's threshold (0.40
  against 0.30) inflates "deletion" exactly where outputs blur.
