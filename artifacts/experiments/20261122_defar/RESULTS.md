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
- Day cows are deleted 3 points more often. Zombies are unchanged against checkpoint 10, and night zombies are worse
  than the continuation.
- The real-frame floor stays at 1.000 for day cows, so the reader still reads real cows. The loss is the world's.

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

## Explanatory readings

(Running: the token-space chain on the new checkpoint, then 8-step deletion.)
