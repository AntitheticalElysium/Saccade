# Does the world's own spread know when its dreamed action ordering is wrong? Results (2026-11-23)

Plan and code: `PLAN.md`, `spread_validation.py` (commit cc5073f9). Output: `spread_validation.json`.
- Saved data only: P1.5's archived DO/NOOP forks, with 8 dreams per action and 16 simulator futures.
- AUROC for "the dreamed ordering opposes the simulator consensus", with episode-bootstrap 95% intervals.

| Depth | Starts (consensus, dream not tied) | Wrong | Dream spread | Dream low confidence (1 / abs z) | Simulator spread |
|---:|---:|---:|---:|---:|---:|
| 1 | 130 | 0 | n/a | n/a | n/a |
| 4 | 178 | 23 | 0.39 [0.29, 0.49] | **0.85 [0.77, 0.91]** | 0.76 [0.67, 0.85] |
| 15 | 309 | 91 | 0.48 [0.42, 0.55] | **0.76 [0.70, 0.81]** | 0.59 [0.52, 0.65] |

| Depth | When the world is confident (abs z >= 2): orderings right | When it isn't: orderings right |
|---:|---:|---:|
| 4 | 103 of 106 (97%) | 52 of 72 (72%) |
| 15 | 101 of 108 (94%) | 117 of 201 (58%) |

## Reading

- **Raw spread doesn't flag wrong orderings.** The spread of the dreamed returns scores 0.48 at depth 15. Within
  terciles of the simulator's own spread it averages 0.51.
  - Spread tracks how eventful a start is, not whether the world is wrong about it. That's the confound Hansen & Wang
    remove by dividing their signals by scene motion.
- **The world's confidence in the comparison flags them.**
  - The confidence is the dreamed return gap divided by its own sampling error across the 8 draws.
  - It scores 0.85 at depth 4 and 0.76 at depth 15. The simulator's spread, which is the real outcome noise, scores
    less: 0.76 and 0.59.
  - When P1.5 is confident, its ordering is right 94-97% of the time. When it isn't, it is right 58-72% of the time,
    near a coin at depth 15.
- **Confidently wrong orderings exist but are rare:** 7 of the 91 wrong at depth 15 (8%), and 3 of 23 at depth 4.
  These are Biased Dreams' false negatives, and the action-difference checks against reality stay necessary for them.
- **What it means for decision 5:**
  - the validated signal is the confidence of a decision across several dreams, not raw spread;
  - it needs several draws per start, so it costs that many dreams;
  - it fits the evaluation readings (the action-difference checks) at once, and dream-trust weighting only on its
    trigger.

## Limits and the competing explanation not yet separated

- **Low confidence also reflects the 8 draws' own sampling noise.** Even an unbiased world errs more on near-ties
  estimated from 8 samples.
  - For dream trust the use is the same either way: don't trust a near-tie.
  - Calling the signal epistemic would need a matched control, a dreamer built from simulator futures at the same
    sample size. That isn't run here.
- **The simulator-spread terciles collapse at depth 4,** because many starts have identical spread.
- **The cohort:** one world (P1.5), the old outcome head, random continuations, a collection-stratified cohort.
- **Returns only.** Flow instability is not in the archive.
