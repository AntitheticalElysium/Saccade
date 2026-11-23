# Does the world's own spread know when its dreamed action ordering is wrong? (2026-11-23)

**Why.** Decision 5 (AGENT.md) proposes the world's own signals, not an ensemble, as the readings for dream trust. It
requires a signal to be validated before any use: it must predict what dream trust needs, namely wrong action
orderings on real forks. The user approved running this check.

**Data (saved, no model run).** The archived 764 DO/NOOP forks of `20261119_s2`:
- `bank_competent.npz`, `model_competent.npz`, `oracle_competent.npz`;
- 8 P1.5 dreams per first action and 16 simulator futures, continuation actions matched within a pair;
- returns with γ 0.997, reward on the terminal step included (`first_action_returns.returns`).

**Cases.**
- At depths 4 and 15, the starts where the two independent 8-future simulator halves agree on a strict ordering, as
  in the return-order audit.
- Among them, those whose mean-of-eight dreamed ordering is not a tie. Depth 1 has no opposite orderings (0.00%), so
  it is reported only for counts.
- "Wrong" is a dreamed ordering opposite to the simulator consensus.

**Claim.** The world's own spread across its 8 dreams per action predicts wrong orderings. Low confidence in the
ordering (small |mean gap| / standard error) predicts them too.

**Competing explanation.** The spread only tracks real outcome noise: wrong orderings happen where real returns are
noisy, so the simulator's own spread predicts them as well, and the dream spread adds nothing beyond it.

**Separating reading.** The dream spread's AUROC against the simulator spread's, and the dream spread's AUROC within
terciles of simulator spread.

**Readings, fixed before results** (AUROC for "wrong", episode-bootstrap 95% intervals):
1. dream spread: pooled standard deviation of the 16 dreamed returns (8 per action);
2. dream confidence: 1 / |z|, with z = mean gap / sqrt(var_DO / 8 + var_NOOP / 8);
3. simulator spread: pooled standard deviation of the 32 simulator futures;
4. dream spread within simulator-spread terciles, averaged;
5. the share of wrong orderings the world makes confidently (|z| >= 2): Biased Dreams' false negatives.

**Limits.**
- One world (P1.5) and a collection-stratified cohort with random continuations.
- The old outcome head.
- Returns, not latents. Flow instability is not saved, so it isn't read here.

**Run:** `.venv/bin/python -m artifacts.experiments.20261123_dream_trust.spread_validation`
