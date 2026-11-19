# P2 decisions: research pass (2026-11-19; user: research first, Codex secondary)

P2 (CURRICULUM 2) trains the world on its own rollouts: 16 exogenous windows x m = 2 rollouts per update, horizon
2 -> 4 -> 8 -> 16, a joint-trajectory energy score, pathwise gradients through the last 2 generated steps, encoder
frozen, G1 / G3 continuing on real streams. Open since 11-02 (P2_READINESS 5): outcome credit, window admission,
coverage; new with P1.5: the record's gradient, Mamba-3's carry, route by role. Nothing here is adopted; each is a
recommendation for discussion.

## 1. Outcome credit

**The defect (P2_READINESS 5).** Reward and termination are sampled and detached in the score, so their probabilities
get no gradient; a wrong death draw changes the loss without the derivative that would change its probability.

**What multi-step-trained models do.**
- MuZero / TD-MPC2 supervise reward (and value) heads on the model's own unrolled states against recorded targets.
  Their unroll is deterministic, so the recorded target is a valid regression target for the expected outcome.
  Stochastic MuZero keeps that validity by encoding the recorded chance outcome and feeding it to the unroll (teacher
  forced chance). We have no chance codes (generator v3), and an independently sampled branch's outcome is not the
  recording's (Huszar 2015; CURRICULUM 2), so pointwise branch targets are out.
- Weather ensembles train a generative simulator on its own rollouts with fair proper scores on tiny ensembles:
  AIFS-CRPS ([Lang et al. 2024](https://arxiv.org/html/2412.15832v1), section 2) trains with 2 members at N320, a
  rollout window grown from 1 to 12 steps, "gradients flow through the entire forecast chain"; NeuralGCM's stochastic
  model uses a two-sample CRPS, the compute "better spent on greater variety of initial times"
  ([Kochkov et al. 2024, supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07744-y/MediaObjects/41586_2024_7744_MOESM1_ESM.pdf)).
  Fair scores remove the finite-ensemble bias ([Ferro 2014](https://doi.org/10.1002/qj.2270)).

**Recommendation: a forecast-space fair score on each step's outcome distribution (Codex's proposal, grounded).**
- For binary events (dead by step k), with branch probabilities p_i and the recorded outcome y:
  (1 / m(m-1)) sum_{i != j} (p_i - y)(p_j - y). Its expectation is exactly the Brier score of the model's marginal
  (p_bar - y)^2 (branches independent). Each factor has the sign of -y, so it is non-negative and minimised at p_i = y:
  no degeneracy. With m = 2 it is (p_1 - y)(p_2 - y). The same U-statistic over the CDF gives a fair CRPS for the
  reward distribution.
- Death by step k from hazards, not draws: P_i(dead by k) = 1 - prod_{j <= k} (1 - d_ij) along a branch whose belief
  is fed "not terminated" (survival-conditioned, as STEP_SPEC's survival-weighted branches for the actor). Its
  expectation over such branches is the model's marginal. The draw is still sampled, detached, and absorbs that branch's
  percept score, as the contract says; the hazard gets credit through the probabilities, pathwise.
- A score-function estimator is not needed: the probabilities are available in closed form (a Rao-Blackwellised
  estimator in place of REINFORCE on the draws).
- Open choice: whether this term trains the outcome heads or only the world (heads detached in this term). Detaching
  makes the world produce states the real-trained heads read correctly (a world fix); training them lets heads absorb
  generated-domain errors. I'd detach (route by role), unverified.
- Prerequisite (Codex): a tiny stochastic example with a known answer, early deaths included, before training.

**A finding for the percept score itself.** With m = 2 the fair energy score is degenerate: if one member equals the
observation, the other is unconstrained (AIFS-CRPS, section 2.2, for the fair CRPS). AIFS uses the "almost fair" mix
alpha * fair + (1 - alpha) * standard. Our G.energy_score is the fair form; P2 would take the almost-fair mix.

## 2. The record's gradient (new with P1.5)

In a rollout, frame t is generated, enters Mamba (belief) and the record, and step t + 1's denoiser reads both. The
record has no parameters (fixed S4D-Lin), and it is built from stop-gradient tokens (LOSSES 7a, there to keep G1 from
shaping the encoder). In P2 the encoder is frozen, so that reason is gone; what remains is whether step t + 1's error
can blame step t's drawing through the record, as it can through the belief.
- Self Forcing ([Huang et al. 2025](https://arxiv.org/html/2506.08009), section 3.2) detaches the context: "we
  additionally detach the gradients of the previous frames from the current frame ... by restricting gradient flow
  into KV cache embeddings". AIFS-CRPS passes gradients through the whole state chain.
- The record is state, like the belief, not a cache of keys. CURRICULUM already passes gradients through the belief
  update for 2 steps. Recommendation: the same for the record inside the window, detached at its boundary (Codex);
  one rule for every route from a generated frame to the next prediction.
- What would test it cheaply: at P2's first update, the gradient reaching the denoiser through each route (belief,
  record, cache) on the same batch, no training. If the record's share is small the choice is moot; if large, a
  matched P2 pair (about 2 h each) answers it.

## 3. Mamba-3 and route by role in P2

- Mamba-3's kernel takes gradients through carried states (`mamba3_siso_combined` backward accepts the final-state
  gradients), but our wrapper recomputes the final angle under `no_grad` (the bf16 fix), cutting that path. Fix: keep
  the kernel's differentiable angle and add the fp32 correction detached; verify the two-call gradient against one
  contiguous call before training (Codex).
- Route by role does nothing with the encoder frozen; one Mamba pass, detaches placed per loss. Caution (Codex):
  detaching a generated percept on its way into Mamba would cut the 2-step credit to the denoiser that drew it.

## 4. Windows and coverage

- **P1's regressions are not coverage.** F4, F5, F6, F6.1 and P1.5 trained on the same corpus (`craftax_support_v2`)
  and saw the identical 25.6M target frames in the identical order (their gaze traces' target and episode columns are
  equal row for row). F4 drew coal 0->1 at 0.98 and iron at 0.90 on that exposure. Coverage was measured before:
  coal increments 9,727 (0.40% of transitions), iron 5,981, stone 88,495 (`20261104_coverage_and_gaze`); count learning
  rose with exposure, then flattened, within F2 (`20261105_consequence_trace` 6); a broader corpus at fixed compute
  (F2.1) did not move coal / iron / danger, and its exposure ratios did not predict its changes (`F21_READING.md`).
  Rare events are fragile, so they are the first to move when a recipe changes; the cause of the change is the recipe.
- **P2's windows (counted 11-04, `20261104_dataset_eda` "P2").** Exogenous H16 windows: original 277k, union 1.13M.
  Danger is covered in the union (44,372 damage, 15,261 deaths); mining is not (coal increments 23, iron 11, first
  unlocks 2 / 0), although the advanced collection already has uniform 16-action bursts after 512 / 1,024-step
  competent prefixes. Random actions rarely mine: no random-action collection will cover it.
- **What could cover competent consequences, if S2 says they matter at depth:** a plan fixed before the future is drawn
  is exogenous to that future (P2_DATA's argument, [Ortega et al.](https://arxiv.org/html/2110.10819v1)). E.g. the
  expert's actions from one simulated rollout of a start, replayed as a precommitted plan in fresh-key futures. Our
  construction, untested; a new collection.
- **Loader:** P2_DATA's admission contract is ready (terminal-inclusive, the whole horizon's precommitted actions,
  absorbing per branch). Open: original corpus only (P1.5's) or the union's exogenous windows too.
