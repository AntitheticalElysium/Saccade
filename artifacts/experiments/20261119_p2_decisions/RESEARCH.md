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

## 5. Follow-up (11-19 afternoon; user: is m = 2 enough, what do imagination-trained models do, coverage)

**Two members, verified.**
- FGN ([Alet et al. 2025](https://arxiv.org/html/2506.10772v1), section 2.2.3, 2.4, table A.2; beats GenCast and ENS):
  "during training we use N=2", the fair CRPS, rollout fine-tuning up to 8 steps "with gradients propagated back
  through the rollout" (8,000 updates at 1 step, 4,000 at 2, 1,000 at each of 3-8).
- NeuralGCM: two samples; compute "better spent on greater variety of initial times". AIFS-CRPS: 4 members at O96, 2 at
  N320 "to mitigate costs". Engression (Shen & Meinshausen 2024): two draws per data point.
- Why it works: the fair score is unbiased for any m >= 2, so its gradient is; more members only reduce variance,
  which more starts reduce too. Two caveats for us: the m = 2 degeneracy (section 1: almost-fair), and comparability:
  weather errors are dense and its losses are per-location marginals (FGN, AIFS), while our failure is sparse (one
  mob token among 81 cells x 16 steps) inside a global joint score; the energy score is weaker on dependence than on
  means and weakens with dimension ([Pinson & Tastu 2013](https://core.ac.uk/download/13805333.pdf); contested by
  [arXiv 1910.07325](https://arxiv.org/pdf/1910.07325)). So m is decided by the objective's power check, at m = 2,
  4, 8, not by precedent alone.

**What imagination-trained RL world models do.** None trains on its own sampled rollouts with a proper score:
- Dreamer 4 ([2509.24527](https://arxiv.org/html/2509.24527), 3.2-3.3): shortcut forcing on real sequences, context
  corrupted to tau = 0.1; in imagination "we only update the policy and value heads and keep the transformer frozen".
- DIAMOND, IRIS, STORM: one-step, teacher forced. SimPLe ([1903.00374](https://arxiv.org/html/1903.00374v5), 4):
  scheduled sampling to 100% with pointwise recorded targets (invalid for independent stochastic branches).
- MuZero, TD-MPC2: deterministic latent unrolls scored against recorded targets (m = 1 works because they predict
  expectations). Self Forcing video models: own rollouts scored by a teacher's distribution matching, context detached.
- So P2's objective comes from the weather lineage. It is justified here by S2's measurement (mobs lost at depth
  despite context corruption and shortcut forcing), not by RL practice.

**The objective's power check (Codex's point, before any P2 run).** On simulator futures, rendered and encoded:
corrupt them in the simulator state (a mob removed at step s, arrivals suppressed), and measure how often one batch's
P2 loss (almost-fair patched + global energy score, the declared scaling) ranks the corrupted world worse than the
true one, at m = 2, 4, 8 and the planned 16 windows per update.

**Survival coupling (Codex).** Rewards and continuation are coupled (`cont_logit(h, r)`). A branch kept alive for the
hazard score must be fed rewards sampled given survival, p(r | alive) proportional to p(r) sigmoid(cont(h, r)), or its
belief carries a reward that implied death. Unit test: Codex's two-step toy (true death 50%, 75% if unconditioned).
STEP_SPEC 3's imagined step (P3) has the same coupling: its return may use the unconditional reward, but the belief
update of a survival-conditioned branch needs the conditioned one.

**Coverage.** Two conditions for P2 data: actions fixed independently of the future being scored (P2_DATA; Ortega et
al.), and starts covering the states the actor will visit (the model error that matters is under the policy's own
state distribution: [Ross & Bagnell 2012](https://arxiv.org/abs/1203.1007)). The simulator replays any corpus start
exactly, so a target need not be a recorded future: a start, an action rule fixed in advance and fresh simulator
futures give an unconfounded target (0.04 s per start for 16 futures, CPU).
- A. The union's exogenous windows (existing): danger covered, mining not.
- B. Competent starts with fresh random plans: advanced states, tools, night danger; mining still rare.
- C. Competent starts under an observation-only BC rule acting on its own frames in each world (S2's "same feedback
  rule"): covers what BC does, mining included, and matches P3's starting policy; BC must read frozen inputs only
  (percepts), so the rule stays the same function while P2 changes Mamba. One round, not iterated with the actor.
- D. Recorded expert plans replayed open loop: valid for evaluation, but in training the world would learn to read the
  expert's plans (chosen from the clean symbolic view, night included) as evidence about hidden state, which misleads
  it under the actor's plans. Evaluation only.
- WITHDRAWN as training data (user, 11-19): B, C and D make the world simulator-supervised. That breaks TEST_SPEC's
  rule ("simulator state for EVALUATION only") and the generality the world is for (a real environment cannot be
  reset to a logged state and resampled). They stay as evaluation (S2 on competent starts, Codex's precommitted
  holds such as mine against wait). C also answers a different question from P2's contract: futures under a
  feedback policy, where P2 scores futures under the same imposed actions, and P3 will try actions BC avoids (Codex).
- P2 training data: A (recorded futures on exogenous windows, original or union corpus). Whether mining needs
  multi-step coverage at all is read first by S2 on competent starts; one-step mining stays trained by G1 / G3 on
  the whole corpus during P2. If it does, the offline answer is a collection designed exogenous and competent
  (plans committed before the future unfolds, chosen only from what the model can see), not simulator targets.

## 6. Why world models trained for imagination rarely train on their own rollouts (11-19, user)

- **Deterministic-latent models do**, against recorded targets: SPR (K-step latent predictions from the transition
  model, `2007.05929`), V-JEPA 2-AC ("a two-step rollout loss ... we only differentiate the predictor through one
  recurrent step", `2506.09985` section 3), MuZero, TD-MPC2. Valid because they regress the expected latent; for a
  sampled world it averages stochastic outcomes away.
- **Sampling world models do not:** Dreamer 1-4 (Dreamer 4 keeps the transformer frozen in imagination; shortcut
  forcing with x-prediction "to ... prevent accumulating errors over time"), DIAMOND, IRIS, DRAMA (`2410.08893`),
  Horizon Imagination (diffusion forcing, `2602.08032`), LeWM (teacher forcing, `2603.19312`). JEDI (`2605.13013`)
  conditions on its own one-step denoised output in a random half of batches ("random switching"); its ablation:
  without it, worse. That is a one-step proxy with a valid target.
- **The reasons, as far as the sources and our history show:**
  1. A sampled branch has no valid pointwise target once it diverges; regressing it onto the recording is
     inconsistent (Huszar 2015), and our own E8 (10-01, pre-saccade) found paired self-fed L1 installed a scroll bias.
  2. A valid target is distributional: several rollouts and backpropagation through the sampler, which is expensive
     and noisy. Weather (FGN, NeuralGCM, AIFS-CRPS) and Self Forcing video (a teacher's distribution matching) pay for
     it because their horizons are days or minutes.
  3. The simulation lemma bounds return error by one-step errors times a factor that grows with the learned model's
     Lipschitz constants (On Training in Imagination, `2605.06732` Lemma 1; Asadi et al. 2018); RL imagination
     horizons are short (15-16). If the one-step model is accurate on the inputs its own rollouts visit, rollouts
     stay accurate: hence context corruption, noise augmentation and switching, which train exactly that, cheaply.
- **For us:** the one-step model is not accurate enough on its own outputs. P1.5's recursion (`readings/recursion_4`,
  128 windows): mobs at step 16, 1.17 under real history against truth 1.27 (one-step error, -8%) and 0.795 free
  running (-37%); self-feeding adds most of the deficit, but far less than a constant per-step loss would compound to
  (0.9^16 = 0.19 of truth), so the world settles at too few mobs rather than collapsing. Two remedies aim at it:
  JEDI's switching (CURRICULUM 2(b): kept as "the FIRST ablation if the curriculum proves to need more self-feeding";
  one-step, valid targets, cheap) and P2's distributional rollout score (multi-step, expensive). Neither is chosen.
- **Credit:** a distributional score never labels a transition wrong; it moves the distribution of the world's
  futures toward the real ones (fewer cows than real on average at step k). Its gradient reaches each frame directly
  and, through the 2-step truncation, the drawing of the 2 frames before; a mob lost at step 3 is penalised at steps
  3-16 but only steps 3-4 can blame the drawing that lost it. Nothing stops the later frames being pushed to add mobs
  back (phantoms) instead: the power check must also read which correction the gradient favours.
