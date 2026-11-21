# The drift-row fine-tune: DEFAR's simulated step on P1.5 (user 11-21: "I read through DEFAR and I think it's reasonable to try it"; 11-22: "3. we should use the fine-step at first ... 4. Without it ... 5. I'll let you make the call"; then Codex's four points, adopted, and "controls ... only after success for the first arm")

## The culprit this targets (measured)

From `20261120_switching_dissection/RESULTS.md`, "The mid-signal estimate is a calibrated hedge" (the token-space
re-read, 022586d8). P1.5's own posterior keeps a visible day cow:
- at tau = 0 the finest code's estimate holds its mass at 0.978-0.996, and its own-cell weight at 0.62 (data 0.603);
- the deployed 4-step chain keeps it in only 0.937 of draws;
- in draws that end deleted, the mass drains over the chain's estimates: 0.90 -> 0.75 -> 0.44 -> 0.23.

The chain's later steps receive the model's own hedged state, a cow superposed over several cells, which training never
shows: its inputs are always one real frame plus noise. DEFAR (arXiv 2606.28226) trains on that kind of state: one
simulated inference step, then a loss at the drifted state toward the real data. This is a DEFAR-inspired recovery test,
with DEFAR's directional loss replaced by Saccade's clean-frame loss (Codex).

## What DEFAR does (read at the source, Sec. 3-6 and Algorithm 1 verbatim; appendix tables only summarised)

- Algorithm 1:
  - x_t0 = t0 x* + (1 - t0) eps and x_t1 with the same eps;
  - v_t0 = f(x_t0, t0), and v_t1 = StopGradient(f(x_t1, t1));
  - x_hat = x_t0 + (t1 - t0) v_t0, "single-step inference simulation", with no stop-gradient on it;
  - ADR: a cosine loss between f(x_hat, t1) and v_ADR = a'_t1 x* + b'_t1 x_hat (for the linear path, x* - x_hat);
  - FC: the ordinary flow loss at t0, reweighted per spatial position by W = 1 + alpha |delta_ij|^2 / (sum |delta|^2
    + xi), with delta = f(x_hat, t1) - v_t1;
  - L = 10 L_ADR + 1 L_FC, alpha = 1.
- (t0, t1) are random with t1 > t0, and there is no fixed gap.
- From scratch on ImageNet-256: FID 56.39 against SiT's 61.64 at 500k iterations. Without FC 59.77, without ADR 57.17.
- Fine-tunes of REPA-XL/2 and DDT-XL/2: 45k iterations, FID 3.99 -> 3.46 and 1.90 -> 1.82.
- About 1.7 times the cost per iteration.
- Tested at 30 or more sampling steps only. Images only, velocity prediction, no history conditioning.

## The design

- **Drift rows:** 4 of the 16 streams per update (the bootstrap rows' 25%), drawn among the flow rows by their own
  seeded generator. That keeps it resumable and leaves the training RNG untouched, so data, levels, noise and bootstrap
  rows are exactly those of the higher-learning-rate continuation.
- **The drifted query, for every frame of a drift row (Codex 3).** Its destination is 4/8 on even frames and 6/8 on odd
  ones, a fixed split that draws no randomness.
  - The real frame is noised to the source level (2/8 or 4/8) with the frame's own noise (DEFAR line 4).
  - One no-gradient evaluation is made there with the **deployed quarter-step output** (Codex 1), followed by the
    sampler's own step and clamp to the destination (DEFAR line 6). That is exactly the step whose repair failed: 2/8
    and 4/8 are the steps with the largest effect on the coin.
- **The drift term.** At the drifted state and the destination level, the **finest output** (user item 3), with
  gradient: the flow term's own currency, w(tau) |x_hat - x*|^2 in x-space with the ramp. It is added to the row's flow
  term (beta = 1) on the physical group. The glimpse keeps its training form: physical tokens never read it, and the
  coin is physical.
- **No gradient through the simulated step** (user item 4); no frequency compensation. Both are left as later arms.
- **Only the target query is drifted (item 5; Codex 3).** The drifted frame is evaluated in a second stream that reads
  the row's ordinary training inputs for its earlier frames (Denoiser.estimate's two-stream form, with gradients). It
  is never the context of a later frame.
  - At deployment, the context is finished frames re-noised to 7/8, never mid-chain states.
  - Drifting the contexts too would train a distortion deployment never shows. It would also mix within-frame repair
    with cross-frame robustness, Self Forcing's problem.
- **The bootstrap rows keep running unchanged,** so the shortcut's quarter-step output has a route to whatever the
  finest output learns (Codex 1).
- **Deviations from DEFAR, stated:**
  1. x-space MSE toward x* instead of the cosine velocity loss. Our objective is x-prediction (Dreamer 4), where x* is
     the target itself, and a cosine would discard magnitude, which carries presence.
  2. No gradient through the first evaluation. DEFAR's Algorithm 1 has none on x_hat.
  3. Two fixed transitions on the deployed ladder instead of random (t0, t1).
  4. No FC term.
  5. 25% of the rows, not every sample.
  6. A short fine-tune (below).
- **Recipe: the higher-learning-rate continuation's exactly.** P1.5 checkpoint 10, its optimizer continued, one
  checkpoint interval, the learning rate held at checkpoint 5's factor (5.8e-5). The same rate means that continuation,
  already trained, is a free reference for what the fine-tune alone does. It moved the day coin -0.2 points (interval
  covering 0), night deletions -2.4 points, and day skeletons +5 points.

## Dose: 625 updates (one checkpoint interval, about 2 h), and why it should give a signal

- **Examples.** 4 drift rows x 256 frames give 1,024 drifted queries per update, 640k over the run. Random frames hold
  1.16 mob cells each, so that is on the order of 700k drifted mob cells. The hedged, low-contrast cases that fail make
  up a few percent of those, tens of thousands.
- **Precedent at this exact dose.** The continuation, at the same 625 updates and rate, moved night deletions by 2.4
  points (95% interval -3.4 to -1.5) and day skeletons by 5. Deletion rates do move at this dose when a gradient asks
  for it.
- **Detectability.** On the day-cow panel the paired interval half-width was about 0.43 points, so a 1-point change
  is resolvable.
- **The gradient is concentrated.** On drifted states, the error should exceed the flow term's, since the model has
  never been trained there, and it sits on the cells that drift: the hedged mobs.
- **Against DEFAR's dose.** Their fine-tunes ran 45k iterations of 256 images, about 18 times our drifted examples, and
  they repair whole-image error rather than one failure. No guarantee.
- **Read inside the run:** the drift term against the flow term over the updates (`drift_terms.jsonl`). A falling
  ratio means the model is learning to repair its own states. If the 4-step coin then doesn't move, the repair doesn't
  reach deployment.

## Readings (declared before running)

Each checkpoint is read with its own S1 reader, on the day-coin panel (2,079 mobs, 8 draws), paired over units.
1. **Primary: native 4-step day-cow deletion** (Codex 4), against P1.5 checkpoint 10 (6.0%) and against the
   continuation's checkpoint 11 (5.8%).
   - **Success:** a fall of 1.0 point or more against both, with the paired 95% interval excluding 0.
   - **Null:** within +-0.5 points of the continuation.
2. **Explanatory** (not success criteria):
   - **8-step deletion.** A gain there alone would show recovery is learnable, but leave the 4-step deployment
     unresolved.
   - **The token-space chain readings on the new checkpoint:**
     - the tau = 0 posterior mass should stay;
     - the 4-step keep should rise toward it;
     - the mass in deleted draws should stop draining.
     A fall in the coin without the gap closing is inconsistent with the route, and is reported as such.
   - **The drift-term curve.**
3. **Guards** (the sampling-time remedies' side conditions, native arm, against checkpoint 10):
   - spurious mobs per frame: at most +10%;
   - static-HUD token error: at most x1.5;
   - reported, without a bar:
     - changed-HUD and mob / entering-terrain token error;
     - staying put among kept cows (real 0.603);
     - night deletions, zombies and skeletons;
     - the monitor's G1 flow term.

   If a guard fails against checkpoint 10, the continuation is evaluated on it before blaming the drift rows: the
   fine-tune alone shifts things.

**No control arm now** (user 11-22). Only if the primary succeeds does the analytic-input control run. It is the same
run with the real frame noised to the destination level in place of the drifted state, same frames and loss weights
(Codex 3). It separates training on drift from extra weight at those levels.

**What each outcome means:**
- **Success with the guards held:** the repair is learnable from the model's own states. Then the control, and the
  question of the next world's recipe, go to the user.
- **8-step only:** the step-code question returns.
- **Null with a falling drift term:** the repair is learned but doesn't reach deployment.
- **Null with a flat drift term:** not learned at this dose. Either outcome goes to the user before any next run.

## Limits

One checkpoint interval; one seed (no compute for another); day cows. A fine-tune of a model trained without drift
rows, not a from-scratch recipe.

## Run notes

- **11-22 check (`defar.py check`).** Exactness: the two-stream pass with q = c equals forward() to 0.0, and the drift
  rows' flow term equals shortcut_loss's to 0.0. The first launch ran out of GPU memory in the first update's backward,
  when two drift rows could share a micro-batch. Drift rows are now at most one per micro-batch, with
  `expandable_segments` on.
- **Cost and memory**, 50 updates from checkpoint 10: 11.5 s per update in steady state, against the continuation's
  9.27 (+25%); peak 3.42 GiB, against 3.23.
- **First drift terms** (per frame, ramp-weighted): 0.0045 at 4/8, 0.0029 at 6/8, against a flow-term mean of 0.0021.
  Over those 50 updates the 4/8 term fell 4.5% and the flow term 1.7%; too short to read.
- **Launched** as unit `saccade-defar` (MemoryMax 11G, below the 11.9 GB free). It resumed at update 15625 at the
  continuation's constant factor 0.579. Expected about 2 h, then `readings.py run`.
