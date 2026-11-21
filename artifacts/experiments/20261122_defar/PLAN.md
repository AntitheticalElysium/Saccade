# Drift rows: DEFAR's simulated step on P1.5 (user 11-21: "I read through DEFAR and I think it's reasonable to try it"; 11-22: "3. we should use the fine-step at first ... 4. Without it ... 5. I'll let you make the call")

## The culprit this targets (measured)

`20261120_switching_dissection/RESULTS.md`, "The mid-signal estimate is a calibrated hedge" (`midsignal_tokens`,
022586d8). P1.5's own posterior keeps a visible day cow:
- at tau = 0 the finest code's estimate holds its mass at 0.978-0.996, and its own-cell weight at 0.62 (data 0.603);
- the deployed K = 4 chain keeps it in only 0.937 of draws;
- in draws that end deleted, the mass drains over the chain's estimates: 0.90 -> 0.75 -> 0.44 -> 0.23.

The chain's later steps receive the model's own hedged state, a cow superposed over several cells, which training never
shows: its inputs are always one real frame plus noise. DEFAR (arXiv 2606.28226) trains on exactly that
kind of state: one simulated inference step, then a loss at the drifted state toward the real data.

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

## Our translation

- **Drift rows:** 4 of the 16 streams per update (the bootstrap rows' 25%), drawn among the flow rows by their own
  seeded generator. That keeps it resumable and leaves the training RNG untouched, so data, levels, noise and bootstrap
  rows are C14's exactly.
- **The simulated step.** In a drift row, each frame keeps its drawn finest-grid level tau1. For frames with
  tau1 >= 2/8:
  - the real frame is noised to tau0 = tau1 - 1/4 with the same noise (DEFAR line 4);
  - one no-gradient evaluation is made there under the finest code;
  - one sampler step of 1/4 follows, with the sampler's own clamp (DEFAR line 6), giving z_hat at tau1.
- **The drift term.** At z_hat, tau1, the finest code (user item 3), with gradient: the flow term's own currency,
  w(tau1) |x_hat - x*|^2 in x-space with the ramp. It is added to the row's flow term (beta = 1) on the physical group.
  The glimpse input keeps its training form: physical tokens never read it, and the coin is physical.
- **Contexts (item 5, my call):** only the frame's own input is drifted. Its earlier frames are its row's ordinary
  training inputs (diffusion forcing levels), read through Denoiser.estimate's two-stream form, with gradients.
  - At deployment, the context is finished frames re-noised to 7/8, never mid-chain states.
  - Drifting the contexts too would train a distortion deployment never shows. It would also mix within-frame repair
    with cross-frame robustness, Self Forcing's problem.
- **Deviations from DEFAR, stated:**
  1. x-space MSE toward x* instead of the cosine velocity loss. Our objective is x-prediction (Dreamer 4), where x* is
     the rectified target itself with no 1/(1 - t) rescaling, and a cosine would discard magnitude, which carries
     presence.
  2. No gradient through the first evaluation (user item 4). DEFAR's Algorithm 1 has none on x_hat, so its ADR
     gradient also reaches v_t0. Left for an ablation.
  3. A fixed step of 1/4, the deployed sampler's, instead of random (t0, t1).
  4. No FC term. With alpha = 1 and a sum normalisation, its extra weight totals at most one position per frame, and it
     is a second component. Left for an ablation.
  5. 25% of the rows, not every sample.
  6. A fine-tune of one checkpoint interval, where DEFAR's fine-tunes ran 45k iterations.
- **Recipe: C14's exactly.** P1.5 checkpoint 10, its optimizer continued, one checkpoint interval (625 updates of 256
  steps), the learning rate held at checkpoint 5's factor (5.8e-5).
  - **Matched control: C14's run** (`20261120_switching_dissection/c14_run/ckpt_11.pt`), already trained. Same data,
    levels, noise, bootstrap rows and schedule; the arms differ by the drift term.
  - C14 alone moved the day coin -0.2 points (interval covering 0). Night fell about a tenth, and day skeletons rose by 5.
- **Cost:** per drift row, a two-stream no-gradient pass and a two-stream gradient pass in place of the flow pass.
  Measured in `defar.py check` against C14's 9.27 s per update and 3.23 GiB peak.

## Readings (declared before running)

Each checkpoint is read with its own S1 reader, on C5's 2,079 units, 8 draws, paired over units against C14's
checkpoint 11.
1. **Primary: day-cow deletion** at K = 4 (deployed; C15's A0 arm) and K = 8 (the finest code, the one trained; C13's
   sampler).
   - **Success:** a fall of 1.0 point or more at K = 4 or K = 8, with the paired 95% interval excluding 0.
   - **Null:** within +-0.5 points.
2. **Mechanism** (`midsignal_tokens` on both checkpoints): the tau = 0 posterior mass M_0 and the K = 4 chain's keep.
   - Through the claimed route, the keep rises toward M_0 and M_0 stays.
   - A fall in the coin without the gap closing is inconsistent with the route, and is reported as such.
3. **Guards** (C15's side conditions, A0 arm, both checkpoints):
   - spurious mobs per frame: at most +10%;
   - static-HUD token error: at most x1.5;
   - reported, without a bar:
     - changed-HUD and mob / entering-terrain token error;
     - staying put among kept cows (real 0.603);
     - night deletions, zombies and skeletons;
     - the monitor's G1 flow term (at most +5% against C14).

**What each outcome means:**
- **Success with the guards held:** the repair is learnable from the model's own states. That makes it a candidate for
  the next world's recipe, to decide with the user.
- **Success at K = 8 only:** the shortcut's d = 1/4 output doesn't inherit it, and the step-code choice returns.
- **Null:** at this dose and form, own-state training doesn't repair. Dose and form questions (rows, length, gradient
  through, FC) go to the user before any conclusion.

## Limits

One checkpoint interval; one seed (no compute for another); day cows. A fine-tune on a model trained without drift
rows, not a from-scratch recipe.
