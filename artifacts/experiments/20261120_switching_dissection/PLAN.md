# Why one-step switching on the record failed (dissection, 2026-11-20)

User: "understand why it failed before the control"; take the time. There is no new training run here. Stage A reads the
saved S2 rollouts on CPU. Stage B (GPU forward passes) is proposed after A and runs only after the user's review.

## What failed

`20261119_switching/RESULTS.md` (`sw_rec2`, P1.5's optimizer continued at 1e-5; the record reads the denoiser's own
one-step estimates on half the frames, p = 0.5; 4,000 denoiser-only G1 updates):
- **Counts:** step-16 mobs 0.603 → 0.718 (simulator 1.196), cows and arrows only.
- **Persistence:** 4-step persistence lower (zombies 0.911 → 0.864, skeletons 0.586 → 0.498).
- **Arrivals:** interior appearances 2-3x, more vanishings.
- **Clocks:** the hunger clock regresses by 4,000 (on schedule 0.873 → 0.781); energy holds.
- **Loss:** the training loss is flat over the 4,000 updates (0.00163 at 100 and at 4,000). On the held-out monitor the
  low-signal levels rise (level 0 +2.5%, level 1 +1.6%, monotone at both reads) and the high-signal levels fall
  (level 7 -1.8%).

## What was predicted before the rerun (transcript, 11-19)

- **The user:** switching "might make the record less reliable for the denoiser".
- **Me, after the first arm** (11:12): one-step estimates at low signal under-report mobs that the target shows. The
  denoiser would learn to add mobs the record does not back up and to trust the record less. Discounting the record
  removes the clocks' source at any learning rate. A 1e-5 rerun would show "the same direction, only weaker".
  The rerun did.
- **The same message split "94% in the record" three ways:**
  - (a) the denoiser's one-step errors are biased toward deleting mobs;
  - (b) the denoiser over-trusts the record;
  - (c) Mamba over-trusts it.

  The readers split has since ruled out (c). Switching acts on (b) only.
- **Codex, plan review:** one-step substitutes made from recorded history "may miss the correlated errors that build up
  in a real rollout".
- **Codex, clocks:**
  - After the 100-update memory-route smoke, due hunger drops fell 42/48 → 35/48.
  - On frozen P1.5, Gaussian noise on the record's input (signal 0.875) cut energy 48 → 23 and hunger 42 → 23. That is
    variant 7 in its extreme, untrained form.

So the shrinkage reading below is last night's prediction. Stage A tests it; it is not a new idea.

## Candidate explanations, and what separates them

1. **Shrinkage.**
   - The idea: the substitutes are noisier, roughly unbiased copies of the recorded frame. The loss-optimal answer to a
     noisier record is to trust it less and fall back on the prior and the other inputs. On a fading record (test),
     that draws mobs at the prior rate in the wrong places (counts up, spawns) and keeps fewer of the record's own mobs
     (persistence down). The low-signal monitor rise fits this.
   - It predicts:
     - the change is already present at step 1, from a real history (a one-step behaviour change);
     - extra appearances are mostly spawns away from any vanished mob;
     - the start's own mobs die faster.
2. **Mismatch.**
   - The idea: the training perturbation (independent estimates on half the frames, each conditioned on the real past)
     differs in size or kind from free running's (every frame generated, errors correlated along the rollout). The
     robustness learned would then not apply at test.
   - It predicts: step-1 behaviour from a real history is nearly unchanged, and the differences grow with depth.
   - Measuring the perturbations directly needs Stage B.
3. **Dose.**
   - The idea: switching moves the loss very little (the training loss is flat), so there is little signal on mobs, and
     what changes is drift.
   - It needs Stage B: the loss gap between the real and the switched record on the same windows, by region (mob cells,
     background, HUD).
4. **Recipe.**
   - The idea: extra denoiser-only training at 1e-5, without switching, does it.
   - The test is the matched control (`--substitute 0`). It runs only after this dissection (the switching plan's own
     header).
5. **Motion.**
   - The idea: the extra "appearances" are mobs jumping two or more cells. S2's gates count such a jump as one
     vanishing plus one interior appearance (`saccade/eval/gates.py`, 3x3 neighbourhood), so they are not creation.
   - It predicts that interior appearances pair with same-kind vanishings nearby.

## Stage A: `events.py` (CPU; S2's saved grids)

- **Worlds:** P1.5, `sw_rec2` at 2,000 and 4,000, `sw_rec` at 2,000 (the first arm's churn), and the simulator's
  half B. The same 4,000 starts, plans and reader are used for all.
- **Bootstrap:** by episode, paired by start.
- **Readings:**
  - a. Cell rates per transition, by depth (step 1; 2-4; 5-8; 9-16), per kind:
    - edge arrivals;
    - interior appearances, split into jumps (a same-kind vanishing within 3 cells in the same transition) and
      spawns (none);
    - vanishings, split the same way.
  - b. Survival of the mobs present at the start.
    - Tracking: a greedy tracker, alive while a same-kind mob is within 1 cell of its last position (Craftax mobs move
      at most one cell per step). It is censored when its neighbourhood leaves the view.
    - Reported as a hazard per step at steps 1, 2-4, 5-8 and 9-16.
  - c. Survival of model-born mobs (born at step 1 or later, by birth type), with the same tracker: the hazard over the
    first 1, 2 and 4 steps after birth.
- **How each reading separates the candidates:**
  - Step-1 rates and start-mob hazard at step 1 differ from P1.5: (1) over (2). Equal at step 1 and diverging with depth:
    (2).
  - A jump share of the extra appearances above one half: (5) carries the churn.
  - Start mobs die faster than P1.5's: (1).
  - Persistence falls only through newborn flicker: the record's mobs are kept as well as before, and the fall comes
    from short-lived new mobs.

## Stage A2: does a deleted mob come back? (`resurrect.py`, CPU, the same saved grids)

- **Why:** switching was meant to teach recovery. When a frame drops a mob that the target still has, the denoiser
  should redraw it from the record's history. Shrinkage teaches distrust instead. The two predict opposite changes in
  the rate at which deleted mobs come back.
- **Reading:**
  - Track every mob as in stage A until its death step s: no same-kind mob within 1 cell of its last position, still
    in view.
  - Recovery = a same-kind mob within 1 cell of that position at step s + 1 or s + 2, before the neighbourhood leaves
    the view.
  - Split deaths at step 1 (from a real history) from deeper ones. In the simulator, deaths are mostly reader misses,
    so its rate is the instrument's floor.
- **What separates:**
  - Recovery rises in R2k / R4k against P1.5: switching taught the lesson, and attenuation outweighed it.
  - Equal or lower: the lesson did not land.

## Stage B (user 11-20: plan and run both, then discuss)

### (1) Where P1.5's one-step deletion comes from

From a perfect history, P1.5 loses 9.5% of the start's cows at step 1 (simulator 1.2%). Candidates:

- **Motion uncertainty:** a mob whose next cell is uncertain has its probability spread over several cells. If the cells
  resolve nearly independently, each cell's likeliest state is "empty", and the mob vanishes. This is our own 10-14
  finding ("direct = per-tile own-noise coin flips").
- **Scroll:** the record is per screen cell, not registered to the world. On a camera shift every mob must move one
  screen cell, while the record under its new cell holds a different world cell's history.
- **Sampler discretisation:** the shortcut sampler takes K = 4 steps, while k_max = 8 is the finest trained step. Few
  steps may fail to commit sparse content.
- **Instrument:** the reader may miss atypical generated mobs. Not separated here; it is a stated limit.

**A3 (`deletions.py`, CPU, saved grids).**
- **Units:** isolated start mobs (no same-kind mob within 2 cells; reader and labels agree at state 0; inner cells).
- **Measure:** the step-1 deletion rate in P1.5's and R4k's 8 futures.
- **Splits:**
  - by the simulator's own motion of that mob over its 16 label futures (P(stays in its cell): 1, [0.5, 1), (0, 0.5),
    0);
  - by whether the world's own camera moved at step 1;
  - by ring (next to the view edge or inner);
  - by day or night.
- **Instrument floor:** the same measure on the simulator's half B read through the reader.

**K sweep (GPU, frozen P1.5).**
- **What runs:** S2's world futures on the first 1,000 starts at K = 1, 2, 4, 8.
  - `s2.py` gains an `S2_STEPS` override of `edm.sampler.steps`, default unchanged.
  - Tags `_K1`, `_K2`, `_K4`, `_K8`; the K = 4 run is this process's own reference.
- **Read with `events.py`'s measures on those starts:** start-mob hazard at step 1 and depth, spawns, arrivals,
  vanishings.
- **What separates:**
  - Deletions fall with K: the sampler's coarse steps.
  - Flat: the denoiser's conditional (the model), with the motion and scroll splits locating it.

### (2) B2: record reliance by region (`b2.py`, GPU, P1.5 and R4k)

- **Windows:** the monitor's 2,000 held-out windows (its picks, noise and fixations). Added: up to 300 windows whose
  target is a food drop and 300 an energy drop, read by S2's HUD reader on real tokens.
- **Error:** the flow term's per-cell x-error on the target frame at each of the 8 levels, under three records:
  - real;
  - none (history removed, record_read's offset kept);
  - switched: half the frames, by a fixed draw, replaced by one-step estimates at random finest-grid levels, made in
    64-frame chunks as in training. Mamba re-reads the switched record, as on the rec route.
- **Regions, read by S2's frozen reader on the real frames:**
  - mob cells;
  - cells next to a mob;
  - cells a mob just left;
  - other map cells;
  - the food and energy cells;
  - other HUD cells.
- **Readings:**
  - Reliance = none - real, at low signal (levels 0-3) and at all levels.
  - The switching penalty = switched - real, per region: its size and its share of the frame's loss is the dose.
- **Predictions of attenuation:**
  - R4k's reliance falls against P1.5's at mob, near-mob and vacated cells and at food drops, and less at static cells.
  - R4k's switching penalty is smaller (it learned its task).
  - Energy drops: the penalty is small at both, if switching's estimates are accurate on the energy cell.

## Stage A4: are mobs under the glimpse deleted less? (`fovea.py`, CPU; user 11-20: "could fixation help?")

- **Step 1, a natural experiment.**
  - The real history's frames in S2 were encoded at a seeded uniform fixation (`S.encode`: `D.fixation_prior` seeded by
    the episode's shard and slot), rebuilt exactly here.
  - Whether a start mob lay in the last real frame's 3x3 glimpse window is therefore random with respect to the mob.
  - Its step-1 deletion rate, glimpsed or not, is split by day and night and by ring.
- **Deeper steps, observational.**
  - The chooser picks generated frame s's fixation after drawing that frame's periphery, so a glimpse can only act on
    frame s + 1, through the context and the belief.
  - Reading: the hazard of a tracked mob at s + 1 when it lay in frame s's window, against not.
  - The chooser selects where it looks, so this comparison is confounded by what it chose.
- **What separates:** glimpsed mobs deleted much less at step 1 means fine detail in the context protects the mob
  (fixation as a lever). Equal rates mean the glimpse does not reach the next frame's periphery.

## Stage A5 and A6 (user 11-20: "let's get it a run and dissection as well")

- **A5 (`a5.py`, GPU, after stage B): the paired fixation intervention.**
  - Built on S2's world loop, copied for step 1 only, with the same batches and seeds.
  - Each start's last real frame is re-encoded in that start's own history copy. Only the glimpse and the fixation
    change; the periphery and lighting do not depend on the fixation, which an assertion checks.
  - Arms:
    - orig: the seeded draw; must reproduce S2's saved step-1 frames;
    - on: the window centred on one isolated inner start mob per start;
    - away: the farthest map window that excludes it.
  - Reading: the target's step-1 deletion, on against away, paired per future. Split by kind, day or night, and ring,
    with episode-bootstrap SEs.
  - Separates: on lower than away at the night edge means the glimpse protects where the periphery token is corrupted
    (the 11-20 hint: -9 points at 1.4 SE). Equal means it does not.
- **A6 (`gaze_dev.py`, CPU): the learned fixation across the learned-gaze worlds (F2 → P1.5).** F2.1 is excluded: it
  trained on `craftax_union_v1`, not on the same frames, and the loader refuses it against our replay store. The other
  nine run.
  - The F2 review's `trace.py` runs unchanged on each world's executed training trace, on the same labelled training
    subset.
  - Reading: crop coverage of entities by kind, HUD changes, terrain edits, entering map and the counters' attempts,
    against paired uniform draws, by day and night. Also dwell and the logged C / S / entropy.
  - Check: the F2 rerun reproduces the archived `20261104_p1f2_review/TRACE.json`.

## The eye's checks (user 11-20: "agreed for the checks; scope them out")

- **C1: why P1.5's chooser predicts better.**
  - The facts: its head error runs 0.031 → 0.015 over training, where F2 / F4 / F6 / F6.1 run 0.013 → 0.03, at a
    similar token scale. Its surprise score runs 4.95 → 2.0, where the others run 2.1 → 4.1.
  - Suspect: route by role. Since P1.5, only G1, TSP and reconstruction train the encoder; in F6 the reward heads'
    gradient on the encoder was 6-11x TSP's.
  - Reading, on the same DEV frames encoded by F6, F6.1 and P1.5, each with its own final chooser head:
    - frame-to-frame token change and the head's surprise, by region: mob cells, HUD cells at a digit flip, terrain
      entering the view, static map; day and night;
    - each policy's coverage.
  - Separates: whether the tokens' own dynamics changed (encoder) or the head did, and where the surprise mass moved.
- **C2: where a memory-informed eye would look.**
  - The world's own zero-signal error used as surprise. Same window logic, S-only shape.
  - A first read from B2's saved errors on the monitor windows (no labels for night): it covers a mob 57.4% of the time
    when one is present (uniform 21.2%, argmax 75.8%), and the HUD at uniform (29.1% against 28.6%).
  - Full version on DEV with labels:
    - day / night (night static is irreducible: the noisy-TV risk);
    - entering terrain;
    - the cows the world deletes at step 1: is its surprise high where its own prediction lost the cow?
- **C3: the cheap memory read.**
  - The chooser head refit offline on frozen DEV tokens with a per-cell read of the record (like `record_in`), against
    the same head without it.
  - Reading: where its surprise moves, and whether HUD-flip surprise drops.
  - Compared with C2, this separates "the eye has the world's memory" from "the eye has the world's own model".
- **Order:** C1, C2, C3, one GPU job at a time. Each is committed as code before it runs.

### C2's cost and cache (user: "don't guess - test it")

- **Baseline, measured by the run itself** (`20261116_p15/run/log.jsonl`, `sec_per_update`):
  - 64-step chunks: 2.284 s per update over 12,500 updates;
  - 256-step chunks: 8.581 s per update over 3,125 updates;
  - total 15.4 h, matching STATUS's 15 h 20 min.
- **Fresh world-eye (`c2_cost.py`).**
  - Needed per target frame: the world's zero-signal estimate of that frame given its real past. In a single-stream
    diffusion-forcing pass, a frame's noisy tokens are also later frames' context, so the exact estimate needs a
    sequential pass: per frame, one periphery evaluation at tau = 0 (the sampler's first step), then the frame's
    context pass.
  - Timed after warm-up, at the trainer's shapes: 16 streams x (8 prefix + 64) frames and 16 x (8 + 256), from P1.5
    checkpoint 10 on its own training streams, with the memory tokens and record from the trainer's own passes
    (computed, not timed).
  - Projection: 12,500 x t64 + 3,125 x t256.
- **Cache.**
  - Read cost: a memmap of fp16 estimates per update, timed.
  - Fill cost: the same estimate made offline in large batches, as throughput per frame.
  - Saving at refresh period k passes: the in-loop cost minus (the fill cost x 10 / k passes) minus reads.
  - The quality side is measured, not assumed. On the same DEV windows, the eye's window probabilities from the stale
    estimate of P1.5 checkpoints 9, 8 and 5 (1, 2 and 5 passes stale) are compared against checkpoint 10's encoder
    tokens: total-variation distance, argmax agreement and mob coverage, against the fresh checkpoint-10 eye.
- **In imagination the expectation is free:** the sampler's first evaluation is the zero-signal estimate. The cost is
  training-only.

### C3, decisive version (user 11-20: "yes")

1. **Readability probe (`c3_probe.py`).** A linear logistic read of the record through t - 1 at the food / drink /
   energy cells against the chooser's own view (the last 8 coarse tokens at that cell), and both. Held-out AP for a
   flip of that digit at t, on C3's split.
2. **If the record beats the 8-frame view clearly:** C3 rerun with the new read at the chooser's peak 1e-3 (fresh
   parameters), the trained head at 1e-4, about 6,000 updates. Otherwise the record does not carry what the eye would
   need, and C3 stops there.

## C4: the sample-eye (user 11-20: "run that sample-eye on the data to see if it does produce coverage worth the cost")

- **What it measures.** Surprise is the squared distance between the real frame's tokens and the world's own sample of
  that frame: the sampler's K = 4 steps from noise, given the real past (the monitor's window, the real record and
  belief). C2's world-eye used the zero-signal mean instead. Windows go through the chooser's policy with contingency
  off.
- **Same data as C2:** its 3,000 DEV target frames and its 2,079 S2 start mobs. Two draws per frame (separate noise
  seeds): the eye's surprise is their mean, and their spread is reported beside it.
- **Readings:**
  - coverage of entities, cows, zombies, HUD flips and entering terrain, by day and night, beside C2's world-eye and
    the chooser;
  - surprise shares by region;
  - the deletion link. The draw's own outcome is read at the mob's cell (the S1 reader on the sampled periphery);
    the eye's coverage of mobs its draw deleted is compared against mobs its draw kept, and P1.5's S2 deletion rate
    is binned by the sample-eye's surprise.
- **Cost:** as C2's cost test (`c2_cost.inputs`, the trainer's shapes, warmed), with K periphery evaluations per frame
  plus its context pass; projected on P1.5's measured updates.
- **What would make it worth its cost:** clearly higher coverage of mobs than the chooser and C2's world-eye, with the
  deleted mobs covered well above the kept ones.

## C5: the sampler's calibration at mob cells, and does Mamba know where the mobs are (user 11-20: "let's run 3. first since it actually attacks our failure"; "shouldn't Mamba know about mobs? Does it?")

C4 showed that reality and the world's mean both keep the cow, and only some of the world's samples drop it. C5
measures that sampling distribution directly, at the deletion.

- **Units:** C4's 2,079 S2 start mobs (frame t = t0 + 1 of a DEV episode; the real past, record and belief), 8 draws
  each with the default sampler (K = 4, `c4_sample.samples`; draws 0-1 are C4's). A draw keeps the mob if the reader
  finds its kind in the 3 x 3 neighbourhood of its cell after the real camera move (C4's rule).
- **(1) Calibration.** The keep rate over draws against the reader on the real frame t at the same cells (the
  instrument floor), by kind and by day / night.
- **(2) Concentration.** Deletions per unit out of 8, against a binomial at the pooled rate of its kind and light:
  share of units never deleted, deleted in >= 4 of 8, the share of all deletions they hold, and the dispersion index.
  - Over-dispersed: particular contexts lose the mob most of the time; the world's conditional is wrong for them.
  - Binomial: every draw flips the same small coin; the deletion is spread over all contexts.
- **(3) Where on the ladder.** The reader's P(kind), max over the neighbourhood, on each step's clean estimate (4 steps)
  and on the final sample, for draws that end deleted against kept; how well each step's P predicts the end (AUC).
  Plus the deleted cell's form: the reader's class and confidence at the mob's cell in the final sample, against the
  real frame's (a confident other class = a committed alternative; low confidence = a blend).
- **(4) Does Mamba know?** A linear probe per map cell of entity presence (any entity; cows) at frame t:
  - from the predicted belief that the denoiser reads for frame t (`bp`, read through `memory_out`);
  - from the reader's mob probabilities on the previous real frame (63 x 4), fitted, and unfitted as a copy (the
    reference: what the last frame implies);
  - from both.
  Probe data: every DEV episode without units (764), 120 frames each, 80% / 20% by episode; held-out AP against the
  base rate. (Amended after the smoke, before the run: C3's 301 unit-free episodes and the 1,008 raw tokens as the
  reference gave a copy AP of 0.05, a data-starved per-cell fit rather than a reference.) Then on the units: the deletion rate by quartile of the belief probe's P(entity in the
  neighbourhood at t).
- **What separates:**
  - deletions concentrated where the belief probe is unsure: the conditioning is the limit (Mamba's side);
  - the probe confident where draws delete, and binomial spread: the denoiser's sampling of its own conditional (the
    user's "solely a denoiser sampling issue").
- **Predictions (before running):** over-dispersed, mostly night and edge; every step's estimate keeps the mob at
  first (C2's mean), and the draws part at steps 2-3; the belief probe well below the copy reference (P1-F1.6: the
  global projection loses spatial detail), but not marking the deleted units.
- **Not in C5:** no inference-time fix (record guidance, other K), and no training. Those wait for the cause.

## C6: is the persistence coin still being learned? (user 11-20: "proceed with those checks")

- **Question.** C5: a visible mob by day vanishes on 6% of draws (cows) while the data keep it. Is the learned
  conditional still converging on persistence, or is the coin flat over training?
- **What runs.** C5's draws and readings (all 2,079 units, 8 draws, the same seeds) at P1.5's checkpoints 2, 5 and 8
  (5.1M, 12.8M and 20.5M of 25.6M frames), beside C5's checkpoint 10. The Mamba probe is not rerun.
- **Reader.** The encoder trains jointly (relative weight change against checkpoint 10: 44% at 2, 7% at 5, 1.7% at 8),
  so each checkpoint gets its own S1 kind reader, fitted by S1's own protocol (`s1.encode` on train400,
  `cell_table`, `token_probe` seed 0). Checkpoint 10 keeps S1's saved reader, fitted by the same protocol. Each
  checkpoint's real-frame keep rate is its reader's floor.
- **Readings.** Keep rate by kind and light against that floor; day cows' dispersion; the ladder's step-1 AUC.
- **What separates.**
  - The day coin falls from 5 to 8 to 10 by more than its noise (about 0.3 points per checkpoint): still being
    learned, so training (length, capacity, data exposure) is the lever.
  - Flat from 5 on: built into the recipe or objective.
- **Prediction:** it falls early and flattens (most of P1.5's other skills did), so flat from 5 on.

## C7: does P1.5's belief track zombies? (user 11-20: "if it tracks zombies that's fine; those matter for the actor")

- **Already measured** (P1.5's evaluation, `20261116_p15/readings/belief_probe.json`): given a zombie next to the
  player, the predicted belief anticipates a hit at AP 0.952 (base rate 0.16; the percept and action alone 0.46).
  That measures the hidden cooldown, not whether the belief knows a zombie is there.
- **Targets** (DEV labels, frame t): a zombie in the 8 cells around the player; within 2 cells; anywhere on the screen;
  and per cell on the 24 cells within 2 (pooled).
- **Inputs:**
  - the corrected belief b_t (what the actor reads);
  - the predicted belief bp_t (the denoiser's memory);
  - as references, the S1 reader's mob probabilities on frame t (for b_t) and on frame t - 1 (for bp_t).
    Fitted by the same probe, and unfitted (the reader's own P(zombie)); the unfitted reference was added after the
    smoke, where 40 episodes starved the fitted per-cell probe.
- **Data and probe:** all DEV episodes (C3's order), 120 frames each, 80% / 20% by episode, `s1.encode`; C5's
  per-output logistic probe; held-out AP against the base rate.
- **Reading:** b_t near its reference means the actor's input knows where zombies are; far below means it does not.

## C8: an eye scored by the world's belief change (user 11-20: "agreed for testing it; would it be an accurate test? If yes, launch it")

- **Score.** For frame t, B_t per cell = || x_hat_{t+1}(real frame t) - x_hat_{t+1}(the world's own estimate of frame t) ||^2:
  how far the real frame moves the world's zero-signal prediction of the next frame, against its own expectation.
  - The expected frame replaces the real one everywhere it enters: the denoiser's context, the record's update and
    Mamba's percept (Mamba re-run on the prefix).
  - Both arms take NOOP at t (causal: the eye chooses before the action), share the real past, the monitor's
    glimpses and noise seeds, and the real lighting.
- **Data:** C2's 150 DEV episodes and picks (frames with t + 1 <= T), P1.5 checkpoint 10.
- **Beside it, on the same frames:** C2's world-eye surprise S_t (recomputed) and the chooser (C2's code).
- **Readings:** coverage under the chooser's policy with contingency off (as C2 and C4), by entity kind, changed HUD,
  entering terrain and night; B's share by region, day and night.
- **Check:** the re-run Mamba with the real action reproduces C2's predicted belief (max abs difference reported).
- **What makes B worth pursuing:**
  - entity coverage above S's (0.484), at night especially;
  - a lower night static share than S's 60%;
  - changed-HUD coverage at or below S's 0.224.
- **Accuracy limits:**
  - a single zero-signal estimate per frame (C2's), and the real lighting in both arms;
  - offline frames under the monitor's fixations;
  - it shows where the eye would look, not what looking there would teach the world.
- **Prediction:** night static falls well below 60%; entity coverage rises at night; HUD flips stay near S's.

## C9: the world-eye's cache, its cheaper route, and what a stale eye looks at (user 11-20: "how bad is the drift for the 5-pass ... Is it worse than using the chooser? Can we see what it would fixate"; "what's your cheaper route?")

- **Data:** C2's 3,000 DEV target frames (150 episodes, the same picks), P1.5 checkpoints 10, 9, 8 and 5.
- **Early training (added after the smoke):** checkpoint 2's score and estimate caches read at checkpoint 5 against
  checkpoint 5's fresh eye (3 passes stale, contingency off). The late checkpoints alone would hide early drift.
- **Early, per pass (`c9_early.py`, added after C9 found early drift):** a score cache one pass stale at checkpoints
  2-5 (checkpoint j - 1's own error read at j), contingency off and deployed with checkpoint j's own chooser head;
  beside j's fresh eye and j's chooser. It sets a refresh schedule from data.
- **Eyes, each scored per window and passed through the chooser's policy:**
  - **fresh:** checkpoint 10's zero-signal estimate against checkpoint 10's tokens (C2's world-eye);
  - **estimate cache, k passes stale:** checkpoint k's estimate against checkpoint 10's tokens (C2's staleness form);
  - **score cache, k passes stale:** checkpoint k's estimate against checkpoint k's own tokens. Each refresh stores
    the error itself (81 values per frame) instead of the estimate;
  - **drift only:** checkpoint 10's tokens against checkpoint k's tokens. It separates encoder drift from staleness;
  - **chooser-proxy belief (the cheaper route):** checkpoint 10's estimate with Mamba's belief built from the
    chooser's sampled fixations, instead of the monitor's prior fixations that C2's fresh eye used. The two
    fixation streams bracket how much the eye's own past looks move its estimate.
- **Policies:**
  - contingency off, for comparison with C2-C8;
  - the deployed form: the chooser's own contingency C (checkpoint 10's head) with the world's S.
  - The chooser itself is its own C + S.
- **Readings per eye:**
  - coverage (C2's table);
  - TV distance and argmax agreement to the fresh eye under the same policy;
  - **where its looks land:** the expected share of looked-at cells per region (mob, changed HUD, static HUD, entering
    terrain, terrain edit, static map), day and night, against uniform;
  - **confirmation:** the look mass on windows the current world already predicts well (the fresh score below its
    frame's median);
  - the estimate cache's TV to the drift-only eye. If it is near, a stale estimate cache chases encoder drift.
- **Figure:** six frames (three day, three night, each with a mob), with each eye's per-cell look probability and
  its most likely window: the chooser, the fresh eye, the 5-stale estimate cache and the 5-stale score cache
  (deployed form).
- **Cost of the cheaper route,** at the trainer's shapes on P1.5 checkpoint 10's training streams (warmed, 3 timed
  each):
  - the perception, record and belief pass (`c2_cost.inputs`);
  - two chunked denoiser forwards without gradients.
  - The exact route is a two-stream pass: the chunk at the context signal for keys and values, then each target at
    signal 0 reading them. Its code is not written, so two plain chunked forwards stand in for its cost.
  - Projected on P1.5's measured updates.
- **Accuracy limits:**
  - offline frames, and late-training checkpoints only (early training drifts faster);
  - it shows where each eye looks, not what looking there teaches the world;
  - the proxy comparison bounds the belief's dependence on the fixation stream in P1.5, where the glimpse barely
    reaches the periphery estimate. With P1.6's foveal record the dependence is larger by design, and this does not
    measure it.
- **Predictions:**
  - the 5-stale score cache is far closer to fresh than the estimate cache (0.25), which mostly chases drift;
  - the proxy-belief eye is within the 1-pass staleness (TV < 0.04);
  - the cheaper route costs under a third of the fresh in-loop eye (+12.2 h).

## C10: history guidance at the persistence coin (user 11-20: "For the history guidance ... Do you want to run it now, or does it take a long time?")

- **Source: DFoT** (Song et al., ICML 2025, arXiv 2502.06764, read 11-20).
  - It is classifier-free guidance where the condition is the history: score = uncond + w (cond - uncond).
  - "Unconditional" is the same model with the history frames masked by full noise. Training needs no dropout,
    because diffusion forcing noises every frame independently; DFoT's binary-dropout baseline is worse.
  - Vanilla guidance is best at w = 1.5 on Kinetics (FVD 208 -> 182). It improves consistency at a cost in diversity,
    and at w >= 3 it can "generate static videos" by copying the last history frame.
- **Ours.** Shortcut forcing noises every frame independently too, so masking our 8 context frames by noise is in
  distribution frame by frame. But our past also enters through the record and Mamba's belief, which training never
  masks except at an episode's first frame. Two unconditional forms:
  - **frames:** the 8 context frames at signal 0, fresh noise; the record and belief kept (DFoT's masking);
  - **all:** no context frames, with the record and belief at their episode-start values (zero), as at an episode's
    first frame; the action is kept in both terms, so the guidance concerns the history only.
- **Applied** at every one of the sampler's four steps of the physical stage: x_hat = x_u + w (x_c - x_u), then the
  sampler's clip and move. The glimpse stage is untouched; the physical tokens never read it.
- **Units and draws:** C5's 2,079 start mobs and 8 draws, with C5's noise seeds. The unguided arm must reproduce C5's
  keep rates exactly.
- **Arms (4):**
  - unguided (C5);
  - frames at w = 1.5, DFoT's best;
  - frames at w = 3, its static-video end;
  - all at w = 1.5.
- **Readings:**
  1. keep rate by kind, day / night, and at night by edge ring (the mob's cell on the map's outer ring) vs inner;
  2. **staying put:** among kept draws, the share whose mob stays at its own cell, against the real frame's (DFoT's
     static-video failure);
  3. **spurious mobs** per frame: map cells the reader calls a mob in the draw and none in the real frame;
  4. token error against the real frame by region (C1's groups), and the spread over draws by region (diversity).
- **What makes it a candidate:** day cow deletion well below 6% and the night edge clearly lower, with staying put
  within a few points of the real frame and no rise in spurious mobs or in the entering-terrain error.
- **Accuracy limits:**
  - one-step draws from a perfect history: not imagination's compounding, which S2 measures;
  - w is not tuned beyond the two published points;
  - "all" pairs an episode-start context with a mid-episode action, a combination training never saw;
  - a few-step shortcut sampler, where guidance of the average velocity is a heuristic; DFoT used 50 DDIM steps.
- **Cost if adopted:** a second denoiser evaluation per sampler step in imagination (evaluation and the later actor),
  none in training.
- **Prediction:** "frames" at 1.5 halves the day coin; "all" does more but costs fidelity; w = 3 raises staying put.

## C11: which refresh schedule (user 11-20: "I meant using the world-eye throughout the entire run, but letting it stay stale longer during the beginning ... roughly equivalent to the chooser at those stages. Is that not the case? If you don't know, can we test this first?")

- **Map:** C2's 3,000 DEV frames at all ten P1.5 checkpoints. For every pair i < j, checkpoint i's score cache (its
  own error) is read at checkpoint j in the deployed form, with checkpoint j's chooser head supplying C. Beside each
  j: the fresh world-eye and the chooser.
- **Schedules,** scored from the map:
  - Pass p is read at checkpoint p, the end of the pass: the cache's oldest point, so the reading is conservative.
  - Pass 1 always uses the chooser. A cache filled at initialisation is uninformative, because the denoiser's output
    layers are zero-initialised and the "error" is the size of a random encoder's tokens.
  - A refresh after pass r serves the passes from r + 1 until the next refresh.
  - Evaluated: every one- and two-refresh schedule, the chooser throughout, and the fresh eye throughout (the upper
    bound).
- **Choice rule (declared before the run):** the two-refresh schedule with the highest mean entity coverage over the
  ten passes, whose mean zombie coverage is no more than 0.01 below the chooser throughout. Changed-HUD coverage is
  reported beside it.
- **Readings:**
  - the map's entity, zombie and changed-HUD coverage per (i, j);
  - the per-pass curve of the chosen schedule and of an early-stale schedule (refresh after pass 1, kept for
    several passes);
  - the mean over passes for every schedule.
- **Accuracy limits:**
  - P1.5's checkpoints, whose world was trained with the chooser's looks. P1.6's world (G1, and the world-eye's own
    looks) will drift differently;
  - 3,000 frames: coverage differences of about 1 point are within noise (paired frames; entity frames n = 2,012);
  - coverage is where the eye looks, not what the world learns from it.

## C11b: does the early chooser beat uniform? (user 11-20: "maybe starting uniform for the first 2k updates if that's better than the current chooser and the stale world-eye")

- **Data:** P1.5's own gaze trace (`gaze.bin`): every look the chooser made in training, with the paired uniform
  window that the chunk held before the chooser replaced it. Restricted to the 400 labelled training episodes
  (`s0_train400_labels`, `20261104_p1f2_review/trace.py`'s mapping and masks).
- **Reading:** coverage when present (any entity, zombies, cows, changed HUD, entering terrain) of the chosen window
  against the paired uniform window. Bins of 250 updates over the first 2,500 updates (pass 1), then one bin per pass.
- **What it answers:** whether the chooser's looks while it is still learning are better or worse than uniform, by
  where they land. It does not measure what the world learns from either.

## C12: is the day coin a hedge over where a mob goes next? (user 11-20: "We don't have a fix for the sampling issue? Why not? Did we dig enough? Do we know what the cause is?")

- **The candidate.** Craftax-Classic moves a cow each step to a random choice of 8 directions: 4 are moves, 4 are
  "stay", and a move into a blocked cell becomes a stay (`craftax_classic/game_logic.py`, `_move_cow`;
  `is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava`).
  - A cow with every neighbour free has its next cell spread over five cells: stay 0.5, each neighbour 0.125. A cow
    with no free neighbour stays with certainty.
  - If the world's conditional hedges over these cells, a sample can fail to commit any of them, and the cow vanishes.
  - This would fit three earlier observations: C8 (the world's next position for a mob is hedged), C10 (whole-history
    guidance cut deletions while cows moved less), and the kinds' order (cows and skeletons, random walkers, are
    deleted more than zombies, which head toward a nearby player).
- **The competing explanation.** A uniform under-confidence about persistence, from the objective or the learning-rate
  schedule (C6 unresolved), predicts the same coin whatever the mob's freedom to move.
- **Test (CPU, C5's draws, no new sampling).**
  - C5's 2,079 units and their 8 draws (`c5_units.npz`), rebuilt in C5's order and asserted equal.
  - Per unit: the number of free 4-neighbours of the mob's cell at t0, by Craftax's rule on the labelled view and
    entity grid: in view, not a solid block (water, stone, tree, coal, iron, diamond, table, furnace, plant, ripe plant),
    not lava or out of bounds, no entity, not the player's cell.
  - Deletion rate (1 - keep over draws) by kind, day / night and free count 0-4, with unit-level standard errors.
  - A bootstrap over units of day cows' deletion at 4 free against at most 1 free.
- **Reading.**
  - The hedge is supported if day cows' deletion rises with free neighbours (ratio of 4 free to at most 1 free of at
    least 2, bootstrap interval above 1) and nears the real floor when boxed in.
  - Uniform under-confidence is supported if the rate is flat (ratio interval covering 1).
- **Accuracy limits.**
  - Free count correlates with terrain (open grass against forest edges) and with other mobs nearby.
  - Mob positions at t0 approximate the simulator's sequential update (zombies move before cows).
  - Units are inner cells (rows 1-5, columns 1-7), so every neighbour is in view.
