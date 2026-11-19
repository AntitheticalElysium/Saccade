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
