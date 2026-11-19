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
