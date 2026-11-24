# What a dreamed step commits to: the two supporting checks (2026-11-22)

**Agreed with the user (decision 2, step 1):** a dreamed step commits to the previous action's outcome (reward,
termination) before the eye looks. The order:
- the coarse view and lighting are drawn;
- the outcome is drawn from what is committed so far;
- the look is chosen;
- the glimpse is drawn.

The user asked for the two supporting checks. The open design question: does the glimpse stage need the committed
outcome as an input?

## Why the question is architectural, not about P1.5's quality

In a perfect world model both orders are valid factorisations of the same joint:
- outcome then glimpse: p(coarse) p(outcome | coarse) p(glimpse | coarse, outcome, look);
- glimpse then outcome, today's P1.5: p(coarse) p(glimpse | coarse, look) p(outcome | coarse, glimpse, look).

What differs is which property holds *by construction*, whatever the model's quality:
- **Committing first** makes the outcome independent of the look exactly. It also makes the glimpse agree with the
  outcome exactly, but only if the glimpse is conditioned on the outcome.
- **Today's order** makes the glimpse agree with the outcome exactly. The outcome is independent of the look only as
  far as the model's conditionals cohere (Codex measured 0.12 pp on average, 5 pp at one branch).

An eye trained on dream returns is an optimiser, and it will find any incoherence. So the properties should hold by
construction.

If the glimpse is drawn *without* the outcome, a perfect world contradicts the committed outcome whenever the
committed state leaves the outcome uncertain. With p = P(event | committed state), the outcome and the crop are two
independent draws of the same event, so the chance they disagree is 2p(1 - p). Check 1 measures this on real frames,
which makes it independent of P1.5's generation quality. Check 2 measures what P1.5's dreams do today.

## Setup (shared)

- Frozen P1.5 checkpoint 10.
- Codex's matched probes (`20261119_s2/dream_outcome_audit.py`), refitted with its exact settings:
  - fitted on its 128 train400 episodes;
  - scored on its 160 disjoint DEV episodes.
- **Before-crop features:** the predicted belief, the previous action, the periphery and lighting tokens, and the
  record read after them.
- **After-crop features** add the crop tokens and the fixation.
- **Event:** a health drop, i.e. reward < -0.025. Craftax-Classic's reward is achievements + 0.1 x the health change
  (game_logic.py), so a negative reward is a drop with no unlock.
- Termination is reported beside it where counts allow.
- **The look that reveals the event:** the window centred at (7, 1). Rows 6-8 and columns 0-2 include health, which is
  cell 63. Codex's HUD look (7, 4) did not include it.

## Check 1: how much of the outcome the committed state already holds (real frames)

- **1a, verification.** The refitted before-crop probe reproduces Codex's DEV average precision (0.9749 / 0.9665 /
  0.9999) within 0.003. Otherwise the refit is reported and used as is.
- **1b, calibration.** Observed against predicted rate in bins of the before-crop probability (0-0.01, 0.01-0.05,
  0.05-0.2, 0.2-0.5, 0.5-0.8, 0.8-0.95, 0.95-0.99, 0.99-1). This shows whether its p can be read as a probability.
- **1c, the architecture reading.** Over all DEV transitions with a full context (frame >= 8), and over those with a
  recorded health drop:
  - the expected contradiction 2p(1 - p);
  - the share of committed drops an unconditioned perfect glimpse would contradict: E[p(1 - p)] / E[p];
  - the same quantities under the after-crop probe beside them (what the look resolves).

## Check 2: what P1.5's dreams do today (generated successors)

- **Starts:** from the 160 DEV episodes, every transition with frame >= 8 and a recorded health drop, plus as many
  random other transitions (seed 20261122).
- **Draws:** four one-step dreams per start, branching after the real frame with the recorded action, through
  `M.sample`:
  - real belief, record and 8-frame context cache;
  - the coarse view generated first;
  - `choose` sets the look to (7, 1);
  - the glimpse is generated there.
- **On each draw:**
  - the before-crop probe on the generated coarse view (the committed p);
  - the after-crop probe and the native reactive head with the generated crop (today's crop-reading design, as a
    reference);
  - the coarse HUD reader (S2's) health digit;
  - the fine reader's health digit.
- **The fine reader:** P1.5's own pixel decoder (`tsp.reconstruct`, `pixels`) on the health cell's glimpse token,
  nearest exact renderer patch (`hud_atlas.npy`, health, 0-9).
- **The drop** is the read digit below the real health of the frame branched from.
- **Readings:**
  - **2a, instrument.**
    - Fine reader on real glimpse tokens at (7, 1) of the same start frames, against the true health label: floor
      >= 0.98 exact. Below it, the fine-crop readings are reported as reader-limited.
    - Coarse reader on real frames, for its floor.
    - The record state at the branch (`S.window` index 8) equals `S.history`'s last state, to within fp16.
  - **2b, outcome on generated coarse views.** Mean committed p per cohort (recorded drop or not), on the generated
    coarse view against the same probe on the real successor; and the average precision of the dreamed p against the
    recorded outcome.
  - **2c, contradictions.** Per cohort, the expected disagreement with the committed outcome, E[p (1 - d) + (1 - p) d]
    with d = 1 if the read digit shows a drop:
    - of the fine crop (the eye's view);
    - of the coarse view;
    - and the coarse-fine disagreement.
    - The share of committed drops the fine crop contradicts, E[p (1 - d)] / E[p], beside check 1c's ideal value on
      the same starts.
    - The native crop-reading head's disagreement with the same crop, as today's reference.

## Decision rule (fixed before results)

- **The architectural question comes from check 1c.**
  - If an unconditioned perfect glimpse would contradict more than 5% of committed health drops, the glimpse stage
    takes the committed outcome as an input.
  - Below 1%, it doesn't need to.
  - In between is marginal, and the user decides.
- **Check 2 is today's world, not a design driver** (user 11-22: building a good enough world is a separate task). It
  says where P1.5 stands and where a contradiction would arise:
  - if the coarse view already disagrees with the committed outcome, the coarse successor and the outcome readout
    must agree first (Codex), and conditioning the crop would not fix it;
  - if only the fine crop disagrees, conditioning it is the repair.

## Limits

- One checkpoint and one event (health drops), seen through one look.
- Positive rewards (unlocks) show up mostly as inventory changes outside this window and are not checked.
- The probes are a lower bound on what the committed state holds, so check 1c's p(1 - p) is an upper estimate for a
  better reader of the same state.
- Generated tokens are read off their training domain (TEST_SPEC).

## Run

`JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261122_dream_commitment.commitment_checks [--smoke]`
GPU, one job.
