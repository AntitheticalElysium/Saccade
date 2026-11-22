# Where does a look help the next decision? (2026-11-23)

**Why.** The user: the eye is one of the architecture's main points, and its looks must be targeted in life and in
dreams. The agent's look tokens need a training signal from the start, and there are no gaze demonstrations. Codex
proposed the first signal: which crop improves the next decision, measured on real frames. This EDA asks whether
Craftax's corpus contains that signal and where. It informs the look's training signal. It doesn't select an
architecture.

**Known before this (the 11-22 fovea digest):**
- coarse 3x3 tokens hold every HUD value (and, 11-23, the health digit on all 43,115 DEV frames);
- the best look in hindsight saves 0.25-0.6% of the next-frame loss;
- night creatures are the fovea's one real information gain for the world.

**The decision value of a look is unmeasured.**

**Claim.** The full-resolution glimpse at some windows improves predicting the demonstrator's next action beyond the
coarse view and memory, and the gain concentrates in identifiable situations.

**Competing explanation.** Apparent gains are the probe reading coarse information better through the crop pathway,
not detail the coarse view lacks.

**Separating control.** The same window rendered from the coarse view: each cell pooled to the periphery's 3x3
acuity, upsampled back to 7x7, and passed through the same fovea encoder. The value of acuity = real - blurred, at the
same window and with the same encoder.

## Setup

- Frozen P1.5 checkpoint 10. The split of the dream-commitment checks: 128 train400 episodes to fit, 160 disjoint DEV
  episodes to score.
- **Decisions:** frames t >= 8 with an expert action (`t_exogenous` false).
- **Target:** the demonstrator's action at t (17 classes).
- **Inputs:**
  - the predicted belief before frame t, the periphery and lighting tokens of frame t, the record read after them,
    and the previous action;
  - a crop block: the 9 glimpse tokens, the window (one-hot over the 49 centres), and a condition flag.
- **One MLP** (512, 256, GELU; Codex's probe family, wider) is trained with conditions mixed: real crop 50%, blurred
  crop 25%, no crop 25% (block zero after normalisation). Windows are uniform. 16 epochs, AdamW 7e-4, cross-entropy.
- **Scoring:** on DEV, the log-probability of the recorded action with no crop, with the real crop at each of the 49
  windows, and with the blurred crop at each.
- **The old chooser:** P1.5's checkpoint fixation controller, run on the real frames with its 8-frame history (as in
  `20261119_s2/s2b.py`), gives P(window) per decision.

## Readings (fixed before results)

1. **Per decision, in nats:**
   - the mean over uniform windows of real minus none, and of real minus blurred (the value of acuity);
   - the hindsight-best window, real and blurred (the blurred maximum carries the same selection bias);
   - the change in top-1 accuracy;
   - the scale: the no-crop cross-entropy.
2. **By situation:**
   - night (light < 0.5) against day;
   - a creature in view against none;
   - windows covering a creature, the faced cell, or the HUD, against the others.
3. **The old chooser:** the expectation of 1's quantities under its window distribution, against uniform.

Episode-bootstrap 95% intervals throughout.

## Reading it

- **A value of acuity near zero everywhere:** at this acuity the look has little decision work in Craftax. Its job is
  then the world's (night creatures) and economy, and the look's signal must come from elsewhere.
- **Positive, concentrated values:** they show where a value-of-information signal points, and whether the old eye
  looked there.

## Limits

- A probe is a lower bound on information (the 11-23 lesson).
- One checkpoint.
- The demonstrator's decisions, not the agent's.
- An expert-mixed corpus.

## Run

`JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_gaze_decision_value.decision_value [--smoke]`
GPU, one job.
