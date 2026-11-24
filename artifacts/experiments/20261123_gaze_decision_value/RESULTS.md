# Where does a look help the next decision? Results (2026-11-23)

Plan and code: `PLAN.md`, `decision_value.py` (commits 20f4a70c, with the instrument repair in 574d8f57). Outputs:
`summary.json`, `scores.npz`.
- Frozen P1.5 checkpoint 10.
- 27,975 expert decisions in 118 DEV episodes. The other 42 of the 160 are pure random play (epsilon 1.0).
- The probe was fitted on 112 train400 episodes, with 16 more held out for validation.

## The instrument is weak

The validation cross-entropy was lowest after the first epoch (1.80) and rose every epoch after, to 5.43, while the
training loss fell to 0.08. The kept probe is that one-epoch fit:
- 1.876 nats on DEV with no crop, against the actions' marginal entropy of 2.328;
- top-1 accuracy 37.6%, against a majority share of 21.0%.

A probe this weak extracts little from 576 crop numbers. Everything below is a lower bound on what the crop carries
for the demonstrator's choice.

## Readings (as declared)

| Per decision (nats) | Mean | 95% interval |
|---|---:|---:|
| Real crop minus none, uniform window | -0.0153 | [-0.0167, -0.0138] |
| Blurred crop minus none, uniform window | -0.0130 | [-0.0145, -0.0114] |
| **Value of acuity (real minus blurred), uniform window** | **-0.0023** | [-0.0034, -0.0012] |
| Value of acuity under the old chooser's looks | -0.0024 | [-0.0038, -0.0009] |
| Hindsight-best window, real / blurred | 0.331 / 0.242 | [0.325, 0.337] / [0.238, 0.246] |

- **Adding any crop slightly hurts the probe.** It treats crop features as noise.
- **Most of the hindsight-best gain is selection.** The blurred crop holds no detail beyond the coarse view, yet its
  best window still "gains" 0.24 nats.
- **By situation, the value of acuity:**
  - night +0.0004 [-0.0015, 0.0024]; day -0.0038;
  - with a creature in view -0.0020; with none -0.0030.
- **By window:** the faced window +0.0009 against -0.0030 elsewhere; creature windows -0.0008; HUD windows -0.0071.
- **Where the acuity-best window lands:** on a HUD window in 78% of decisions (29% of windows). That fits noise, since
  digit crops vary most, rather than information.
- **The old chooser** put 31% of its mass on creature windows (15% by chance) and 21% on the faced window (18%). Its
  looks gain no decision value under this probe.

## What it means

- **With this instrument there is no detectable decision value of full resolution at any window, day or night.**
  - This agrees with three earlier, independent measurements: the coarse 3x3 tokens hold every HUD value; they hold
    the health digit exactly (43,115 / 43,115 DEV frames); and the best look in hindsight saves 0.25-0.6% of the
    next-frame loss.
  - Night creatures remain the one known gain for the world's prediction. Under this probe it doesn't reach the
    demonstrator's choice.
- **It does not establish that the crop carries no decision information.** The probe peaks after one epoch and
  extracts 0.45 nats about the action from all its inputs.
- **The direct test is the full system's own read:** the agent tokens, learning to act from the glimpse they chose.
  If decision value stays near zero there, the look's signal in Craftax at this acuity will come from the world's
  surprise, not the decision.
- **The implication for the eye.** In Craftax-Classic at 3x3 periphery acuity, almost everything a decision or a
  prediction needs is already in the coarse view. That leaves the eye three jobs, none of them information gathering:
  - the world's night creatures;
  - economy: in dreams, only the looked-at crop is generated;
  - being ready for worlds where detail matters.

  This bears on acuity (decision 11) and on which environment can show what the eye is for. It is not a reason to
  lower acuity just to make the fovea useful (the user, 11-21).
