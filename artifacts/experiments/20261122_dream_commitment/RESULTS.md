# What a dreamed step commits to: results (2026-11-22)

Plan and code: `PLAN.md`, `commitment_checks.py` (commit 49c172c4). Outputs: `commitment_checks.json`, `draws.npz`, and
the log `run.log`.
- Frozen P1.5 checkpoint 10.
- Probes fitted on 39,039 transitions from 128 train400 episodes and scored on 42,955 transitions from 160 disjoint
  DEV episodes (Codex's split).
- Event: a health drop (reward < -0.025). The look that reveals it: the window centred at (7, 1).

## Instrument

- **Codex's probes reproduce to the fourth decimal.**
  - Before-crop average precision: 0.9749 / 0.9665 / 0.9999 (positive reward, health drop, termination).
  - After-crop: 0.9743 / 0.9609 / 1.0000.
  - Brier scores also match.
- **The fine reader** (P1.5's pixel decoder on the health cell's glimpse token, nearest exact renderer patch) reads
  the real health digit exactly on all 1,846 real start frames: 1.000.
- **The coarse reader** (S2's, periphery tokens) is exact on 93.8% of them. It misses 9.9% of real drops.
- **The record state at the branch** equals the history's last state (max difference 6e-5, fp16).

## Check 1: how much of the outcome the committed state holds (real frames)

- **Calibration of the before-crop probe on drops.** It is overconfident at both ends:

| Predicted p | 0-0.01 | 0.01-0.05 | 0.05-0.2 | 0.2-0.5 | 0.5-0.8 | 0.8-0.95 | 0.95-0.99 | 0.99-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transitions | 41,739 | 158 | 89 | 63 | 39 | 58 | 57 | 752 |
| Mean predicted | 0.0001 | 0.023 | 0.101 | 0.355 | 0.653 | 0.888 | 0.974 | 0.9995 |
| Observed | 0.0007 | 0.082 | 0.169 | 0.302 | 0.462 | 0.707 | 0.895 | 0.981 |

- **Perfect world, glimpse drawn without the committed outcome** (declared reading 1c, all 41,835 full-context DEV
  transitions):

| Event | Expected contradiction 2p(1 - p) | Share of committed events contradicted, before-crop probe | After-crop probe | Binned recalibration (upper bound) |
|---|---:|---:|---:|---:|
| Health drop | 0.21% | **4.7%** | 4.4% | 11.5% |
| Positive reward | 0.21% | 3.1% | 3.3% | 11.3% |
| Termination | 0.012% | 1.6% | 1.5% | 0.7% |

- Within transitions that really had a drop (923), the share is 1.9%. The rest of the 4.7% is committed damage where
  none happened.
- The binned column replaces each probability by its bin's observed rate. Because p(1 - p) is concave, that is an
  upper bound for this information. The raw probe's overconfidence makes its own column an underestimate. A better
  reader of the same state would lower both.

**Declared reading:** 4.7% of committed health drops lies in the marginal band (1-5%), which the plan leaves to the
user.

**Corrected 11-23: the committed coarse view determines damage exactly.** My first reading said the coarse state
does not, citing the probe's AP of 0.966 and the coarse reader's 9.9% miss. Both are readers, and they read the coarse
view imperfectly.

`coarse_health_codebook.py` reads the health cell's periphery token by nearest exact renderer glyph, through P1.5's
frozen encoder with the frame's own lighting (the method of `20261104_learned_fine_candidates/hud_codebook.py`). On
the same 160 DEV episodes:
- exact on all 43,115 frames;
- all 962 true drops read;
- no false drops.

This agrees with the 11-04 acuity finding that coarse tokens carry every HUD value. So:
- the 4.7% / 11.5% "perfect world" figures measure the probe's reading limits, not information missing from the
  committed state;
- for Craftax damage, a world that draws the glimpse given the committed coarse view has no irreducible
  contradiction;
- what remains is model error: the outcome readout misreading the coarse view, or the generator disagreeing with it.

## Check 2: what P1.5's one-step dreams do today (1,846 starts, half with a recorded drop, 4 draws each)

| Cohort | Committed p, generated coarse view | Same probe, real successor | Fine crop vs committed, expected disagreement | Committed drops the fine crop contradicts | Committed drops the coarse view contradicts | Coarse-fine disagreement, dream / real |
|---|---:|---:|---:|---:|---:|---:|
| Recorded drop (923) | 0.653 | 0.916 | 3.6% | 3.1% | 13.9% | 7.4% / 9.9% |
| No recorded drop (923) | 0.0067 | 0.0018 | 0.53% | 43% of a small mass | 43% | 0.08% / 0.00% |
| Balanced panel | 0.330 | 0.459 | 2.1% | 3.6% [2.7, 4.6] | 14.2% | 3.7% / 4.9% |

- **The panel row is not a deployment rate** (corrected 11-23, Codex; recomputed from `draws.npz`).
  - The panel is half drops by design.
  - Reweighted to the event's frequency in full-context DEV (2.2%), the fine crop contradicts **15.7%** of committed
    drops.
  - Five of the 1,846 starts carry 82% of that mass: quiet starts where the dream commits some damage and the crop
    shows none. So the estimate is unstable, and neither 3.6% nor 15.7% should be read as P1.5's rate.
- The bracket is a start-level bootstrap.
- On the same panel, the perfect-world value under the raw probe is 1.9%.
- **The dreamed p tracks the recorded outcome with AP 0.950**, against 0.998 for the probe on the real successor.
  - In the drop cohort the dream commits less damage (0.65 against 0.92). The real successor already shows the
    realised drop, while a dream estimates the chance of it, so this mixes legitimate uncertainty with error.
  - Matched simulator forks would separate them. In the self-feeding evaluation, first-step damage beside a zombie
    matched the simulator (0.170 / 0.169).
- **Coarse view vs committed outcome:** 13.9% of committed drops, against the coarse reader's own 9.9% miss rate on
  real drops. It is mostly reader-limited, so it can't show whether the generated coarse view disagrees with the
  outcome.
- **Today's crop-reading head agrees with the generated crop** (0.5% disagreement) because it reads it. That is the
  same design whose outcome changes with the look (Codex, 0.12 pp on average, 5 pp at one branch).

## What it means

- **Committing before the look is supported.** The committed state carries most of the outcome: AP 0.966 for damage,
  0.975 for rewards, 0.9999 for termination.
- **The glimpse needs nothing new (user, 11-23, after the codebook check).**
  - The glimpse is already drawn given the committed coarse view, and that view holds the damage outcome exactly.
  - Contradictions in P1.5's dreams are model error, which training and a better world reduce.
  - Conditioning would be needed only where a committed outcome depends on detail the committed state can't carry
    and the eye can reveal. That is measured absent for Craftax damage, and is a trigger to measure for other senses.
  - Two earlier points still hold:
    - **not the task's score** (Codex): of 962 DEV health drops, 39 coincide with an unlock and have nonnegative
      reward, and a score changes with the goal while the scene doesn't;
    - agreement among the committed event, the coarse view and the fine evidence is the judge (Codex).
- **P1.5 doesn't block building the order.** The first loop keeps the look fixed, so nothing yet trains the eye on
  dreams. Its contradiction rate is unresolved (above). The codebook makes an exact coarse-view reading available to
  localise it.

**Not covered:**
- the 39 health drops that coincide with an unlock (the event is reward < -0.025);
- unlocks, which show outside this window;
- termination's visible evidence;
- multi-step looks;
- matched simulator forks for the dream's committed rates.
