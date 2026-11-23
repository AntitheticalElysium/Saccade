# What does a look add to the agent's knowledge? Two definitions of the eye's information drive (2026-11-23)

**Why.** Decision 3 gives the look two drives: relevance through the shared advantage, and information. The user asked
how the information drive should be measured.

**What is measured so far** (`20261123_eye_drives/RESULTS.md`):
- raw pixel surprise, the old chooser's and the world's own foveal likelihood alike, can't separate night creatures
  from night static;
- at a fixed physical state its best window flips with sensor noise in all eight dark scenes (Codex's static
  control).

**Two candidate definitions, each computable for the one crop acquired, with no full-frame teacher:**
- **Belief change** (Itti & Baldi's surprise as belief change): how far the crop moves the corrected belief,
  ||b_t(real crop) - b_t(the same window at periphery acuity)||.
- **Predictive gain** (Codex): how much the crop improves the prediction of the next real evidence at the same world
  place. That is the trans-saccadic NLL of the next frame's real crop at the window shifted by the player's move,
  from the predicted belief, with the periphery-acuity crop minus with the real crop. Sensor noise in this crop can't
  help predict the next one.

**Claim.** Predictive gain is larger for creature windows than for entity-free map windows at night. Belief change may
not be, if the belief reacts to noise.

**Competing explanation.** P1.5's belief carries almost nothing of the glimpse forward. We measured that F1.6's
global 256-D projection loses most of the fine evidence. Then both definitions read near zero everywhere, and the
result says the information drive depends on the memory keeping what was seen, not on the definition.

**Separating reading.** The absolute size of both, against the same crop's immediate foveal surprise at t, which is
large.

## Setup

- Frozen P1.5 checkpoint 10. The 160 DEV episodes of the dream-commitment split.
- **Starts:** up to 6 night and 6 day frames per episode (t >= 8, t + 1 in the episode), seed 20261153.
- **Windows per start:** one covering a creature (if any), two entity-free map windows, one HUD window.
- **For each window, two variants of frame t's glimpse:** the real crop, and the same window rendered at the
  periphery's acuity (each cell pooled to 3 x 3 and upsampled).
- **The belief:** Mamba is run over the real prefix 0..t-1 (as `20261119_s2/s2b.py`), then one step on frame t with
  each variant and the recorded action.
- **Predictive gain:** TSP NLL of frame t + 1's real crop at the shifted window (HUD windows unshifted; skipped if
  the shift leaves the grid), from the variant's predicted belief and frame t + 1's real periphery and lighting.
- **Reference:** the same window's immediate foveal surprise at t (TSP NLL from the encoded predicted belief).

## Readings (fixed before results)

1. Mean predictive gain (nats per RGB component) by category (creature, entity-free map, HUD) and day / night. The
   creature-minus-static difference is paired within starts.
2. Mean belief change, absolute and relative to ||b_t||, by the same cells.
3. The immediate foveal surprise on the same windows, for scale.

Episode-bootstrap 95% intervals.

## Run

`JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_look_information.look_information`

## Third reading: belief change on the 11-20 eye's failure classes (added 11-23, before running)

On 11-20 a different "belief change" failed (the switching dissection's belief-change eye). It measured the shift in the world's
zero-signal prediction of frame t + 1 when given the real frame t instead of its own estimate. Entering terrain took 43%
of its score, static 49% at night, and mobs 10%, because a mob's next position is hedged. Today's definition (fine
minus periphery-acuity crop, the current corrected belief) was read only on creature, static and HUD windows, so it
has not met the 11-20 eye's failure classes.

- **Claim:** today's belief change ranks creature windows above entering-terrain windows, and changed-HUD windows
  above unchanged ones.
- **Competing explanation:** like the 11-20 eye, it is drawn to newly entering terrain.
- **Setup:** the same split and model. Starts are frames where the player moved or a HUD value changed (up to 8 per
  episode, seed 20261163).
- **Windows per start, at most one per category:**
  - creature;
  - entering terrain: map-only, covering the edge cells that just entered view, with no creature;
  - static map: map-only, no creature, not entering;
  - changed HUD: covers a HUD cell whose value changed;
  - unchanged HUD.
- **Reading:** mean belief change (absolute and relative) per category, day and night, with episode-bootstrap 95%
  intervals.
