# Where do surprise and contingency send the eye, by day and by night? (2026-11-23)

**Why.** The eye's learning design (decision 3) asks what drive the look should start from. The chooser's rule is
0.8 softmax(0.5 C + 0.5 S) + 0.2 / 49, with S its own forecast error and C the taken-minus-NOOP forecast difference.
Raw surprise is known to chase irreducible noise (Burda et al. 2018). Craftax has a known noise source: entity-free
map cells change 0.101 per cell at night against 0.0009 by day (the P1-F6 dissection). This EDA measures, on real
frames, whether the chooser's surprise goes to that static at night, and where contingency points. It is about the
data, not a selection of architecture.

**Claim.** At night, S puts more of its mass on entity-free map windows than by day, at the expense of entity windows.

**Competing explanation.** Night simply has more entities (zombies spawn at night), so a shift in where S sits could
come from entity counts. **Separating reading:** the share of S mass per window category, normalised by the
category's share of windows on the same frames (a lift over uniform), split by day and night.

## Setup

- Frozen P1.5 checkpoint 10 and its fixation controller head.
- The same 160 DEV episodes (the dream-commitment split), all frames t >= 8.
- The chooser on real frames exactly as `20261119_s2/s2b.py` and `20261123_gaze_decision_value/decision_value.py`
  run it.
- **Window categories per frame:**
  - entity: covers a creature (`gates.episode_grids`);
  - HUD: covers row 7 or 8;
  - static map: map-only with no creature.
- Night: light < 0.5.

## Readings (fixed before results)

1. For S, C and P(window): the mean share of mass on each category, and the lift (share / the category's window
   share), by day and night, with episode-bootstrap 95% intervals.
2. The chooser's P on entity windows when an entity is in view, by day and night, against uniform.

## Run

`JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_eye_drives.eye_drives`
