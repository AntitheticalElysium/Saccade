# Decision 6's place memory, the world's side, tested whole: results (2026-11-24)

**Verdict: not working yet. The chain breaks at localization.**
- **Reading 2 fails.** The best localizer finds the right entry on 0.438 of revisit lookups, against a bar of 0.95.
- **The cause is measured.** P1.5's coarse tokens change with daylight by more than the match's repeatability.
  - Two daylight sightings of the same world cell match 0.94-1.00 of the time when the light hasn't changed, at
    every gap (65 frames and more included) and every screen shift.
  - They match 0.56-0.85 after a change of 0.02-0.1, and 0.00-0.14 after a change of 0.1 or more.
  - So a place seen again after the light has moved isn't recognized. In the filter, the pose an old entry implies
    never beats the current one (0 of 920 steps), the map fragments, and on the diagnosis episodes 32% of lookups
    find their cell stored six or more cells from where the pose says it is.
  - In the dark the view itself is weaker: moved steps are right 0.688 after the frame, against 0.968 by day.
- **Readings 1, 3 and 4** run with the fifth localizer, as the PLAN's rule requires. They show what the read does with
  the entries this localizer puts in reach. The fine-tune ends about 10:30, and the readings follow. They are added
  here when they land.

Everything is one seed, as the PLAN declares.

## Reading 2: the localizer, five versions

Panel: 120 DEV episodes, 23,782 movement commands, 85,863 revisit lookups. Simulator positions score only. The bar is
correct lookups of at least 0.95. Each version's code is pinned beside its diagnosis (`diagnosis_v1/` to
`diagnosis_v5/`), and each amendment was written in the PLAN before that version's results.

| Version (what changed) | Prior, moved / blocked / dark | Posterior, moved / blocked / dark | Exact pose over 16 steps | Relocalisations (pending aligned) | Lookups | Lookups, episodes 23-119 |
|---|---|---|---|---|---|---|
| 1: robust Laplace match against a broad background | 0.719 / 0.922 / 0.519 | 0.696 / 0.971 / 0.491 | 0.492 | not scored | 0.394 | not scored |
| 1, odometry only | 0.651 / 0.917 / 0.490 | 0.613 / 0.970 / 0.457 | 0.373 | not scored | 0.217 | not scored |
| 2: spatial-prior null, per-observation sensor terms | 0.771 / 0.801 / 0.630 | 0.757 / 0.970 / 0.562 | 0.532 | 961 (0.390) | 0.434 | not scored |
| 3: plus a sensor-fixed null component | 0.764 / 0.788 / 0.582 | 0.750 / 0.855 / 0.584 | 0.435 | 3,829 (0.001) | 0.345 | 0.353 |
| 4: sensor-fixed scored by memory's own function; content trust | 0.899 / 0.133 / 0.625 | 0.887 / 0.276 / 0.643 | 0.037 | 7,083 (0.003) | 0.120 | 0.118 |
| 5: closed-form calibrated match, binary evidence | 0.769 / 0.860 / 0.642 | 0.852 / 0.959 / 0.732 | 0.603 | 1,116 (0.070) | **0.438** | 0.440 |

**The switch rule** (PLAN, 06:35, before the fifth's result): the fifth's lookups (0.4378) beat the second's (0.4343).
- The margin is small, but the rule is binding, so it was applied at 07:31.
- The second localizer's read fine-tune was stopped at update 800 of 2,000. Its log is kept as
  `run/train_log_v2_stopped_800.json` and `train_v2_stopped_800.log`.
- The read is retraining with the fifth (`train_v5.log`, then `readings_v5.log`).

**What each earlier failure was, measured:**
- **First localizer.** "No motion" was favoured by three things:
  - the night static's strength grows with screen eccentricity;
  - coverage: absent cells cost the background, about 19 nats each;
  - the agent's sprite, about 30 nats.
- **Second localizer.** Daylight was solved (moved, match alone 0.976). The dark was not: sensor-fixed content matches
  under "no motion" (the sprite gave +27.5 nats at night).
- **Third localizer.** The sensor-fixed component's free scale collapsed onto exact repeats, a mixture singularity
  (Bishop 2006, section 9.2.1).
  - Its log-scale was -8.3, against memory's -4.6, so the stay bonus survived.
  - The spatial prior's free scale collapsed too. A present uniform cell then cost log(1 - trust), and 52% of jumps
    landed on empty grid.
- **Fourth localizer.** The stay bonus went (+0.1 nats), and the moved match rose to 0.973 by day and 0.854 in the
  dark. But blocked steps were lost.
  - With the true explanation inside the null, "no motion" can't gain, while a wrong hypothesis wins chance matches in
    dark noise.
  - Godard et al.'s auto-masking assumes a camera that moves.
- **The learned per-observation sensor terms were never learned,** in any version.
- **What the checks for the fifth showed:**
  - P1.5's coarse tokens are translation-equivariant by day: the same world cell one screen cell over changes by a
    median 0.165, against 0.15 at the same cell. A per-cell least-squares offset broke that (1.06).
  - Laplace densities over 16 dimensions made a single rare cell worth +100-186 nats.
  - Binary evidence, as FAB-MAP's observations are, took the untrained match on moved daylight steps from 0.889 to
    0.995 (episodes 0-2, a map built at the true poses). The intermediate forms are recorded in the PLAN; their code
    isn't pinned.

## Where the fifth localizer's lookups go

`diagnosis_v5/chain.py` runs the real filter (its own map, the full grid, as reading 2 scores it) on the diagnosis
episodes 0-22. Episodes 23-119 stay unread by any diagnosis. Output: `chain_localizer_v5_0_23.json`.

**Per step, the trail holds by day and breaks in the dark:**

| Movement commands | n | Prior right | Posterior right | Match alone ranks the truth first | Support cells written under the current drift |
|---|---|---|---|---|---|
| Moved, day | 2,285 | 0.914 | 0.968 | 0.940 (ties 0.036) | 0.924 |
| Moved, dark | 1,516 | 0.600 | 0.688 | 0.656 (ties 0.071) | 0.915 |
| Blocked, day | 567 | 0.866 | 0.972 | 0.545 (ties 0.451) | 1.000 |
| Blocked, dark | 278 | 0.910 | 0.953 | 0.442 (ties 0.522) | 1.000 |

- In the dark, 227 of the 473 wrong posteriors on moved steps have the match losing on a consistent map. That's the
  dark observation model, separate from memory.
- On blocked steps the match mostly ties. The posterior is right because the prior decides.

**The lookups (16,050 on these episodes; 0.429 right):**

| Where the right entry was | Share |
|---|---|
| At the prior's address | 0.429 |
| Within the read's 9 candidates around it | 0.546 |
| Stored somewhere in the map | 0.892 |
| Stored, but 6 or more cells away | 0.316 |
| Not stored (overwritten at its address, or only ever seen under the agent) | 0.108 |

- The prior was right on 0.745 of lookups. Even then, the entry sat at its address only 0.576 of the time.
- So most of the loss is the map, not the step: the cell was written under one drift and is read under another.
  Revisits return to a place whose entries belong to an earlier, differently drifted stretch of the trail.

## Why loops don't close: the light changes what a place looks like

The agreed form corrects drift when an old place comes back into view. That needs the old view to match the new one.

**Could the filter's own evidence have closed the loop?** `diagnosis_v5/closure.py`, at each step where a lookup's
cell is stored under an older drift (6 or more cells off). It compares the summed match at the filter's pose with the
pose that stored entry implies, three ways. Relocating costs about 17.6 nats (-log(eps / 96²)).

Output: `closure_0_23.json`, 920 such steps (345 by day, 575 in the dark).

| Scored as | Cells at the old pose (holding that world cell) | Old pose minus filter's pose, median nats [IQR] | Old pose beats the relocation cost |
|---|---|---|---|
| The filter does: the local hypotheses' common support, every entry | 38.1 (26.8) | -273 [-323, -202] | 0.000 |
| Every present cell, every entry | 49.1 (34.8) | -336 [-390, -276] | 0.000 |
| Every present cell, entries older than 16 frames only | 45.3 (33.2) | -104 [-136, -50] | 0.049 |

- Even with the recent trail removed, the old pose loses by a median 104 nats, by day as in the dark (-108 and -101).
- Three quarters of its cells hold the very world cells now in view, so they ought to match. They don't. The
  sighting pairs below show what breaks a same-cell match by day. These 920 steps' own lighting wasn't recorded, and
  575 of them are in the dark. There, two sightings of a cell match 0.71 of the time, mostly one frame apart, against
  0.96 by day (`diagnosis_v5/repeat.py`, `repeat_0_23.json`).

**Two sightings of the same world cell, by day** (`diagnosis_v5/light_gap.py`, `light_gap_0_23.json`; simulator
positions pair the sightings, scoring only):

| Gap | Light change < 0.005 | 0.005-0.02 | 0.02-0.1 | 0.1 or more |
|---|---|---|---|---|
| 1 frame | 0.962 (192,924) | 0.965 (167,500) | — | — |
| 2-16 | 0.972 (1,521) | 0.956 (2,147) | 0.850 (2,382) | 0.141 (255) |
| 17-64 | 0.966 (148) | 0.975 (239) | 0.560 (835) | 0.005 (790) |
| 65 or more | 1.000 (32) | 0.935 (108) | 0.594 (421) | 0.000 (1,108) |

- **Time and viewpoint don't break recognition; lighting does.** At a steady light, the match holds at every gap.
  Split by screen shift (0-1 cells against 2 or more), it holds there too: 0.93-0.99 at 17-64 frames.
- **Most long revisits cross a lighting change:** of daylight revisits, 81% at 17-64 frames and 92% at 65 frames or
  more come after a change of 0.02 or more.
- **The match's tolerance comes from consecutive frames.** Its scale and threshold are calibrated on the change
  between consecutive frames after non-movement commands, where the light barely moves. The binary evidence also
  assumes a same-place match rate of 0.9. So an unmatched old cell costs -1.6 to -2.3 nats, depending on its
  content, instead of counting as uninformative.

**A label-free match that survives the light exists in the agent's own data** (`diagnosis_v5/within_scatter.py`,
`within_scatter_0_23.json`).
- Every run of non-movement commands in the TRAIN episodes shows the same cells, apart from whatever moves or is
  changed, while the light changes: 3.3 million pairs.
- Their change covariance is the within-place scatter, the scatter Fisher's discriminant discounts. Belhumeur,
  Hespanha & Kriegman's Fisherfaces project it away to recognize faces across lighting
  ([TPAMI 19(7), 1997](https://doi.org/10.1109/34.598228)). Here it is used only as the match's metric (a Mahalanobis
  distance), with the threshold at its 90th percentile on the still pairs, as the current threshold is.
- The comparison is a daylight sighting against the same cell's previous sighting, and against a random other cell of
  that earlier frame:

| Light change | n | Current match: same / other / AUC | Within-place scatter: same / other / AUC |
|---|---|---|---|
| < 0.02 | 353,085 | 0.963 / 0.452 / 0.756 | 0.970 / 0.508 / 0.756 |
| 0.02-0.1 | 3,539 | 0.754 / 0.342 / 0.771 | 0.962 / 0.454 / 0.766 |
| 0.1 or more | 2,082 | 0.018 / 0.010 / 0.789 | 0.864 / 0.345 / 0.794 |

- **Same-place recognition across a large lighting change rises from 0.018 to 0.864,** and the ranking power (AUC) is
  unchanged.
- **Chance matches rise from 0.010 to 0.345.** Many "other" cells are the same terrain, and FAB-MAP's new-place term
  is what discounts them: a match on common content is weak evidence.
- **A linear map of the lighting latent's change** removes only about a third of a same-cell change (median 1.36 to
  0.86, held-out half, changes of 0.05 or more). The lighting response isn't a simple shift.

## What it shows

- **The world's read works when the entry is right** (stage 1: 0.927 with the correct entry, 0.310 without), and **the
  localizer keeps a consistent trail by day** (posterior 0.968 on moved daylight steps).
- **Memory across time fails at recognition, not at the trail or at the read.** P1.5's coarse tokens aren't stable
  under the day's changing light. The localizer's match, calibrated on consecutive frames, treats a changed-light
  sighting of the same place as a mismatch.
- **This is a general problem,** not a Craftax one: any robot's camera sees a place differently across the day. It
  is the problem appearance-based loop closure exists to solve (FAB-MAP).

## What it doesn't show

- Whether the within-place scatter closes loops in the filter, or what it does to the filter's per-step accuracy.
  It was measured per cell, on sighting pairs, not inside the localizer.
- **The trail written during a drift** stays misplaced even after the pose recovers. The PLAN declared this: confident
  drifted writes need pose-graph loop closure, which the design doesn't have.
- **The dark's per-step weakness** (moved posterior 0.688) is separate. The night static degrades the view itself.
- **Reading 2's reference** (S2's simulator-trained shift reader on this panel) hasn't been run; it needs the GPU.
- **The fifth localizer's odometry-only score** is still running on the CPU.
