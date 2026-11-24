# Decision 6's learned location, the locked design: results (2026-11-24)

**Verdict: stage A stops at its recognition gate. Stage B was not run.** Recognition almost never proposes the true
place: a single cell's appearance is shared by many places, and the shortlist is cut per cell before the arrangement
of the view can count. The plan's rule holds: the failure is diagnosed, and nothing changes until it is discussed.

One seed, as the PLAN declares.

## The pilot gates (`gates.json`, `stage_a.log`)

Read on held-out TRAIN episodes and the panel's episodes 0-22, with the evaluation outages.

| Gate | Result | Bar | |
|---|---|---|---|
| 1. The code's smoke tests | closed paths 0.0, order 0.0, offset composition 1e-6, origin change 0.0 | pass | **pass** |
| 2. Coverage with the sense | the right place among the read's candidates on 1.000 of revisits | ≥ 0.95 | **pass** |
| 3. Recognition | recall within 16 steps 0.019 (no recognition: 0.019); precision of accepted corrections 0.494 | recall ≥ 0.35 and precision ≥ 0.95 in every band | **fail** |
| 4. Drifted returns | 1,827 (1,520 on the panel); drift at the return 2.0 / 10.2 / 24.9 cells (10th, 50th, 90th percentile) | ≥ 200 | **pass** |

- **Recognition adds nothing.** Recovery within 16 steps is 35 of 1,827 drifted returns with recognition, 34 of 1,830
  without it.
  - 2,690 corrections were accepted, on 1,409 of the 1,859 events. 1,329 of them (0.494) moved the code toward the
    truth.
  - Recall in every band of light change and day or dark: 0.000 to 0.032.
- **The calibration is exact and stable.** All six refits found the 62 terrain cells and none of the 19 status-row and
  sprite cells, with offsets exact up to a constant. 4 of the 216 links are inconsistent throughout, and the robust
  fit absorbs them.
- **The key's training already said it couldn't do the task as posed.** Its InfoNCE top-1, against 127 negatives
  half of which are the closest raw tokens elsewhere, ended at 0.11-0.14, with a loss of 4.2 (chance: 4.85).

## Why recognition fails (`diagnosis_recognition.py`, `diagnosis_recognition.json`, `diagnosis_key_probe.json`)

Panel episodes 0-11, with the gates' outages. Every proposal made while the code is drifted is scored: 3,178 frames,
simulator positions for scoring only. 1,308 of them revisit, meaning some world cell has an earlier entry at its true
place. On those frames, 46% of the view's world cells have one, on average.

- **The vote almost never picks the true place.**
  - On revisit frames the winner is the true place 0.011 of the time, and the drifted place 0.139 of the time.
  - The winner's support has a median of 49.7 cells; the true place's, 0.
  - Even where at least half the view was seen before (559 frames), the winner is true only 0.027 of the time.
- **The true place's entries rarely reach the candidates.** For a cell with an entry at its true place, that entry is in
  the cell's top 32 by key for 0.17 of cells, and among its kept 8 for 0.13. Over all the view's world cells, those
  shares are 0.078 and 0.059.
- **The cause is aliasing: one cell's appearance is shared by many places.** For such a cell, a median of 114 entries
  at other places are at least as close by raw token (ties count). That is 106 at matching light and 130 after a light
  change of 0.05 or more, out of a median of 4,588 entries.
- **The learned key doesn't resolve it.**
  - By key, a median of 271 entries at other places rank at least as high: 298 at matching light, 255 across a change.
  - Its top 32 holds the true entry for 0.17 of cells, against 0.13 by raw token.
- **What wins instead is a recent frame, matched cell for cell.**
  - 0.976 of the winner's supporting entries were written at the query's own screen cell.
  - Their median age is 35 frames (10th to 90th percentile: 18 to 184).
  - An entry at the same screen cell implies the agent stood where it stood then. So every cell's shortlist points at
    the same few recent frames, and the place they imply collects about 50 cells' support.
- **Why the same screen cell wins among lookalikes.** The trained key keeps some dependence on the screen cell and on
  the lighting.
  - For the same token, another screen cell gives a similarity of 0.81, and another lighting 0.88. Another token gives
    0.56.
  - Among hundreds of equally close tokens, that leftover dependence decides the shortlist.
  - It isn't a shortcut learned from the training pairs: only 0.09 of the key's training positives share the screen
    cell (median age 40 frames).
- **Ruled out:**
  - the calibration, which is exact;
  - the stored entries and their offsets: coverage is 1.000 with the sense;
  - drifted entries outvoting old ones. The true place's support comes 0.98 from entries written before the outage,
    and the drifted place's only 0.05 from entries written during or after it.
- **Drift at these revisits:** 5.0 cells at the median (10th to 90th percentile: 1.7 to 11.0).
- **Paired against no recognition (Codex, 11-24, checked):** on the 1,522 panel returns where the no-recognition arm
  is drifted, recognition rescued 22 that the control missed and spoiled 21 that the control recovered.
- **The view's arrangement already separates many lookalikes (Codex, 11-24, rerun and reproduced).** This is a
  read-only ranking check on 136 drifted revisit frames from these episodes, each with at least 16 cells seen before at
  the true place.
  - Memory: the newest observation at each place from before the outage, at the true positions. Before an outage the
    estimator's positions are exact too.
  - Candidates: every place in the episode's extent with at least 16 observed cells.
  - The frozen raw tokens, compared under one shift by mean L1 over the overlap, rank the true place first on 0.728 of
    frames (0.630 after a light change of 0.05 or more), and in the top five on 0.853.
  - Comparing pooled means instead gives 0.404 (0.167 across a light change).
  - Not tested: a drifted map, and declining a convincing false match.

## What this does and doesn't touch

- **The locked design says "a learned recognition key of the new view"** (AGENT.md decision 6). The PLAN made the key
  per cell, κ(token, context), and built the context from the lighting and the screen cell only, with a shortlist of 8
  per cell. The failure is in that choice: no context from the rest of the view enters before the shortlist.
- **Not touched:** the calibration, the position code, path integration, the entries and their offsets (coverage
  1.000), and repair. The gate can only weigh the candidates it is given. Stage B's read was never run.
- **Next:** a research pass on recognition among lookalikes, then a proposal for the user and Codex. The design is
  amended before any rerun.
