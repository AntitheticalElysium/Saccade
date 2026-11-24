# When the coal digit increment is learned, or not (11-19)

Codex's `localize.py` functions unchanged, its bank and seeds, 16 histories x 4 paired draws per cohort, first denoiser
call. A score is the move along the world's own true NOOP->DO step (0 = NOOP successor, 1 = DO successor), DO minus NOOP.
Checkpoints are 2.56M frames apart in every world (1-5: 64-step chunks, 6-10: 256-step). Checkpoint 10 reproduces Codex's
`p15_10_paired.json` (max difference 2e-5).

## Trajectory (`trajectory.py`, `p15_KK.json`)

| Coal 1->2 digit, both inputs DO | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F4 (Codex; 4 from `routes.py`) | 0.00 | | 0.05 | 0.09 | 0.85 | | 0.96 | | 0.98 | 0.99 |
| HUD-Change (Codex) | 0.00 | | 0.02 | | 0.02 | | 0.08 | | 0.22 | 0.24 |
| P1.5 | 0.00 | 0.00 | 0.15 | 0.07 | 0.16 | 0.10 | 0.09 | 0.12 | 0.15 | 0.15 |
| P1.5 faced tile | 0.00 | 0.01 | 0.87 | 0.89 | 0.85 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 |

- **The declared rule reads "rising", but at a rate that is no near miss.** Checkpoints 6->10: +0.041 [+0.015, +0.067]
  (episode bootstrap), +0.014 per checkpoint. P1.5's response formed at checkpoint 3, with the tile, and crept after;
  F4's jumped 0.09 -> 0.85 between checkpoints 4 and 5 (2,500 updates). More P1-style training at this rate does not fix it.
- **Stone 1->2 also creeps (+0.013 per checkpoint, to 0.95); stone 2->3 stays flat at 0.50-0.63.**

## Loss currency (Codex's saved `raw_digit_step`)

The coal digit's true token step has the same size in F4 and HUD-Change (median norm 3.68 against 3.58 at checkpoint
10). Within each world coal's step is about the size of stone 1->2's (HUD-Change 3.58 against 4.03), which is learned.
A copied coal digit costs about as much in either world; a smaller price for copying does not explain the gap.

## Crossed inputs (`routes.py`, `routes_*.json`): direct action x post-action belief

| Coal 1->2 | Tile: direct alone | Tile: belief alone | Tile: both | Digit: direct alone | Digit: belief alone | Digit: both |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| F4, checkpoint 3 (Codex) | 0.08 | 0.03 | 0.13 | 0.02 | 0.02 | 0.05 |
| F4, checkpoint 4 | 0.43 | 0.17 | 0.59 | 0.02 | 0.02 | 0.09 |
| F4, checkpoint 5 (Codex) | 0.59 | 0.25 | 0.82 | 0.27 | 0.11 | 0.85 |
| HUD-Change, checkpoint 3 (Codex) | 0.24 | 0.40 | 0.65 | 0.01 | 0.00 | 0.02 |
| P1.5, checkpoint 3 | 0.67 | 0.19 | 0.87 | 0.02 | 0.00 | 0.15 |
| P1.5, checkpoint 5 | 0.63 | 0.48 | 0.85 | 0.02 | 0.01 | 0.17 |
| P1.5, checkpoint 10 | 0.33 | 0.15 | 0.98 | 0.00 | 0.00 | 0.15 |

- **Route competition is refuted by its declared test.** P1.5's coal tile at checkpoint 3 is carried by the direct action
  alone (0.67, against 0.19 for the belief) while the digit stays at 0.15: the denoiser's own action-and-faced-block
  route formed without carrying the digit. HUD-Change's belief-carried tile does not generalise to P1.5.
- **F4 also learned the tile first, through the direct action** (checkpoint 4: tile 0.59, digit 0.09). Its coal digit then
  arrived abruptly within one checkpoint. So F4 and P1.5 do not differ in which part came first or through which input,
  but in whether that abrupt step happened.

## What remains

- Two explanations still stand:
  - **The recipe.** F4 is the only world with both shortcut forcing and the map-only change head. F3 had the head with
    EDM; HUD-Change and later worlds have shortcut forcing without that head (`ckpt_10.pt` configs). Codex attributes the
    loss to HUD-Change's grounding package.
  - **Run-level chance in an abrupt transition.**
- Every world is one seed. F3 failed coal increments, F4 learned them all, and HUD-Change, multi-step, F5, F6, F6.1 and
  P1.5 failed. F4's success was not predicted (F4 RESULTS: "my prediction REFUTED").
- Separating them takes training: the F4 recipe at another world seed, read at checkpoints 4-6 (F4's jump is at 4->5;
  F4 reached checkpoint 5 in 7.5 h). Not run.
- P1.5's direct record read reinforces the copy on top (Codex's centered test), but the failure began before the record
  existed (HUD-Change).
