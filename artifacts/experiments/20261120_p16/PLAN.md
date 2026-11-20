# P1.6 (2026-11-20; the user: "let's bundle it for tonight's overnight run - P1.6")

From scratch, with P1.5's recipe, schedule (25.6M frames) and world seed, plus one change: G1, the foveal record.
- **Launcher:** `run.py`, P1.5's one-shot worker. It is gated by `checks.json`, its sources are sealed by
  `pin_sources.py`, and it runs `evaluation.py` (written during training) on the selected checkpoint.
- **Status:** checks passed (`checks.json`); not launched; it waits for the user's review of this plan and the code.

## The change: G1, the foveal record ("write where you look")

**What it is.**
- `Record` gains a second channel group when `record.fovea_modes` is set.
  - It records the glimpse canvas the belief already reads: each fixated cell's 64-dimensional fine token, zero
    elsewhere.
  - It uses the first 8 modes of the record's own S4D-Lin basis (dt 1/32, reset at episode starts, stop-gradient
    inputs: LOSSES 7a).
- The record's input per cell is the periphery token and the canvas side by side (`Record.input`). Its features are
  the periphery group's 1,024 followed by the fovea group's 1,024.
- Both reads take both groups, so their input widens from 1,024 to 2,048:
  - the denoiser's `record_read`, whose last layer stays zero-initialised, so the read is inert at the start;
  - Mamba's `record_in`.
- Under the factorised denoiser this is the first route by which past fine evidence reaches the physical tokens
  directly. Today the only route is the 256-D belief.
- **In imagination,** the generated glimpse at the chosen window writes the same record: the rollouts pass the
  generated glimpse and its fixation to `s1.history`.
- **The code** (`saccade/`): `Record.input` and the grouped `scan`; the reads sized by `Record.size`; the trainer's
  and the monitor's record input; the `s1` helpers (`history`, `before`, `window`) carry the glimpse. Without
  `fovea_modes` every path is P1.5's, bit for bit: the periphery group equals test 2's record exactly, P1.5's
  checkpoint loads strictly, and the 78 earlier CPU tests pass.

**Why (measured).**
- **11-02, F1.5** (`20261102_foveal_information`):
  - At night, fine crops carry entity evidence that the coarse scene and the 8-frame history lack. Macro recall
    rises 5-9 points, and skeletons go from .70 to .84 and from .77 to .93.
  - The only route from a past crop to a future prediction is the 256-D belief and its four memory tokens. The crop's
    contribution there adds nothing (79.2% against 79.8%): an accessibility gap.
- **11-20, the switching dissection:**
  - A5: a glimpse gives a mob no protection.
  - C5: Mamba's belief barely holds mob positions (AP 0.048 against the previous frame's 0.255), and night deletions
    concentrate at the edge ring.
  - C7: the belief knows a zombie is adjacent (0.921) but not which cell (0.116).

**Sources.**
- Attention gates what is encoded (Aly & Turk-Browne 2016).
- Writes restricted to where the agent is (Neural Map, Parisotto & Salakhutdinov 2018), and attention-gated writes
  (DRAW, Gregor et al. 2015).
- The basis (S4D, Gu et al. 2022).

**Parameters.**
- **`fovea_modes = 8`:** 2 x 64 x 8 = 1,024 features per cell, the periphery record's size (P1.5's measured memory
  fits). Mode 7 turns 7 pi / 32 = 0.69 rad per step, resolving recency to about 4.6 steps inside the basis's 64-step
  decay.
- **No "seen" flag:** glimpse tokens are layer-normalised and never zero, so a written cell is distinguishable from
  an unwritten one.

**Known limits, stated before the run.**
- **Screen-registered,** as the periphery record is.
  - The camera moves on 35.3% of steps (all 1,009 DEV episodes). A trace then sits under another world cell until the
    denoiser's spatial attention re-registers it.
  - 37% of frames lie in stretches of at least 8 still steps, where a trace stays registered.
  - There is no remapping: a scroll rule would be Craftax-specific. The brain remaps with a copy of the motor command
    (Duhamel, Colby & Goldberg 1992); a learned remap is a later option.
- **Fixed decay:** a trace fades whether or not its content changed. G2's learned write and hold is the later fix.
- **It stores what P1.5's chooser looks at.** The eye is unchanged; the alternatives tested on 11-20 were not better
  (C4, C8).
- **Gaze as a cue.** The physical prediction now depends on past gaze. A trace's existence means "looked here", which
  correlates with content through the chooser's policy, so the world could learn to draw entities where the eye
  looked. Reading 6 measures it.

**Not changed:** the eye, the sampler, the schedule, the data, the seed.

## Pre-launch checks (`checks.py` -> `checks.json`)

- **units** (GPU), measured:
  - the periphery group equals test 2's record (0.0);
  - the fovea group is zero wherever the eye never looked;
  - terms are finite, and route by role is as in P1.5;
  - the gradient reaches the foveal columns of both reads (denoiser read from G1 0.0072; belief input from G1 6.7e-5
    and from reward 0.012);
  - the carried state is [streams, 81, 2,048];
  - the evaluation adapters (`s1.encode`, `s1.window`) reproduce a contiguous scan within 7.4e-5 (fp16-stored
    tokens).
- **CPU:** `test_fovea_record_writes_where_it_looks` and the 78 earlier tests pass.
- **throughput** (measured): the trainer itself, 30 updates each.
  - 64 steps: 2.08 s per update (P1.5 2.03, +2.6%), peak 3.66 GiB (P1.5 3.08).
  - 256 steps: 7.94 s (P1.5 7.65, +3.8%), peak 3.56 GiB (P1.5 3.02).
  - Projected 14.1 h against P1.5's 13.7 h on the same estimate; P1.5 actually took 15.4 h, so expect about 15.9 h.
- **Gate (`criteria`):**
  - the periphery group exact, foveal writes only where looked, finite terms, routes, and foveal columns reached;
  - adapters within 1e-2;
  - peak under 4.5 GiB;
  - projected training under 17 h (scaled by P1.5's measured 15.4 h over its own estimate).

## Readings (the evaluation, written during training; P1.6 against P1.5)

1. **P1.5's evaluation, same scope** (S1, the S2 battery, the monitor), with the rollouts writing generated glimpses.
2. **Does the world use the foveal record?** The G1 loss per region with the foveal features zeroed against intact
   (B2's reliance form), by day and night and by camera still or moved.
3. **Does looking protect a mob?** A5's paired on / away intervention.
4. **C5's calibration:** keep rates by kind, light and ring, beside P1.5's.
5. **C7's zombie probe:** does the belief now hold which cell?
6. **Gaze as a cue:** entities drawn at cells looked at against cells not looked at, in the world's own rollouts.

**Predictions.**
- (2): reliance concentrated at night and at mob cells, near zero by day.
- (3): a glimpse protects night mobs at t + 1 by a few points (P1.5: -0.1 +- 0.1).
- (4): night and edge keep rates rise for looked-at mobs; the day coin is unchanged (C5: by day the evidence is
  already present).
- (5): per-cell zombie AP rises modestly.
- (6): no excess spawns at looked-at cells.

## Considered on 11-20 and not in P1.6

- **G2, a learned write and hold:** it fixes the decay, but needs a gradient through the scan and a request for it.
  It waits for (2).
- **G3, a token memory:** it overlaps Mamba's memory tokens and is the heaviest.
- **Hand-coded entity files and `RECORDS.md` v3:** hand-coded rules (the user, 11-20) and heavy machinery.
- **Learned entity slots** (object files). These are the brain's closer analogue: transsaccadic memory is a
  limited-capacity store of 3-4 objects, not tied to absolute position (Irwin 1991), linked by spatiotemporal
  continuity (Kahneman, Treisman & Gibbs 1992). Not tonight:
  - every part is learned (encoder, binding across frames, dynamics, a read path into the denoiser and the actor);
  - binding is fragile, and peers rely on pretrained slot encoders (SOLD, ICML 2025; ObjectZero, 2026).
  - G1 is the cheap test of glimpse-written memory; slots are the fuller version if (2) shows the world uses it.
- **The eye's score:** C8's belief change failed. The world-eye ties the chooser on entities and halves HUD fixations,
  at measured cost (fresh +12.2 h; refreshed every pass +6.3 h). It waits until G1 lets us measure what looking buys.
- **History guidance** (DFoT, Song et al. 2025) for the persistence coin. Shortcut forcing already noises every frame
  independently, so it can be tried on P1.5 offline; P1.6 needs no training change for it.
