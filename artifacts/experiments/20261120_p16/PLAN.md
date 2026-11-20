# P1.6 (2026-11-20; the user: "let's bundle it for tonight's overnight run - P1.6"; "Bundle the world-eye into P1.6, two refreshes are fine")

From scratch, with P1.5's recipe, schedule (25.6M frames) and world seed, plus two changes: G1, the foveal record, and
the world-eye from pass 7 (one fill after checkpoint 6). Fixes found by the checks and by Codex's review (11-20 evening)
come with them: the checkpoint monitor's memory, its resume, and the widened reads' initialisation.
- **Launcher:** `run.py`, P1.5's one-shot worker. It is gated by `checks.json`, its sources are sealed by
  `pin_sources.py`, and it runs `evaluation.py` on the selected checkpoint. The evaluation is written and smoked before
  launch (Codex: P1.5's evaluation failed three times after training).
- **Status:** checks passed on the final code (`checks.json`); not launched; it waits for the user's review of this plan
  and the code.

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
- **Initialisation (Codex's review):** both reads are built exactly as in P1.5 (their own seeded draws, bias and
  Mamba's downstream layer), then widened with the fovea group's columns at zero (`model.widen`). The initial world
  is P1.5's: every shared parameter is bit-equal, and on a real episode with glimpses written, Mamba's belief equals
  P1.5's (RMS 0.0) and the denoiser's read 2.4e-8 against a scale of 0.066. Before the fix, widening redrew the
  coarse columns and the downstream layer (the belief moved by 0.161 RMS, Codex). The first passes now isolate what
  the new route learns.

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
- **Through pass 6 it stores what P1.5's chooser looks at;** afterwards, what the world-eye looks at.
- **The world-eye's schedule comes from P1.5's checkpoints.** P1.6's world (G1) may mature at a different pass. C11
  also leaves open how the foveal record changes which looks are useful (Codex).
- **The world-eye has no form in imagination yet:** it needs a real frame. The evaluation's imagination keeps the
  chooser, as P1.5's did.
- **G1 and the eye change together.** In P1.5 the two eyes differ by TV 0.13 (deployed). Readings 2-3 isolate G1's use;
  reading 7 reads the eye.
- **The fill's looks** are each frame's most recent actual look. With G1 the estimate depends on looks by design, and
  this was not measurable on P1.5 (there, borrowing the chooser's looks moved the eye by TV 0.007).
- **Gaze as a cue.** The physical prediction now depends on past gaze. A trace's existence means "looked here", which
  correlates with content through the chooser's policy, so the world could learn to draw entities where the eye
  looked. Reading 6 measures it.

**Not changed by G1:** the sampler, the schedule, the data, the seed.

## The world-eye (from pass 7)

**What it is.**
- Passes 1-6 keep P1.5's chooser.
- Right before checkpoint 6 is saved, a fill writes every training frame's per-cell error under the world's own
  estimate: ||p_t - x_hat_t||^2, with x_hat_t the flow estimate at signal 0 (the conditional mean, C2's) given the real
  past (8 context frames at the context signal, the record, the predicted belief).
  - The belief and the foveal record are built through each frame's most recent actual look in the gaze trace.
  - The fill is `eye_cache_06.npy` in the run directory.
- After the fill, the chooser's S is replaced by the cached error, summed per window. Unchanged: the head's contingency
  C, the policy (0.8 softmax(0.5 C + 0.5 S) + 0.2 uniform), the head's own training, and the gaze forecast, whose
  teacher becomes this distribution.
- **Code** (`saccade/`):
  - `Denoiser.estimate`: a two-stream pass that gives every frame's estimate given its real past in one chunked pass.
    It equals `forward` on each frame's own window (CPU test).
  - `chooser.world_errors`: the fill, with per-episode draws, so the result does not depend on segmentation.
  - `Controller.refresh` / `looks`, and a resume that checks the cache's digest.
  - The trainer's refresh, before the listed checkpoints.
  - The training forward is bit-equal to the prepared P1.6's; all 80 CPU tests pass.

**Why (measured, `20261120_switching_dissection` C9-C11b).**
- **C9:** in the form used here (the chooser's C with the world's S), the fresh world-eye beats the chooser modestly
  on P1.5's checkpoint 10: entities 0.514 against 0.484, cows +4.4, zombies tie, and changed-HUD looks 0.299 against
  0.409. An estimate cache chases encoder drift; a score cache (the error stored in its own tokens) does not.
- **C11:** a cache's coverage is set by the checkpoint that filled it, not by its age, and early worlds make poor eyes.
  - The rule declared for two fills picked passes 5 and 6: entities 0.4790 over the ten passes.
  - One fill after 6 gives 0.4784, against the chooser's 0.4660 and the fresh eye's 0.4805; zombies 0.342 against
    0.341.
  - One fill was adopted after Codex's review: it gives up 0.07 points of entity coverage to save 47 minutes, under
    the user's "privilege the smallest possible cost". It is one line in the recipe if two are wanted.
- **C11b:** the chooser beats uniform from its first updates, so the early passes keep it.

**Parameters:** `fixation.world_eye.refresh_after = [6]`; the fill's own seed. **Cost:** one fill of 0.78 h, measured
on training episodes.

## A fix found by the checks: the checkpoint monitor

The monitor scanned the record over whole monitor episodes at once. With G1's 2,048-dim record a long episode needs
1.55 GB per stack, and the scan builds two stacks plus their per-frame lists. The first checkpoint ran out of memory
even in a clean process. P1.6 as prepared this morning would have crashed there; its throughput check ran without
checkpoints. The monitor now scans in 256-frame pieces and keeps the denoiser's read only for its windows' frames. On
P1.5's checkpoint 10 it equals the old monitor within bf16 (g1 2.1e-4 relative, per level at most 1e-3).

Codex's review found two more problems:
- **The kept rows were views,** which keep their whole 256-frame chunk alive: up to 1.55 GiB retained in the worst
  episode. They are now copies (216 MiB there). `s1.history`'s kept rows had the same pattern and are copies too.
- **A monitor that fails after its checkpoint is saved would be skipped on resume,** because `latest.pt` has already
  advanced the counter, while the selector needs every row. On resume, a missing row is now backfilled from that
  checkpoint's own weights. The `resume` check injects one failure.

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
- **CPU:** `test_fovea_record_writes_where_it_looks`, `test_two_stream_estimate_equals_per_frame_windows` and the 78
  earlier tests pass (80; Codex's CPU-only run: 72 passed and 8 skipped, those needing CUDA).
- **init** (Codex): the initial world equals P1.5's, as above.
- **resume** (Codex): the monitor fails right after checkpoint 1 is saved (injected). The resumed run backfills it,
  rows 1-4 are present once and in order, the selector accepts them, and the backfilled value equals a fresh monitor
  of checkpoint 1 exactly.
- **throughput** (measured): the trainer itself, 30 updates each.
  - Morning (G1 only): 64 steps 2.08 s per update (P1.5 2.03, +2.6%), peak 3.66 GiB (P1.5 3.08); 256 steps 7.94 s
    (P1.5 7.65, +3.8%), peak 3.56 GiB (P1.5 3.02).
  - Evening (final code): 2.27 / 8.59 s, then 2.24 / 8.53 s.
  - **A/B, same session, alternating:** the prepared P1.6 2.251 / 2.246 s against the final code 2.253 / 2.254 s. The
    code is not slower; the GPU is in a slower state than in the morning. P1.5's own run averaged 2.284, so the evening
    rate is what a long run gets.
  - **With bootstrap active** (Codex: the throughput runs stopped before update 5,000, where the shortcut bootstrap
    starts): 2.44 s at 64 steps (2.19 without, +11%, as in P1.5's own log: 2.084 -> 2.312) and 9.21 s at 256 steps.
    One checkpoint monitor takes 100 s.
  - **Projection:** 5,000 updates at 2.19 s, 7,500 at 2.44 s and 3,125 at 9.21 s, plus ten monitors and one fill:
    17.2 h. The earlier 16.7 h left out the bootstrap and the monitors. The old criterion's scaling by P1.5's
    run-over-estimate ratio is reported only.
- **eye** (measured):
  - the fill equals `forward` on each frame's own window within 0.2% (first frames, segment boundaries, beyond the
    window);
  - segmentation changes it by 0.23%;
  - in a short trainer run the fills happen after the listed checkpoints. The looks after a fill are the cache's
    argmax in 100% of rows, against 2.8% before (the control);
  - a resumed controller reopens the cache, and a changed file is refused;
  - the run's peak with fills and monitor is 3.66 GiB, and the fill's own is 1.52 GiB;
  - the fill takes 1.15 ms per frame: 0.78 h for the split's 2.44M frames.
- **Gate (`criteria`):**
  - the periphery group exact, foveal writes only where looked, finite terms, routes, and foveal columns reached;
  - adapters within 1e-2;
  - peak under 4.5 GiB;
  - the eye's checks as listed, peak with fills under 5 GiB, and the projection with fills under 19 h;
  - init equal to P1.5's, and the resume backfill.

## Readings (P1.6 against P1.5)

1. **P1.5's evaluation, same scope** (`evaluation.py`: P1.5's queue with P1.5 as the comparison arm; the S2 battery
   and the monitor), with the rollouts writing generated glimpses.
   - Every record call on the evaluation path now passes the glimpse and its fixation: `s1`, `s2`, `interface`,
     `routes`, `sleep`, `dissect`, `native_pairs`, `trace` and `trajectory`. P1.5's record ignores them.
   - The adapter check scans the record with the glimpse and checks that the generated glimpse moves the futures'
     read.
   - Readings 2-7 are their own scripts, written during training and run after the queue.
   - **Smoke, before launch** (`checks/smoke/`): on a real P1.6 checkpoint (two updates, checkpoints and a fill), the
     adapter check passes. The futures' Mamba read equals a glimpse-written scan exactly, and zeroing the glimpse moves
     it; for the check, Mamba's fovea columns are briefly made non-zero, because they start at zero. Every smoke step
     completes: `s1`, `native_pairs`, `consequences`, `interface`, `recursion_4` and `natural_thirst`.
   - The first smoke failed in `recursion_steps` and `natural_pair`, which the earlier search missed; both now pass
     the glimpse, as does `s2/prefix_check`.
   - `s2b` is left unpatched on purpose: its gaze and evidence modes vary independently, so a one-line patch could pair
     a real glimpse with a chosen fixation. The guard makes it fail loudly if used.
   - The full queue's stages (clocks, trajectory, movement, mining, sleep, entities, compare) cannot run on a smoke
     checkpoint, which they require to be the run's own. They go through patched tools; a failure there stops only
     the queue, and training and selection stand.
2. **Does the world use the foveal record?** The G1 loss per region with the foveal features zeroed against intact
   (B2's reliance form), by day and night and by camera still or moved.
3. **Does looking protect a mob?** A5's paired on / away intervention.
4. **C5's calibration:** keep rates by kind, light and ring, beside P1.5's.
5. **C7's zombie probe:** does the belief now hold which cell?
6. **Gaze as a cue:** entities drawn at cells looked at against cells not looked at, in the world's own rollouts.
7. **The eye in training:** coverage of the looks actually made (C11b's method on the labelled training episodes),
   per pass, against P1.5's trace at the same passes.

**Predictions.**
- (2): reliance concentrated at night and at mob cells, near zero by day.
- (3): a glimpse protects night mobs at t + 1 by a few points (P1.5: -0.1 +- 0.1).
- (4): night and edge keep rates rise for looked-at mobs; the day coin is unchanged (C5: by day the evidence is
  already present).
- (5): per-cell zombie AP rises modestly.
- (6): no excess spawns at looked-at cells.
- (7): passes 1-6 track P1.5's chooser. From pass 7, entity coverage rises by about 2-3 points over P1.5's at the same
  passes, and changed-HUD coverage falls by about 7-11 points.

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
- **The eye's score:** C8's belief change failed. The world-eye is now in (above), from a score cache, at one fill.
- **A uniform start** (user 11-20): the chooser beats uniform from its first 250 updates (C11b), and the deployed eye
  needs the head's contingency in any case. Removing the chooser entirely would need the world to supply C as well
  (its predicted effect of the action), at about 1.5 times the fill's cost; not measured.
- **History guidance** (DFoT, Song et al. 2025; tested on P1.5 as C10).
  - Masking only the frames leaves the day coin untouched, and helps at night (inner cows deleted 14.9% -> 6.3% at w 3).
  - Masking the whole history cuts the day deletion by 60%, but overshoots where the history is decisive (static-HUD
    error x10), makes mobs move less and makes draws less varied.
  - It is sampling-only, so no training change.
