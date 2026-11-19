# Random switching before P2 (2026-11-19; user: test switching and its variants before committing to P2; Codex reviews
before launch; a control arm only after the switch arm is dissected)

## Why

S2 on P1.5 (`20261119_s2/RESULTS.md`): mobs fade over the world's own rollouts (cows -37%, skeletons -40%, arrows -32%
by step 16; TEST_SPEC S3's bookkeeping blocker fires). In P1.5's recursion (128 windows), step 16 reads 1.17 mobs under
real history and 0.795 free running, against 1.27 in truth: reading its own frames costs more than the one-step error.
The sampling world models in our sources do not train on their own rollouts; they make the one-step model robust to its
own inputs (`20261119_p2_decisions/RESEARCH.md` 6). CURRICULUM 2(b) keeps JEDI's switching as "the FIRST ablation if the
curriculum proves to need more self-feeding". P2's rollout score stays the alternative, with its loss-sensitivity
check before any trainer is built.

## Sources (read for this plan)

- **JEDI** (`2605.13013`, section 3.4, Algorithm 1, its LaTeX source in `third_party/paper_sources`): "with every
  trajectory batch we randomly switch with uniform probability between using the denoiser output and the encoder output
  as the next-step conditioning signal"; the denoiser output is its one-step estimate of a noised copy of the recorded
  latent at an EDM-sampled noise level; ablation: deterministic switching, or none, is worse.
- **Self-conditioning** (Chen, Zhang & Hinton, Analog Bits, ICLR 2023): the model's own earlier estimate fed back as an
  input on half the training steps, without gradient; a different purpose (refinement), the same 50% device.
- **Noise augmentation** (GameNGen 2408.14837; Dreamer 4's context corruption; diffusion forcing): corrupting the context
  lets the model "correct information sampled in previous frames"; without it quality degrades within tens of frames.
  We already train with per-frame noise (diffusion forcing); Gaussian noise is not the model's own error, which is the
  gap switching targets.
- **Scheduled sampling** (Bengio 2015; SimPLe 1903.00374: mixing raised to 100%) and its inconsistency for sampled
  continuations (Huszar 2015); our E8 (10-01, pre-saccade): paired self-fed L1 installed a scroll bias. Switching avoids
  this: the substitute is an estimate of the recorded frame itself, so the recorded continuation stays its target
  (CURRICULUM's caveat: only to the extent the estimate keeps what the continuation depends on).
- **Professor forcing** (Lamb et al. 2016): match the hidden-state dynamics of free-running and teacher-forced modes with
  a discriminator. Heavier; kept as a variant.
- **The brain, with limits stated:**
  - The wake-sleep algorithm (Hinton, Dayan, Frey & Neal 1995) trains recognition on the generative model's own
    "fantasies", where the causes are known exactly.
  - The overfitted-brain hypothesis (Hoel 2021, Patterns): dreams as self-generated out-of-distribution input that
    keeps perception general. Untested.
  - Object permanence is carried by dedicated object representations (object files: Kahneman, Treisman & Gibbs 1992),
    not by redrawing the scene. That points to the agreed entity records (DECISIONS: "Entities as object records";
    increment I2 on a measured trigger), a structural alternative and not a switching variant.

## Step 0: which route's self-feeding loses the mobs (frozen P1.5, `20261119_s2/s2b.py routes`)

Stage 1's bank, 1,000 starts x 4 futures, the same seeds in every arm. In free running three routes read the generated
frames: the denoiser's context, Mamba and the record. Four arms feed the recorded frame instead, while the recording
lives: none (free), the context only (ctx_real), Mamba and the record only (mem_real), all (teacher). The scored frame
is always the generated one. After Codex's review:
- every arm uses the recording's own fixation schedule, so a recorded glimpse sits at the crop its fixation names (no
  chooser);
- every arm feeds the belief the world's own sampled reward and termination, so arms differ in frames only.
Read: mob counts by step paired against free running, with survival beside them.
**Rule for the first arm's routes, fixed now:** switch the routes whose recorded input recovers at least a third of the
free-running step-16 deficit (teacher minus free); both if each does, or if only teacher does.

**Result (`20261119_s2/RESULTS.md` 5):** Mamba and the record fed recorded frames recover 96% of the step-16 deficit
(+0.44 of +0.46 mobs per frame); the context fed them recovers nothing (+0.000). By the rule the first arm switches
**mem only** (`--routes mem`): Mamba and the record read the substitute, the denoiser's context keeps the recorded
frames. Mamba (frozen here) and the record (no parameters; read by the denoiser's record_read) were one arm.

**Codex's review of the split and the arm (11-19):** adopted. On the mem route alone a switched frame changes the
conditioning from the next frame on, so every G1 target is kept (the exclusion applies to the ctx route only); the
control arm (`--substitute 0`) is then the same fine-tune without substitution, with identical draws. Reading rule:
success is less decline from step 1 to step 16 and better 4-step persistence (a higher final count alone could come
from drawing extra mobs at step 1), with clocks, danger and first-step action effects (the holds) as guards; a flat
result bounds this fine-tune (one-step substitutes from recorded history may miss errors correlated along a rollout).
The exact `--routes mem` command is smoked before launch.

**The split (`20261119_s2/RESULTS.md` 5, user: run before the review):** the record carries 94% of the loss (+0.43 of
+0.46 at step 16), Mamba's percept input about 2% (+0.010, SE 0.008). The arm keeps the mem route as declared: it
mirrors free running on the memory side, and the part that matters, the record's read, is denoiser parameters that
this fine-tune trains.

## The first arm (`switch_ft.py`)

- **Protocol:** that of the earlier denoiser fine-tunes (traces, SWAX, record).
  - From P1.5 checkpoint 10 with its streams, carry, cache and record state; the denoiser alone trains.
  - G1 alone (shortcut forcing, bootstrap rows included); AdamW 1e-4, 100 warm-up updates, clip 1.0.
  - 64-step chunks, 16 streams; the checkpoint's chooser picks the fixations (frozen).
  - 4,000 updates, read at 2,000 and 4,000 (tests 2-3's length; the earlier denoiser fine-tunes were still improving
    at 4,000-6,000), extended to 6,000 only if the 4,000 reading is still moving. JEDI uses switching from scratch,
    throughout its 400,000 updates (1,000 epochs x 400, Atari100k online); this fine-tune screens the mechanism, and a
    positive result goes into a from-scratch run, as the record went from tests 2-3 to P1.5.
- **Switching:** each chunk frame enters the switch mask with p = 0.5 independently (JEDI's uniform choice) and is
  replaced on the chosen routes by the denoiser's own one-step estimate of it. Masked frames serve as history only: their
  own G1 terms are excluded (JEDI switches the next prediction's conditioning; Codex's review).
  - The estimate comes from an independent noise draw at an independent flow level, conditioned on the recorded past
    and memory, and is clamped.
  - Targets are always the recorded frames. The carry, cache and record passed to the next chunk come from the
    recorded frames.
  - In this single-stream diffusion-forcing world a frame's noisy tokens are both its own denoising input and later
    frames' context; excluding the switched frames' own terms removes the same-frame self-correction term.
  - On bootstrap rows the target is the model's own finer-step prediction, as in P1; only flow rows target the recorded
    frame.
  - The substitute's error against the recorded frame is logged by signal level (at zero signal it carries no direct
    information from the recorded frame: the model's conditional prediction).
  - Three seeded generators shared by both arms: G1's levels and noise, the mask, the substitute's draws; the smoke
    checks that the arms' draws are identical.
- **Readings, at 1,500 and 3,000 updates:**
  - Primary: S2 stage 1 against P1.5 on the same bank and simulator futures: mob counts and 4-step persistence by
    kind, the bookkeeping blocker, mobs by step.
  - Guards: the G1 monitor (P1.5's 2,000 held-out windows), S2's L2b clocks and danger, first-step consequences,
    internal consistency, N5; stage 1b on competent starts, once it exists.
- **Decisions:**
  - If the mob deficit shrinks with the guards intact, the control arm runs next, to separate switching from extra G1
    training: the same fine-tune, the same mask and excluded frames, no substitute (`--substitute 0`).
  - If it does not shrink, the variants below come before P2.
- **Cost:** about 3 s per update (one extra no-gradient denoiser pass and a belief pass), so about 3.3 h, plus readings.

## Variants (brainstormed; one at a time, after the first arm is read)

1. Routes: context only (JEDI's) against context plus memory and record (set by step 0 for the first arm).
2. The substitute: a one-step estimate at a random level (cheap, JEDI) against the sampler's own K-step output (the
   inputs free running actually sees; sequential, expensive).
3. Depth: chained estimates (the estimate of frame t + 1 from a substituted frame t), V-JEPA 2-AC's two-step rollout in
   our sampled form.
4. Rate: fixed 0.5 (JEDI) against annealed toward 1 (SimPLe).
5. Training Mamba too, in the full recipe (route by role kept), so the belief update itself learns to read generated
   frames.
6. Hidden-state matching (professor forcing) or a sleep phase on dreamed frames (wake-sleep), only if the above fail.
7. (After the split) Corrupt the record's input as the context is corrupted: in P1 the record reads clean recorded
   tokens while the denoiser's context frames are noised; building it from the noised (or context-level) tokens would
   give the record the robustness the context already has. A recipe change for a from-scratch run, not a fine-tune.

## Limits

One seed; a fine-tune, not a from-scratch run; the readers read generated frames off their domain (TEST_SPEC).

## Launch (11-19, user: run to 4,000, maybe 6,000; Codex's review of the record split adopted)

- **The arm:** the record route alone (Codex: Mamba's percept input keeps the recorded frames and reads the switched
  record; its percept route carried about 2% of the loss), fresh substitutes, p = 0.5 per frame, every G1 target kept,
  4,000 updates in one continuous run.
- **Saved at the reads:**
  - S2 checkpoints at 2,000 and 4,000;
  - the live state at 4,000 (optimizer, streams, carry, cache, record, generators, chooser), so `--resume` can extend
    the same run to 6,000 (Codex).
- **Readings (`run_arm.sh sw_rec 4000`):**
  - S2 stage 1 at 2,000 and 4,000;
  - stage 1b at 4,000;
  - P1.5's exact clock checks at 4,000 (check 17's due-versus-delayed pairs, check 15's natural catalogue).
  - Reason for the clock checks: Codex saw due-step hunger drops fall 42/48 to 35/48 on a frozen replay after the
    100-update memory-route smoke.
- **Decision at 4,000:** extend to 6,000 if persistence is still improving and the guards hold. The matched no-switch
  control (`--substitute 0`) comes after, if the arm passes.
- **Not run:** the grounding replay of the F4 regression (user: not worth the compute for that diagnostic).
- **Parked for after the arm:** the epoch-stale cache (STALENESS.md), if switching carries over to a from-scratch run.

## Rerun (11-19 night; user: rerun, then call the shots)

- **Why:** the first arm (`RESULTS.md`) re-warmed a fresh AdamW to 1e-4, P1.5's peak and ten times the rate its cosine
  schedule ended at (1e-5 for the denoiser), with weight decay on every denoiser tensor (P1.5 decays Linear / Conv
  weights only). Its smoke (on the context and memory routes, not the arm's record route) moved the held-out one-step
  monitor +3.5% in 20 warm-up updates, against +0.46% for this rerun's smoke on the record route: suggestive, not a
  clean comparison (the routes differ). Re-warming a converged model costs upstream loss over a short horizon (Gupta et
  al. 2023, arXiv 2308.04014).
- **The readers (S2 `RESULTS.md`, `routes_readers.json`):** the denoiser's own read of the record carries the whole
  self-feeding loss (+0.433 of record_real's +0.432 at step 16), Mamba's `record_in` none, so the denoiser alone trains,
  as before.
- **The change:** P1.5's optimizer continued: its two denoiser groups (51 Linear / Conv weights at decay 0.01, 48
  others at 0), the rates its schedule ended at (1e-5), its moments and step counts (15,625), no warm-up. Everything
  else as the first arm (record route, p = 0.5, fresh substitutes, every G1 target kept, the same seeds and draws).
- **Smoke (`smoke_lr.log`):** the groups and step load as above; peak 3.2 GiB; the monitor moves +0.46% in 20 updates.
- **Readings and rule:** as the first arm (`run_arm.sh sw_rec2 4000`). Pass: less step-1-to-16 decline and better
  4-step persistence than P1.5, without interior spawns above P1.5's, with the clocks (energy curves, overdue
  food / drink), N5 and return no worse than P1.5's.
- **Then:** a pass runs the matched control (`--substitute 0`, the same optimizer and draws) to separate switching
  from extra G1 training; persistence still improving at 4,000 with the guards intact extends to 6,000 first. A fail
  with the first arm's direction (spawns, record distrust) at the right rate bounds one-step switching on the record.

## Control after the fail (11-20 early morning; user: call the shots)

- **Why now, although the plan ran it only after a pass:** the rule judged the package against P1.5. The package is
  switching plus 4,000 extra denoiser-only G1 updates at 1e-5. Its changes could belong to either part:
  - the hunger clock (on schedule 0.873 → 0.781);
  - zombie persistence (0.911 → 0.864);
  - interior spawns (cows by day 0.0052 → 0.0121);
  - the count gain (step 16: 0.603 → 0.718).

  The next step (another switching variant, P2 or from scratch) depends on which. A switching variant would be the same
  fine-tune, so it inherits the recipe's part.
- **Arm:** `bash run_arm.sh sw_rec2_ctl 4000 --substitute 0`.
  - Same script, optimizer continuation, seeds and readings as `sw_rec2`.
  - The mask is drawn but nothing is substituted. On the record route every G1 target is kept, so this is P1.5's
    denoiser trained on, alone, on identical draws.
  - Smoke: the first update's draw fingerprints must equal `sw_rec2`'s.
- **Attribution rule, per reading** (the four above, plus the S2 overdue-food curve, the edge arrivals and return):
  - The control moves less than 1/3 of the rerun's change from P1.5: the change is switching's.
  - It moves more than 2/3 in the same direction: it is the recipe's.
  - In between: unresolved.
