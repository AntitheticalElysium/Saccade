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

Stage 1's bank, 1,000 starts x 4 futures. In free running three routes read the generated frames: the denoiser's
context, Mamba and the record. Four modes feed the recorded frame instead, while the recording lives: none (free),
the context only (ctx_real), Mamba and the record only (mem_real), all (teacher); the scored frame is always the
generated one. Read: mob counts by step against the simulator's.
**Rule for the first arm's routes, fixed now:** switch the routes whose recorded input recovers at least a third of the
free-running step-16 deficit (teacher minus free); both if each does, or if only teacher does.

## The first arm (`switch_ft.py`)

- **Protocol:** that of the earlier denoiser fine-tunes (traces, SWAX, record).
  - From P1.5 checkpoint 10 with its streams, carry, cache and record state; the denoiser alone trains.
  - G1 alone (shortcut forcing, bootstrap rows included); AdamW 1e-4, 100 warm-up updates, clip 1.0.
  - 64-step chunks, 16 streams; the checkpoint's chooser picks the fixations (frozen); 3,000 updates.
- **Switching:** each chunk frame, with p = 0.5 independently (JEDI's uniform choice), is replaced on the chosen routes
  by the denoiser's own one-step estimate of it.
  - The estimate comes from an independent noise draw at an independent flow level, conditioned on the recorded past
    and memory, and is clamped.
  - Targets are always the recorded frames. The carry, cache and record passed to the next chunk come from the
    recorded frames.
  - In this single-stream diffusion-forcing world, a frame's noisy tokens are both its own denoising input and later
    frames' context. So on the ctx route a switched frame's own target is reached from a noised estimate: a
    self-correction term (declared; JEDI's frame-by-frame training does not have it).
- **Readings, at 1,500 and 3,000 updates:**
  - Primary: S2 stage 1 against P1.5 on the same bank and simulator futures: mob counts and 4-step persistence by
    kind, the bookkeeping blocker, mobs by step.
  - Guards: the G1 monitor (P1.5's 2,000 held-out windows), S2's L2b clocks and danger, first-step consequences,
    internal consistency, N5; stage 1b on competent starts, once it exists.
- **Decisions:**
  - If the mob deficit shrinks with the guards intact, the control arm (the same fine-tune, p = 0) runs next, to separate
    switching from extra G1 training.
  - If it does not shrink, the variants below come before P2.
- **Cost:** about 3 s per update (one extra no-gradient denoiser pass and a belief pass), so about 2.5 h, plus readings.

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

## Limits

One seed; a fine-tune, not a from-scratch run; the readers read generated frames off their domain (TEST_SPEC).
