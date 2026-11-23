# Decision 6's place memory, the world's side, tested whole (2026-11-23)

**Why.** Stage 1 (`20261123_place_read/RESULTS.md`) established that the denoiser uses a correct old observation at
the correct cell: hard revisits 0.927 against 0.310 with the entry removed, and it follows the entry's content (a wrong
entry's class is drawn 0.935 against a base of 0.376). Its addresses were the simulator's.

This test asks whether Saccade finds the right observation by itself, with the architecture agreed for decision 6:
- learned self-location corrected by memory;
- entries of the world's own tokens with their context;
- reversible writes;
- a read keyed by content and relative location;
- dreams that advance by predicted motion.

It is what the user asked for: "test the full architecture, to derisk it". Only the agent's off-screen read is left
out, because no policy or value exists yet to teach it.

**Nothing the agent uses touches simulator labels.** Simulator positions and terrain are used only to score.

## The architecture (AGENT.md decision 6, re-read 11-23)

**Self-location: Markov localization on the agent's own grid** (Fox, Burgard & Thrun 1999).
- **The grid:** 96 × 96 cells at the sensor's resolution, in the agent's own frame, starting at (48, 48). Nothing is
  given about the world's size or position.
- **The belief** is a probability over the grid.
- **Prediction:** the belief is convolved with a learned motion prior p(d | current coarse view, lighting, command)
  over the 9 offsets within one cell. That is the same head dreams use. It is mixed with a learned relocation
  probability ε spread uniformly over the grid.
- **Update:** the new view's 63 map cells are matched against the memory, the previous frame included, under every
  grid hypothesis.
  - Each cell scores against the latest entry at its hypothesised address with a robust mixture: a Laplace density
    around the entry, whose scale and trust are learned functions of the entry's age and the current lighting, mixed
    with a background density.
  - A cell with no entry scores the background density.
  - Frame-to-frame odometry and recovery at a remembered place are therefore one rule.
  - Because every grid hypothesis is scored, a remembered view can propose a pose far outside the motion window
    (Codex 1).
- **Training:** label-free, on TRAIN episodes' coarse tokens and commands. It maximises the filter's marginal
  likelihood of each new view (truncated: the past belief is detached).
- **The match, amended 11-24 before the retrained localizer's results.** The first localizer failed reading 2:
  - lookups were 0.394;
  - on moved commands the estimate was 0.70 right after seeing the frame against 0.72 before it.

  The diagnosis is in `diagnosis_v1/`, with its code and outputs. Two measured causes, both biasing the match toward
  "no motion":
  - **The sensor's own response.** Classic's night static grows with distance from the screen centre (renderer
    lines 408-439). A cell's token therefore depends on where on the screen it was seen.
    - On dark moved steps the match alone was right 0.40; by day, 0.89.
    - The motion prior, trained against that posterior, learned "no motion at night": right on 0.38 of dark moves,
      against 0.93 by day.
  - **Coverage.** Cells with no entry scored the broad background. In uniform terrain, the hypothesis that covers
    more of the store won by about 19 nats per extra cell. The agent's own sprite, fixed at the screen centre, and
    its trail added about 30 more.
    - Offline, scoring absent cells by a spatial prior and dropping the sprite cells raised day moved steps from 0.89
      to 0.94. Darkness was unchanged by that (0.40 to 0.42).
- **The match now** (general, learned, label-free):
  - Each observation, new or stored, gets a learned token offset, scale part and trust part from its screen cell and
    its lighting. This is the sensor's own response, which online photometric calibration estimates as vignetting
    alongside odometry ([Bergmann, Wang & Cremers 2018](https://arxiv.org/abs/1710.02081)). It also covers content
    that moves with the sensor ([Godard et al. 2019](https://arxiv.org/abs/1806.01260), auto-masking).
    - The terms are per observation, never per pair. Otherwise "seen at the same screen cell" could itself become
      evidence of no motion.
    - The entry's age adds its own scale and trust parts.
  - Entries store the screen cell they were seen at (source context, like the fine entry's crop position). The read
    gets it too.
  - A cell with no entry is scored by its predictive density given its parent cell in the same view, one step toward
    the centre, with the centre cell scored by the background. Memory counts only as a likelihood ratio over that
    spatial prior, and the same density is the match's outlier component.
- Training, readings and bars are unchanged. The first localizer is kept as `run/localizer_v1.pt`.
- **The geometry, declared:**
  - which screen cells view the world (the 7 × 9 map cells), and their offsets, are the sensor's embodiment facts;
  - the status rows are not written to the place memory;
  - the denoiser itself is unchanged and treats all 81 cells alike.

**The entry, per grid cell (decision 6's write step, every field):**
- **coarse:** the token (16), write time, the pose confidence it was written with, the lighting at write time (8:
  the source context), a replay pointer (episode frame index), and provenance;
- **fine,** where the eye looked: the token (64), write time, the crop position within the 3 × 3 window (9: the source
  context), the coarse token at that time (16), confidence, lighting (8).
  - The doc's "marked stale when a newer coarse view contradicts it" is given to the reader as evidence instead: the
    fine entry's own coarse token beside the current coarse entry. The contradiction is learned, not thresholded.
- **Replacement:**
  - coarse: newest wins;
  - fine: replaced only by a newer glimpse of the cell.

**Reversible writes** (Codex):
- A frame whose pose confidence (the posterior mass at the most probable cell) is below 0.5 writes to a pending layer,
  attached to its source frame.
- When the pose relocalises (the most probable cell jumps beyond one move), the pending layer is moved by the
  correction.
- Pending writes are committed at the next confident frame.
- Reads see committed and pending entries, with a pending flag.

**The world's read** (decision 6, with Codex 2):
- **What it attends to:** each drawn map cell's input token queries the entries in the 3 × 3 neighbourhood of its
  predicted address, plus a learned null slot.
- **Keys and values** come from an MLP of each candidate's whole entry:
  - content: coarse and fine tokens, the fine entry's coarse-at-write;
  - context: lighting, crop position;
  - presence, both age codes (log age and decays at 8, 32 and 128 frames), write confidences, the pending flag and
    provenance;
  - the relative offset, and the current pose confidence.
- Four heads at width 64. The output projection is zero-initialised, so the read starts inert.
- **The address for the frame being drawn comes from the motion prior, before that frame is seen,** as one pose
  hypothesis for the whole view: the most probable cell of the predicted belief. Reading from the frame being
  predicted would leak it.

**Dreams** (decision 6's dream flow):
- the memory is copied at the dream's start (copy-on-write);
- each dreamed step advances the pose by the motion prior on the dreamed frame and the recorded command;
- the read happens before the draw;
- new places (no real entry) go to an overlay with provenance "dream";
- real entries are never overwritten, and the overlay is discarded at the end.
- Writes of changes caused by the dream's own actions are not in this test. Identifying "caused by my action" needs a
  learned change model, and it is listed as a limit.

## Training

1. **The localizer**, label-free, on 300 TRAIN episodes (seed 20261123): Adam at 1e-3, 4 passes; then frozen.
2. **The read**, by stage 1's recipe:
   - P1.5 checkpoint 10 plus the read; denoiser only, G1, AdamW 1e-4 with 100 warm-up updates, clip 1;
   - 64-step chunks of P1.5's own streams, uniform fixations, encoders frozen, stored tokens detached;
   - **2,000 updates**, for parity with stage 1's 2,000-update ceiling;
   - the memory is built online from the frozen localizer's own poses, so the read learns trust under real pose
     errors.

## Readings, fixed before results

On stage 1's evaluation panel: 120 DEV episodes, 40 movement starts each, 4 draws, S2's terrain reader,
episode-bootstrap 95% intervals.

1. **Real revisits with inferred addresses** (stage 1's cells and protocol).
   - **The gain:** class-matched revisit-minus-first in the surprising context. The stage 1 ceiling at 2,000
     updates is +0.570 at 2-16 and +0.569 at 17-64.
   - **Working:** at least +0.47 at both, within 0.10 of the ceiling. **Partial:** +0.20 to +0.47.
   - **Identity:** the wrong-address entry's class drawn at least +0.30 above its base rate (stage 1: +0.56).
   - **Fine only (Codex 3):** the coarse entries are held fixed and the fine fields are swapped for another cell's of
     the same age band. Reported, with no bar: Classic's terrain is almost all in the coarse tokens.
   - **Guards:** first sightings within 0.03 of P1.5 (0.341 / 0.829); G1 not rising.
2. **The localizer** (the panel's full episodes, scored with simulator positions):
   - per-step displacement accuracy before the frame (prior) and after it (posterior) (Codex), on moved and
     blocked movement commands and in darkness (light < 0.5, reported only);
   - exact relative pose over 16 steps;
   - correct revisit lookups: the entry at the inferred address is the same world cell.
   - **Bar:** correct lookups at least 0.95.
   - **Reference:** S2's simulator-trained shift reader with the action constraint, recomputed on this panel.
3. **Return after an injected pose error beyond the local window** (Codex 1).
   - At movement steps that precede a revisit within 32 steps, the belief is shifted by 4 cells, beyond both the
     motion window and the read's neighbourhood.
   - **The last frame's writes move with it** (amended 11-23, before any result). A real drift is self-consistent:
     the recent entries share the error. Without this, the previous frame's correct entries would undo the injection
     at the next frame, odometry included, and the control couldn't separate anything. Only older memory can reveal
     the error.
   - **The repair reading applies to doubtful writes only.** Writes during a self-consistent drift are confident, so
     the agreed rule commits them; reversibility covers doubtful writes (Codex's "uncertain writes").
     - The repair bar is read on natural relocalisations in the panel: pending entries aligned after the jump.
     - Confident drifted writes are reported as the known limit. Repairing them needs pose-graph loop closure, which
       the design doesn't have yet.
   - **Three localizers on the same trajectories,** with the store and read fixed:
     - oracle (the injection undone);
     - odometry only (the memory term limited to the previous frame);
     - corrected against memory (the full rule).
   - **Bars:**
     - **recovery:** the full rule restores the exact relative pose within 16 steps in at least 0.80 of injections;
       odometry only is the separating control;
     - **the following view:** on the first revisit after recovery, agreement in the surprising context at least
       0.8 of the oracle's gain over first sightings;
     - **pending writes repaired:** after a natural relocalisation, pending entries from the run before it are
       aligned with the corrected pose in at least 0.90 of cases;
     - **confident drifted writes:** their alignment after recovery is reported (expected to fail by design).
4. **Dreams** (Codex 3): 16-step self-fed dreams under recorded commands from real starts (the panel's starts).
   - **Scored on world cells entering view** that the real prefix verified (a clear sighting) and that the
     simulator says are unchanged at that step: does the dream draw the verified class?
   - Stratified by gap (≤ 16, 17-64) and by whether the dream's own pose is right (motion error reported
     separately).
   - **Arms on the same starts and seeds:**
     - the same weights with an empty memory (no recall);
     - memory with oracle dream poses;
     - memory with inferred dream poses.
   - **Bar:** with inferred poses, where the dream's pose is right, agreement at gap ≤ 16 at least 0.85 (the
     simulator through S2's readers is 0.94), and at least +0.20 above the empty memory.

**Working memory, as the user asked, means readings 1, 2, 3 and 4 all pass.** Anything less is reported with where
the chain broke: localization, retrieval, or use.

## Limits, declared

- One seed.
- A denoiser fine-tune, not a world trained with the memory from scratch.
- Encoders frozen.
- Uniform glimpses, not the eye.
- No agent read.
- No writes of changes caused by the dream's own actions.
- The grid hypothesis space is Classic's sensor resolution. A robot's version is continuous poses with particles,
  behind the same interface.
