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
- **Amended again 11-24, before the third localizer's results.** The second localizer (`run/localizer_v2.pt`;
  diagnosis, code and outputs in `diagnosis_v2/`) solved daylight but not the dark:
  - Daylight moved steps: 0.976 on the match alone, 0.969 after the frame (3 episodes).
  - Dark moved steps: 0.49 on the match alone. On the panel, lookups were 0.434.
  - **The cause.** Trust caps what a mismatch costs, but it can't remove a bonus. Under "no motion", content fixed
    to the sensor *matches*: the agent's sprite, and the night static's pattern. At night that matching decides
    alone (the centre cell gives +27.5 nats, its neighbours +12.9 and +6.6), and the objective doesn't remove it.
  - **The amendment.** The null that memory evidence is measured against gains one component: the same screen cell
    in the previous frame. It is Godard et al. 2019's auto-masking (evidence only where the warped source beats the
    unwarped one) in probabilistic form.
    - Memory evidence is now the improvement of the world-fixed explanation over the spatial prior and the
      sensor-fixed explanation together. Whatever moves with the sensor is explained equally under every hypothesis.
    - Its scale and trust come from the same per-observation terms, plus a learned offset of its own. The mixture
      weights are learned.
  - **Reading 2 also reports panel episodes 23-119.** No diagnosis or code check touched these (the diagnoses read
    episodes 0-11, and the code checks 0-2 and 20-22).
- **Amended a third time 11-24, before the fourth localizer's results.** The third localizer (`run/localizer_v3.pt`;
  diagnosis, code and outputs in `diagnosis_v3/`) failed reading 2:
  - lookups 0.345 (0.353 on panel episodes 23-119);
  - on moved steps, 0.750 after the frame against 0.764 before it; dark 0.584;
  - 3,829 natural relocalisations, with 0.1% of their pending writes landing aligned.

  Two measured causes:
  - **The sensor-fixed component's own scale collapsed onto exact repeats.** On static day frames, 12-15% of cells
    repeat exactly, and a mixture component with a free scale fits them (Bishop 2006, section 9.2.1).
    - Its log-scale reached -8.3 by day; memory's age-1 entry sits at -4.6.
    - On the agent's own cell after a move (median change 0.30 L1), it scored -1,025 nats, against memory-under-"no
      motion" +34. So the "no motion" bonus survived: +34.5 nats by day, +27.8 in the dark. 90% of wrong dark matches
      still said "no motion".
  - **Indistinctive content made mapped poses lose to empty grid.** 52% of the jumps landed on a pose with no entries
    (score exactly 0), while the true pose scored -23.3 (median).
    - The negative part comes from cells the spatial prior explains: uniform terrain, whose neighbours in the same
      frame repeat exactly. There memory is 66 nats below the prior, and each present cell pays log(1 - trust).
    - A present cell therefore costs what an absent one doesn't.
  - **Correction to the second amendment.** The night static is drawn anew every frame. Only its strength by screen
    position is fixed.
- **The amendment:**
  - **The sensor-fixed explanation is scored by memory's own function:** the previous frame's token as an age-1 entry
    seen at that screen cell. Only its mixture weight is its own.
    - Godard et al.'s auto-masking compares the unwarped and warped sources under the same error. With a scale of its
      own, the third localizer's component broke that.
    - Under "no motion", memory and the sensor-fixed explanation are now the same density. Checked: they are
      identical.
  - **Memory's trust gains a part from each observation's own content:** a small network on the token, zero at the
    start. It enters memory's trust only, not the null's weights.
    - The marginal likelihood can then learn that content the null already explains is little evidence of place:
      "identical but indistinctive observations receive a low probability of having come from the same place"
      ([Cummins & Newman 2008](https://doi.org/10.1177/0278364908090961), abstract).
  - Training, readings and bars are unchanged. Reading 2's panel episodes 23-119 stay unseen: the third localizer's
    diagnoses read episodes 0-2.
- **The fourth localizer, 11-24** (`run/localizer_v4.pt`; diagnosis, code and outputs in `diagnosis_v4/`). It removed
  the "no motion" bias: the agent's cell went from +34.5 nats to +0.1. On 3 episodes, moved steps on the match alone
  were 0.973 by day and 0.854 in the dark (the third localizer: 0.55 in the dark). But it fails reading 2 worse:
  - lookups 0.120; blocked steps 0.276 after the frame; 7,083 relocalisations.
  - On blocked dark steps the match never says "no motion" (0.0 of 61). The commanded move's memory gains +90 nats on
    the cells where it beats the null, against +18 for "no motion", with all 63 cells present under both.
  - **The cause.** Under "no motion", memory is the same density as the sensor-fixed component in the null. So its
    gain per cell is capped, at a fraction of a nat. A wrong hypothesis's costs are capped too, at log(1 - trust)
    (about -0.29 here), but its gains are not. On the 29% of dark cells where noise happens to favour the shifted entry
    (median log density 6.4 for the sensor-fixed explanation against -32 for the shifted entry), sharp scales turn
    small residual differences into tens of nats.
  - Godard et al.'s auto-masking assumes a camera that moves, and removes the evidence of a static one. Here, "did I
    move?" is the question.
  - The content trust didn't separate uniform cells from the rest (median -0.97 against -0.90). The learned
    per-observation offset is the same at every screen cell (0.81). The sensor-response terms of the first amendment
    were never learned in any version.
- **Decided 11-24 at 06:20, before any of readings 1, 3 and 4.** The read's fine-tune and readings 1, 3 and 4 run with
  the localizer that scores best on reading 2's bar measure, the second one (`run/localizer_v2.pt`, lookups 0.434), on
  its own pinned code. `diagnosis_v2/memory_v2.py` has the same training and reading functions as `memory.py`.
  - The memory as a whole therefore fails reading 2, so it is not "working" by this PLAN.
  - Readings 1, 3 and 4 show where the chain breaks beyond localization: whether the read tolerates these pose errors,
    the return, and dreams.
  - The next localizer is designed in parallel, from the four measured failures, and scored on reading 2 alone.
- **The fifth localizer, amended 11-24 before its reading 2.** The code is `memory.py`; the development checks and
  their outputs are in `diagnosis_v5/`.
  - **What the four failures share.** In each, a scale, a null or a sensor term was learned inside the mixture that
    judges with it, and collapsed or miscalibrated there.
  - **The design:** the match becomes closed-form, from the agent's own TRAIN tokens, lighting and commands (no
    simulator labels):
    - Each screen cell's noise scale is linear in the lighting latent. It is fitted as one Laplace on the change after
      a command that doesn't move the agent (the agent knows it didn't move). A single component can't collapse.
    - **No token offset.** By day the same world cell gives the same token one screen cell over (median change 0.165,
      against 0.15 at the same cell, 20 episodes, `equivariance.py`). A per-cell offset fitted by least squares broke
      that (1.06). Dark: 1.18 against 0.90.
    - **Binary evidence, as FAB-MAP's observations are** (Cummins & Newman 2008):
      - A cell matches its entry when their change is within the sensor's repeatability, the 90th percentile of the
        static change in units of the pair's scale. So P(match | same content) = 0.9 by construction.
      - P(match | new place) is the share of 512 sampled TRAIN observations it would match: the content's frequency
        (their section 4.3.2).
      - Evidence is log(0.9 / p_new) for a match and log(0.1 / (1 - p_new)) for a miss. It is bounded, and weak where
        the content is common.
    - Local hypotheses are compared on the cells all nine can predict, so coverage can't favour any of them.
    - **The agent's own cell is declared to show its body, not the world.** It is neither matched nor written, like
      the status rows.
    - Only the motion prior and relocation are trained, as before.
  - **The development checks** used the untrained match, a map built at the true poses, and panel episodes 0-2 (the
    diagnoses' episodes). On moved daylight steps:

    | Form | Moved, day |
    |---|---|
    | Laplace density, per-cell offsets | 0.756 |
    | Without offsets | 0.887 |
    | Plus a dead zone | 0.889 |
    | **Binary evidence** | **0.995** |

    - The Laplace forms lost to single rare-content cells: +101 to +186 nats against about +3 for ordinary cells. A
      sharp 16-dimension density made the per-cell evidence a lottery.
    - The binary form also has blocked day 0.982, moved dark 0.798 and blocked dark 0.951.
  - Reading 2 then runs on the full panel with the bar unchanged, plus episodes 23-119 (unseen by every check).
  - **Decided 11-24 at 06:35, before the fifth localizer's reading 2:** the 06:20 rule applies (the read runs with
    the localizer that scores best on reading 2's bar measure).
    - If the fifth localizer's lookups beat the second's (0.434), the second localizer's fine-tune is stopped. The
      read is then retrained, and readings 1, 3 and 4 rerun, with the fifth, by the same recipe and code.
    - Otherwise the second localizer's run continues.
  - **Applied 11-24 at 07:31:** the fifth's lookups were 0.4378 against 0.4343. The second's fine-tune was stopped at
    update 800, and the read retrains with the fifth.
- **The sixth localizer, amended 11-24 at 08:05, before its reading 2.** The code is `memory_v6.py`: `memory.py`
  patched in its own process, so the fifth's read fine-tune and readings, which import `memory.py`, are untouched.
  - **The fifth's measured failure** (`RESULTS.md`, `diagnosis_v5/`, episodes 0-22 only):
    - by day the trail held (moved posterior 0.968), but 32% of lookups found their cell stored six or more cells
      away: no loop ever closed;
    - P1.5's coarse tokens of a world cell change with daylight. Two daylight sightings match 0.94-1.00 at a steady
      light, at every gap and screen shift, and 0.00-0.14 after a change of 0.1 or more. The match's tolerance came
      from consecutive frames.
  - **The change: the match's metric only.**
    - A token and an entry match when their Mahalanobis distance under the within-place scatter is below its 90th
      percentile on the pairs that define it. That's the same construction as the fifth's threshold, so P(match |
      same content) = 0.9 as before.
    - The within-place scatter is the change covariance of the same screen cells across the agent's own runs of
      non-movement commands (TRAIN episodes, label-free; 3.3 million pairs). It is the scatter Fisher's discriminant
      discounts, which Fisherfaces project away to recognise across lighting (Belhumeur, Hespanha & Kriegman 1997,
      TPAMI 19(7), DOI 10.1109/34.598228).
    - Hits and the bank's new-place rate both use it. Everything else is the fifth's: the evidence form, the common
      support, the agent's own cell, relocation. The motion prior and relocation are retrained by the same recipe.
  - **The development check** (`diagnosis_v5/metric_chain.py`): the fifth's trained prior with the metric swapped, the
    real filter, episodes 0-22.

    | | The fifth's metric | The within-place metric |
    |---|---|---|
    | Posterior, moved day / moved dark | 0.968 / 0.688 | 0.993 / 0.911 |
    | Posterior, blocked day / blocked dark | 0.972 / 0.953 | 0.979 / 0.942 |
    | Lookups | 0.429 | 0.569 |
    | Cell stored six or more cells away | 0.316 | 0.088 |

    Per cell (`within_scatter.py`), at a daylight change of 0.1 or more: same-cell matches 0.018 to 0.864, chance
    matches 0.010 to 0.345.
  - Reading 2 then runs on the full panel with the bar unchanged, plus episodes 23-119 (unseen by every check).
  - **Decided before its reading 2:** the 06:20 rule applies again.
    - If the sixth's lookups beat the fifth's (0.438), the fifth's read fine-tune is stopped. The read is retrained,
      and readings 1, 3 and 4 run, with the sixth, by the same recipe and code (`memory_v6.py train`, then `readings
      2000`).
    - Otherwise the fifth's run continues.
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

**Amended 11-24, before any of readings 1, 3 and 4 at update 2,000 (code only, no bar changed):**
- **Why:** the first attempt ran out of GPU memory, holding all 120 episodes' read features at once
  (`readings_v5.log`). The 8-episode smoke never reached that size.
- **The fix:**
  - reading 1 processes one episode at a time, with features on the CPU, and the sampler's seed set per episode;
  - reading 3 keeps only the frames its following view needs, and builds features only from P frames before each
    event (the rerun after a machine restart ran out of host memory on a long episode, 11-24);
  - reading 4 keeps the real prefix's features on the CPU, and no longer converts the terrain reader's output, which is
    already an array (it crashed at its first drawn step; the smoke had never run reading 4).
- **One correction, to match this PLAN's "on the first revisit after recovery"** (Codex, 11-24): reading 3's
  following view is the first return step strictly after the frame at which the full rule recovered. The earlier code
  could score the very frame that made the correction, whose read was computed before it.
- **Known limits, reported with the results** (Codex, 11-24, checked):
  - reading 3's "oracle" arm is the localizer with no injection, so it inherits the localizer's own drift; there is no
    true-alignment arm;
  - the swap's identity labels assume the entry three rows away holds the true cell three rows away, which an
    inferred map doesn't guarantee;
  - the fine swap also moves the fine entry's age and metadata.
- The readings run with the fifth localizer, which the read was trained with. Since the 11-24 design decision they
  measure the grid baseline, not the design (AGENT.md decision 6).

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
