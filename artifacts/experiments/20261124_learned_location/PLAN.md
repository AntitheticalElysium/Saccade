# Decision 6's locked design, tested on the world's side (2026-11-24; plan for review, nothing built or run)

**Amended 11-24 after Codex's review (each point checked) and the camera-model check (`eda_camera.py`).** The
changes:
- commuting rotations for the code;
- a camera model learned from the agent's own data, as an option for the user's decision;
- diversified shortlists;
- reading 3's arms isolated;
- a recall bar;
- paired dreams;
- pilot gates before the GPU fine-tune.

**Why.** Decision 6's design was locked 11-24 (`saccade/design/AGENT.md`, "The design, locked 11-24"). This plan tests
that design. Where it has to choose something the design leaves open, the choice is listed with its reason, so the
review can change it. The grid localizer (`20261123_located_memory/`) is the Classic baseline, not the design.

**What has to be shown.** The chain the design exists for, measured on one return:
1. With its motion sense, the agent's own position code finds the right remembered entries, and the world uses them
   (stage 1's gain, with the agent's codes instead of the simulator's).
2. When the motion sense is missing for a stretch, the code drifts. An old view is then recognised, under changed
   light and among lookalikes, and corrects the code.
3. The correction repairs the entries written during the drift. That repair improves a later, unseen view beyond what
   correcting the current code alone achieves (Codex, 11-24).
4. The repaired memory makes later dreams of that place more faithful.

**Nothing the agent uses touches simulator labels, except the declared motion sensor** (below). Simulator positions and
terrain otherwise only score.

## Why the motion sense needs outages (the user's question, 11-24)

With an exact motion sensor, Classic's awake code never needs correcting, so recognition and repair would get nothing
to learn from. Three conditions keep them necessary:
- **Dreams have no motion sense.** The motion is predicted there.
- **The learned code can drift** even when the motion is exact (Codex). Its transition is learned, and TEM-t notes
  that "recurrently generated positional encodings accumulate noise and drift".
- **Real motion senses fail and drift.**
  - IMU biases are modelled as random walks: "assumed to evolve according to a Brownian motion" (ORB-SLAM3).
  - Odometry alone turns the world into an "infinite corridor" (Cadena et al. 2016).

The choice here: **the motion sense is withheld in stretches ("outages")** during training and in the readings. During
an outage the agent integrates its own predicted motion, as it does in dreams.
- So the drift comes from the agent's own motion errors (blocked moves, darkness), not from an invented noise model.
- The withheld sense is still logged, so it can be the target that teaches the estimator to recover. That is
  self-supervision by masking an input the agent owns.
- Removing the sense entirely, as the grid baseline did, is the camera-only case. It is kept as an evaluation
  condition, not a training regime, because it leaves recognition without a teacher.

## The design as built (each part, its source; choices marked)

**Inputs.**
- P1.5 checkpoint 10's frozen perception:
  - coarse tokens of all 81 screen cells (16 each);
  - glimpse tokens;
  - the lighting latent (8).
- The command.
- **The motion sensor (declared):** the displacement since the last frame, as a vector δ_t (in Classic, (Δrow, Δcol)
  from simulator positions), or "withheld". A robot's equivalent is odometry; an animal's, its vestibular and
  proprioceptive senses. Its values are numbers, not classes, so the same interface takes a robot's continuous
  odometry.

**The motion model (learning how movement works).**
- It predicts δ_t from the view tokens, the lighting and the command, before the frame.
- **The form:** a mixture density network over δ (Bishop 1994). **Choice:** 4 components. Classic needs at least two
  (moved or blocked), and the extra components leave room. Dreams sample one δ: one coherent motion for the view.
- **Its teacher is the motion sense:** negative log likelihood whenever the sense is present. That is GTM-SM's "the
  agent feels its movement" (displacements as a regression target, never an input), with our own sensor as the
  target.

**The position code.**
- g ∈ C^(D/2): D/2 phases. It is path-integrated as g_t = g_{t-1} M(δ_t), where M(δ) multiplies phase j by
  exp(i ω_j · δ), with learned frequency vectors ω_j ∈ R². These are commuting rotations with learned frequencies
  (amended 11-24, Codex).
  - Codex checked the first form, two general skew-symmetric generators. With it, "east then north" and "north then
    east" reach different codes, and even a closed four-step loop doesn't return. That is because exp(A)exp(B) ≠
    exp(B)exp(A) unless A and B commute.
  - Classic's motion is translation, whose group commutes. So its representation must commute: then closed loops
    return exactly and any order gives the same code.
  - This is Gao et al.'s earlier block-diagonal form, and a learned grid-cell-like code.
  - A robot that turns needs a representation of SE(2), which does not commute. That is outside this test.
  - That is Gao et al. 2019's "vector representation of self-position coupled with matrix representation of
    self-motion", which they report "capable of error correction, path integral". It is also TEM's learned
    action-dependent transition.
  - M is unitary, so codes don't grow or shrink.
  - **A correction is itself a displacement δ̂, applied through the same M,** so the corrected code stays one the
    motion can reach. Spreading it back along the trail spreads a displacement.
  - **Smoke tests before any training run:** closed paths return; equivalent paths in different orders agree; cell
    offsets compose with motion; retrieval is unchanged when the arbitrary origin moves.
- δ_t is the sensed motion, or the predicted one during outages and in dreams.
- **Choice: D = 64 (32 phases).**
  - TEM's location code totals 120 units over 5 modules, for worlds of at most 127 states.
  - Ours span up to Classic's 64 × 64 world and must stay distinct across an episode.
  - The reading on codes (reading 2a) checks that 64 suffices. Codex agrees it is a reasonable start.
- **How the code's geometry is taught** (the agent's own motion, in place of Gao et al.'s true positions):
  - their adjacency kernel: code similarity between two frames tracks the distance of their integrated sensed
    displacement;
  - loop consistency: frames that the integrated motion says are at the same place get the same code (Codex).
- **A cell's code:** c_k = g_t M(o_k). o_k is screen cell k's offset in motion units, from the camera model.
- **The camera model (open; the user decides, 11-24).**
  - **Proposed: learned from the agent's own data.**
    - At stage A's start, each screen cell's offset o_k and a world-anchored weight a_k are estimated from token
      correspondences under sensed motion on TRAIN episodes. That is Pierce & Kuipers' (1997) problem of learning an
      uninterpreted sensor's geometry.
    - A cell whose best match shifts with the move shows the world: its o_k is solved by least squares, and a_k = 1.
    - A cell whose best match doesn't shift is fixed to the sensor (the status rows, the player's sprite): a_k = 0,
      and it has no world projection.
    - The check (`eda_camera.py`, panel episodes 0-22):
      - every map cell matched one move back (212 of 212 pairs, by day and in the dark);
      - the offset table recovered exactly;
      - all 18 sensor-fixed cells flagged. Blank cell 80 ties its blank neighbour 79, which doesn't shift either.
  - **Alternative (Codex):** declare Classic's top-down projection as the sensor model, with the status rows marked as
    having no projection. ORB-SLAM3 and DROID-SLAM take calibration as given; MASt3R-SLAM assumes only "a unique
    camera centre".
  - **Either way, there is no mask.** Every token is still written, kept in the source record, and read by the world
    and the belief. Tokens with no world projection carry no evidence of place, and get no same-place supervision
    (Codex: under the first plan, 34.0% of genuine first sightings found an earlier status-row token at their
    assigned address).

**Entries (anchored).**
- Every frame writes every screen cell, all 81. There is no mask. A cell with no world projection (a_k = 0) is
  written to the frame's source record, not to a place.
- **The coarse entry:** token, time, context (lighting, screen cell, crop position), write confidence, replay pointer,
  source frame id, and offset o_k.
- **The fine entry, where the eye looked:** the glimpse token, with its own time and crop position.
- **An entry's code is computed from its source frame's current code,** c = g_frame M(o_k), so repairing a frame moves
  all its entries.
- **Choice: write every frame.** The design writes whenever an entry adds evidence; writing everything is the
  generous end of that (ORB-SLAM: "generous in the spawning"). A Classic panel episode averages 411 steps, so about
  33,000 entries, which fits. Keyframe selection and pruning wait for a capacity trigger, as decision 6 already says.

**The trail, recognition, correction, repair.**
- **Recognition key.** A learned key κ(token, context) per cell.
  - **Its teacher (direct, Codex):** a contrastive loss over memory.
  - Positives: world-anchored entries that the integrated sensed motion places at the same world position as the
    current cell, at any age or light. Returns across lighting changes are included (Codex: 81% of daylight revisits
    at 17-64 steps cross a change of at least 0.02).
  - Negatives: other entries, including hard negatives with the most similar raw tokens at other places (the
    lookalikes).
  - NetVLAD trains place recognition the same way, from views of a place "taken at different times", with only
    approximate GPS as supervision.
- **Candidates.**
  - After frame t arrives, each world-anchored cell retrieves its top 8 entries by κ (**choice**). There is at most
    one per source frame and one per proposed place (amended, Codex).
  - Entries from the last 16 frames are excluded (**choice**), as ORB-SLAM excludes keyframes already connected to
    the current one.
  - Each match implies a code for the agent: the matched entry's code transported back by the current cell's offset,
    c_entry M(-o_k).
  - The consensus is the code most cells agree on: one mean-shift step from the best-supported candidate.
- **Correction:** g_t ← g_path + γ (ĝ - g_path), TEM-t's form.
  - The gate γ is learned. It sees the agreement (how many cells support ĝ, and how strongly), the distance from
    g_path, the lighting, and whether the sense is present.
- **Acceptance over frames.** A correction larger than the read's similarity width is applied only after 3
  consecutive frames support the same correction once their own motion is accounted for. ORB-SLAM3 verifies in three
  covisible keyframes, and FAB-MAP needs "two or three".
- **Repair.**
  - The frames since the last confident anchor are re-estimated by least squares over the trail:
    Σ_τ w_τ ||g_τ - g_{τ-1} M(δ_τ)||² + λ ||g_t - ĝ||².
    - g_anchor is fixed.
    - The weight w_τ is high where δ_τ was sensed and low where it was predicted.
  - So the correction is spread back over the uncertain steps.
  - This is SLAM's back end on our codes: Cadena et al. describe it as "a nonlinear least squares problem" over
    factors, and ORB-SLAM's pose graph "distributes the loop closing error along the graph". With orthogonal M it is
    linear, a chain solvable in one pass.
  - Entries move with their frames.
- **Lost:**
  - The agent is lost when the predicted-motion uncertainty accumulated in an outage exceeds the read's similarity
    width.
  - From then on, it writes into a fresh trail with no link to the old one.
  - A recognition later joins the two, and the repair re-places the fresh trail (ORB-SLAM3's Atlas).
- **Choice: the gate is trained on outages,** with the withheld sense's code as the target: the code error after
  correction, with false corrections on lookalike returns penalised. The causal check is reading 3.

**The world's read.**
- Each drawn cell's token queries entries at its code before the frame, c_k = g_t M(o_k), with g_t predicted by the
  sense (or the motion model).
- **Retrieval: 25 candidates** (**choice**: the agreed read reached within two cells, 5 × 5 = 25), diversified.
  - Codex counted that with write-every-frame, 51.3% of revisit queries already have at least 25 earlier sightings of
    the very cell queried, so a plain top 25 can be all duplicates. Checked on all 120 episodes.
  - So candidates are grouped by proposed place at the read's similarity width. The 5 best places each give their 5
    newest distinct source frames.
  - Then attention over them, plus a learned null slot.
- **The keys:** an MLP of the whole entry plus relative-position features. Per phase, the query's conjugate times the
  entry gives the cosine and sine of their phase difference, so the read sees relative position exactly. (Raw
  elementwise products don't, Codex.)
- 4 heads at width 64, output zero-initialised: located_memory's read.

**Dreams.** Copy-on-write overlay. The motion is sampled from the motion model, one per step for the whole view. Real
entries are never overwritten.

## Training

**Stage A, location (label-free but for the declared sensor):**
- The data: located_memory's 300 TRAIN episodes, tokens cached, plus their S0 positions for the motion sense.
- Trained together: the motion model, the generators, the recognition key, the gate.
- Outages: in each 64-step window, with probability 0.5, the sense is withheld for 8-48 steps (**choice**: revisits'
  median gap is 20 steps, OPTION3_NOTES, so outages must span typical returns).
- 4 passes, Adam 1e-3, as the localizers trained.

**Stage B, the read:** stage 1's recipe.
- The location modules are frozen.
- P1.5's denoiser plus the read are trained; everything else is frozen; G1 only.
- AdamW 1e-4, 100 warm-up updates, clip 1, 64-step chunks of P1.5's streams.
- The memory is built online from the agent's own codes, with the same outages.
- 2,000 updates (stage 1's ceiling is read at 2,000), about 3 h of GPU.

## Readings, fixed before results

Panel: stage 1's 120 DEV episodes and starts, 4 draws, S2's readers, episode-bootstrap 95% intervals. One seed.

| Reading | What it measures | Bar | Competing explanation and its control |
|---|---|---|---|
| **1. Revisits, sense present** | Stage 1's class-matched revisit-minus-first gain in the surprising context, with the agent's codes | Within 0.10 of stage 1's +0.570 (2-16) and +0.569 (17-64) | More training, not memory: first sightings; the empty-memory swap |
| **2a. Codes, sense present** | Right entry among the read's 25 candidates on revisits | ≥ 0.95 | — (a component check) |
| **2b. Recognition** | On drifted returns: the precision of accepted corrections (the code moves toward the truth) and recall within 16 steps. Missed returns are counted. Split by light change since the entry (< 0.02, 0.02-0.1, ≥ 0.1), by day and dark, and by gap | Precision ≥ 0.95 and recall ≥ 0.35 in every band with at least 30 returns. These are FAB-MAP's and DBoW2's operating points: at 100% precision, 48% / 37% recall (FAB-MAP), 30-40% (DBoW2) | Matches only at steady light: the split. Lookalikes: hard-negative returns counted in precision. A matcher that fixes one easy return: the recall bar (Codex) |
| **3. The causal chain** (central) | Outages of 8, 16 or 32 steps before returns to old memory, chosen as the baseline's reading 3 chose events. On the same trajectory and weights: no recognition; the current code corrected only; the current code corrected plus repair. Two references: the clean-sense code (the sense never withheld) and true alignment (simulator, scoring only). **The current-only and repair arms share the same accepted match and the same corrected current code;** repair's endpoint is fixed, so only earlier anchors move (Codex). Scored on a later, unseen view's cells first seen during the outage and revisited after the return | Repair beats current-only by ≥ +0.10, and reaches ≥ 0.5 of the clean-sense reference's gain over no recognition | The sense's return fixes it: the "no recognition" arm. Snapping the pose suffices: the current-only arm. Too few cells means the reading is unmeasurable, said so |
| **3b. Declining** | Returns to mined lookalikes (another place, most similar view) | Wrong corrections accepted ≤ 0.05 | — |
| **4. Dreams** | 16-step dreams from real starts, motion sampled, scored by verified-terrain agreement at revisits. **Paired:** starts after an outage-and-return, repaired against unrepaired memory, with the same motion draws and seeds, so only memory differs (Codex). Free-running performance across all starts is also reported | ≥ 0.85 at ≤ 16 steps where the dream's motion is right, and ≥ +0.20 over empty memory. The paired repair effect is reported with its interval | Empty memory; motion errors reported separately |
| **5. One-step motion cost** | During outages, with the reader frozen: true against predicted motion for the read's code (Codex) | Reported | — |
| **6. Identity and fine swap** | The donors are logged from the entries actually injected, and the fine swap holds the fine entry's age and metadata. On plausible, metadata-matched donors: the wrong class drawn. On inconsistent donors (code far from the queried place): how often they are rejected, i.e. not drawn (Codex) | Plausible donors: wrong-class ≥ base + 0.30 (stage 1: +0.56). Inconsistent donors: reported | — |

**Guards:** first sightings within 0.03 of P1.5 (0.341 / 0.829); G1 not rising; the grid baseline's readings beside
these, same cells.

**What the outcomes rule out:**
- Reading 1 failing with 2a passing localises the fault to the read under learned codes.
- 2b failing localises it to recognition.
- Reading 3 with repair no better than current-only means anchoring doesn't pay at this scale.
- Reading 4 says whether awake repair reaches imagination.

## Pilot gates (amended, Codex): stage B only if stage A passes

Stage A runs first, then its gates are read on held-out TRAIN episodes and on the panel's episodes 0-22. Stage B,
the 3 h denoiser fine-tune, is released only if all four hold:
1. the code's smoke tests pass (closed paths, order, offsets, origin);
2. with the sense, the right place is among the diversified candidates on ≥ 0.95 of revisits (reading 2a's bar);
3. recognition meets reading 2b's precision and recall bars on held-out returns;
4. the outages produce at least 200 drifted returns, with drift beyond the read's width, on the panel.

If a gate fails, the run stops and the failure is diagnosed before anything changes.

## Limits, declared

- One seed.
- A denoiser fine-tune, not a world trained with the memory from scratch.
- Encoders and Mamba frozen.
- Uniform glimpses, not the eye.
- No agent read.
- The camera model is learned from the declared motion sensor, or declared (the user's call). Classic's top-down
  render makes each cell a fixed offset. A real camera gives a ray, not a ground point (Codex), so a robot would
  learn a ray and depth instead.
- The motion sensor is exact in Classic, so outages are what make the corrective machinery necessary.

## Build and cost

- One module beside `memory.py`, reusing its read installation, panel, one-step and dream code: the motion model,
  codes, entries, recognition, gate, smoother, and stage A and B training.
- Smoke on 8 episodes before any run.
- GPU, one job at a time: stage A about 1-2 h; stage B about 3 h; readings about 2 h.
