# Does P1.5 remember places beyond the frame? Results (2026-11-23, decision 6)

Plan and code: `PLAN.md`, `place_memory.py` (commits da11ccc5 and the dtype fix). Outputs: `place_memory.json`,
`cells.npz`.
- Frozen P1.5 checkpoint 10, 120 DEV episodes, about 40 movement starts each.
- One-step dreams from real history with 4 draws, and 35,170 entering terrain cells.
- Agreement is S2's terrain reader on the dreamed frame equalling the same reader on the real frame. "Surprising":
  the cell's class differs from its inward neighbour's, so context doesn't give it away.

| Agreement (95% episode interval) | First sighting | Revisit, gap 2-16 | 17-64 | 65-256 | > 256 |
|---|---:|---:|---:|---:|---:|
| Surprising context | 0.341 [0.329, 0.354] (n 4,425) | 0.347 [0.331, 0.363] (2,529) | 0.325 (1,497) | 0.323 (828) | 0.389 (142) |
| Predictable context | 0.828 [0.820, 0.835] (12,295) | 0.828 (6,785) | 0.834 (4,198) | 0.808 (2,092) | 0.848 (379) |

| Class-matched revisit minus first sighting | 2-16 | 17-64 | 65-256 | > 256 |
|---|---:|---:|---:|---:|
| Surprising context | +0.000 [-0.017, +0.019] | -0.019 [-0.036, -0.001] | -0.022 [-0.048, +0.003] | -0.003 [-0.060, +0.057] |
| Predictable context | -0.013 [-0.022, -0.003] | -0.009 [-0.017, -0.000] | -0.028 [-0.043, -0.014] | +0.001 [-0.025, +0.025] |

## Reading

- **P1.5 does not remember places beyond the frame, even after a few steps away.**
  - Where context doesn't give a re-entering cell away, the world draws it right about a third of the time, whether
    it saw that cell 2 steps ago or never.
  - Where context does give it away, it is right 83% of the time, again regardless.
  - Re-entering terrain is drawn from the visible neighbours alone.
- **S2's self-fed revisit agreement (0.68 within 16 steps) was context, not recall.**
- **What memory could add:** October's S5 found that memory plus neighbours predicts re-entering terrain 91% of the
  time, against 77% for neighbours alone, and terrain essentially never changes on revisits (S3).
  - The surprising context, about a quarter of entering terrain cells here, is where recall would lift the world from
    about 0.34 toward the reader's own ceiling on simulator frames (0.94 at short gaps in S2).
- **For dreams:**
  - more than half of the cells entering a dreamed view are revisits in real play (October E2), and the world
    reinvents them;
  - a dream that walks back toward water or a table seen earlier can lose it or invent another.

  That is imagination coherence, the job a place store exists for.

## Limits

- The reader's own errors are shared by both arms.
- One step only: deeper dreams also compound the world's own drawings.
- Revisit status comes from simulator positions, which the model has to infer.
- One checkpoint.
- The small negative differences in the predictable context (1-3 points) aren't explained here.

## Sensitivity of revisit lookup to inferred motion (11-23)

`pose_memory_audit.py` uses 8,000 stored simulator continuations from the 1,000 competent
history / random-plan starts, taking oracle half B. It uses rendered **real peripheries**
and S2's saved camera-shift reader. All positions are rebased to the first view. A geometry-only
place store writes the identity of the actual cell into the inferred key, and checks each newly
entering cell before its current write. Readings stop after the first terminal observation.
True cell identities and poses are used for scoring, except in the explicit oracle control.

| Pose source | Relative position exact at step 16 | Correct lookup on 25,371 revisits | Wrong-place lookup | Missing lookup | First visit gets a false memory |
|---|---:|---:|---:|---:|---:|
| True displacement (oracle) | 100% | 100% | 0% | 0% | 0% |
| Integrate each attempted movement | 26.68% | 82.64% | 15.44% | 1.92% | 8.14% |
| Existing visual shift reader | 80.37% | 86.65% | 11.97% | 1.38% | 5.16% |
| Visual reader with action constraints | 95.96% | 97.20% | 1.82% | 0.98% | 2.72% |

The action constraint discards an incompatible argmax shift and substitutes no movement.
No logits, retraining or simulator pose are used by that rule. It tests whether the
recorded action can constrain localization alongside the existing reader. A geometry
failure can still yield a plausible same-type block, so the metric asks whether the
lookup retrieves the correct identity rather than just the correct class. The result
leaves drift on generated frames, content quality, recovery after a positioning error
and longer-term lookup open.

Reproduce with `.venv/bin/python -m artifacts.experiments.20261123_place_memory.pose_memory_audit`.

## How long a single crop perturbs Mamba (11-23)

`glimpse_retention.py` uses the saved fixed-state renderer repeats (seven dark scenes
with a complete suffix and eight near night onset, four repeats per scene, twelve fixed
windows). A paired intervention changes only the crop at frame t, with recorded
periphery, lighting, record, outcomes, actions and all t+1 onward observations and
fixations shared. The two perturbations are the full-resolution crop versus its blurred
counterpart and differences among full-resolution crops when only sensor static changes.
The output is the predicted belief after the corresponding recorded action, before the
next observation. Mean L2 distances average repeats and windows within each scene.

| Dark scenes | Immediately after the action at t | +1 | +4 | +8 | +16 |
|---|---:|---:|---:|---:|---:|
| Fine versus blurred crop | 0.219 | 0.127 | 0.077 | 0.077 | 0.082 |
| Static alone | 0.135 | 0.091 | 0.068 | 0.074 | 0.080 |

Near night onset the initial differences are 0.249 and 0.057, falling to 0.076 and
0.066 at +16. Native reward and termination forecast changes are saved in the JSON.
The dark-panel mean absolute positive-reward forecast changes at +16 are 0.0035 and
0.0027 percentage points, respectively. This is a causal influence test on selected
recorded histories. It leaves semantic retention, conditional forecast improvement,
and the effect of training a delayed prediction loss open.

Reproduce with `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_place_memory.glimpse_retention`.

## Which features to keep at a world address (11-23)

`anchored_trace_audit.py` tests the payload independently of localization. Both the
compact entries and the P1.5 traces use the same true place addresses from replay.
The target is the simulator's terrain class when the cell re-enters view. No observation
from that frame goes to the reader. At a view's entry, the four payloads are:

- The last 16-value P1.5 periphery token.
- A 16-value per-place EMA, initialized from the first observation and weighted 0.875
  old and 0.125 new on each later observation.
- P1.5's 1,024-value S4D trace, with its local state paused while absent.
- The same trace, advanced by its native no-input dynamics over unobserved frames.

All readers also receive last-observation lighting and log age. The fixed P1.5
encoder processes the real frames. Shallow readers fit 20,185 revisit cells from
48 TRAIN episodes and score 22,497 cells from 64 disjoint DEV episodes. Episode
prefixes are capped at 384 frames. Only two held-out terrain classes change from
the previous observation. Confidence intervals resample episodes.

| Held-out revisits | Cells | Last seen | Compact EMA | Paused trace | Elapsed trace |
|---|---:|---:|---:|---:|---:|
| All | 22,497 | 98.09% | 97.93% | 98.12% | 93.65% |
| Inward neighbour has another terrain class | 6,026 | 96.83% | 96.71% | 96.61% | 87.82% |
| Last seen more than 64 frames earlier | 3,400 | 98.38% | 98.15% | 98.76% | 89.71% |
| Just one previous observation of this place | 632 | 97.31% | 94.78% | 95.73% | 91.14% |

The paired paused-trace improvement over last seen on all revisits is **+0.03 pp**
with a 95% interval of **[−0.27, +0.34] pp**. Over 256 silent applications of the
native record dynamics, individual mode amplitudes retain 1.83–29.93% of their
initial magnitude. The exact values are in the JSON. A fixed 64×64 dense map payload
costs 16 MiB for the coarse fp32 trace, and 0.25 MiB for the fp32 last-seen coarse
tokens, excluding metadata, lighting, fine features and fork management.

The small pilot used six calibration and eight evaluation episodes. It favoured the
paused trace over last seen by 2.9 points. The larger comparison finds similar
terrain decodability for the compact and paused formats. This is readout performance
for nearly static terrain with oracle geometry. Generated frames, changing or mobile
content, learned use of the retrieved tokens, and navigation are outside the test.

Reproduce with `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_place_memory.anchored_trace_audit`.

## Which reinvented cells matter for decisions (11-23, from `cells.npz`, no new run)

The expected wrong draws on revisits in the surprising context (3,310 of 4,996 cells), by the true terrain class:

| Class | Share of the errors | Cells | Agreement |
|---|---:|---:|---:|
| Stone | 20.3% | 1,018 | 0.341 |
| Tree | 17.8% | 623 | 0.052 |
| Path (mined-out stone) | 16.3% | 642 | 0.160 |
| Grass | 14.1% | 1,332 | 0.650 |
| Sand | 12.1% | 574 | 0.301 |
| Water | 6.8% | 381 | 0.409 |
| Lava | 2.7% | 96 | 0.068 |
| Coal / iron / diamond | 2.4% / 2.2% / 0.3% | 81 / 74 / 9 | 0.015 / 0.010 / 0.000 |
| Crafting table / furnace | 2.1% / 1.4% | 69 / 47 | 0.007 / 0.000 |
| Plant | 1.0% | 34 | 0.007 |

- **About 57% of the errors fall on classes a decision depends on:** resources (stone, trees, water, coal, iron,
  diamond, plants), the hazard (lava), and the agent's own constructions (table, furnace).
  - The world almost never redraws the table or furnace the agent built when it comes back (0.7%, 0%).
  - It redraws lava it saw 7% of the time, so the dream removes a danger.
  - Its own mined paths come back 16% of the time.
- **Grass, sand and path account for the rest (42%).** Decisions rarely depend on them.
- First sightings show the same low agreement on rare classes (trees 0.056, coal 0.010), as they must without memory.
  Revisits are where memory could recover them.

## Coarse-map coverage and routes to off-screen water (11-23)

`offscreen_routes.py` replays recorded terrain views and poses in the same DEV episodes
used by the archived P1.5 chooser analysis. It examines every fourth nonterminal
frame after frame 8 when the player is awake, drink is at most 3, and water is
absent from the current labelled view. This yields 1,540 snapshots from 85 non-random
episodes. Every comparison gets the same current view. The dense memory writes all
observed terrain, while two alternatives write only the saved chooser's sampled
windows or uniform windows, each with four sampled histories per episode. Gaze is
uniform in the first eight frames where the archive lacks predictions.

Use BFS on observed walkable terrain only, with the game’s static passability and
the simulator poses and terrain labels. Stop adjacent to previously seen water.
The route length omits the final turn and interaction. Dynamic entities and actual
action success are outside this test. A matched-path control gives each gaze arm
the dense coarse layout while leaving its candidate water addresses gaze-gated.

| Route budget | All-coarse solvable states | Old-eye-only retains | Old eye targets with full route layout retains |
|---:|---:|---:|---:|
| 16 moves | 906 of 1,540 | 91.36% [87.57, 94.51] | 95.31% [92.08, 97.69] |
| 32 moves | 1,275 of 1,540 | 89.59% [84.48, 94.14] | 96.75% [92.87, 99.35] |

Retention is conditional on a qualifying route with all-coarse memory. Intervals
resample the original episodes. Uniform gaze retains 92.10% at 32 moves, and 97.29%
with the full-layout control. Thus some misses are due to forgetting the target,
and much of the loss can be recovered by keeping the traversable layout. Sparse
updates sometimes leave an older terrain entry. At the 16-move threshold the
archived-eye maps report four sampled routes that the latest all-coarse layout
does not confirm; there are no such cases at 32 moves. The diagnostic prices
storage and geometric coverage under exact observations and assumed poses.
Learned policy returns and an eye that adapts its looks to navigation remain open.

Reproduce with `.venv/bin/python -m artifacts.experiments.20261123_place_memory.offscreen_routes`.

## Unconditional writes from dream renders (11-23)

`dream_write_audit.py` copies the last 64 observed real frames plus the start into
a geometry-only anchor map for the 1,000 competent-history random-plan roots. It
then looks at the saved eight P1.5 dreams and oracle-half-B real continuations per
root, using the same terrain reader and applying the same action constraints to
the camera shifts. The readout reports positions in the same origin and the map
indexes the previous real evidence at those positions. Only stable anchor classes,
frames with valid terrain readouts, and steps through termination and before the
first action that could edit terrain are counted. The set of potential edit actions
includes DO, placing stone, table, furnace or plant. The study measures what would
happen if each new frame unconditionally replaced the observed entry in its
branch-local store. It does not run a store-trained world.

| Reading | Real-rendered continuations | P1.5 dreams |
|---|---:|---:|
| Returning observed places whose reader class disagrees with the anchor | 285 / 18,035 (1.58%) | 5,843 / 18,137 (32.22%) |
| Any valid exposure to a previously observed anchor whose class disagrees | 20,589 / 1,165,317 (1.77%) | 69,857 / 1,172,386 (5.96%) |
| Touched anchor slots whose last eligible write would disagree | 6,377 / 354,307 (1.80%) | 22,176 / 354,673 (6.25%) |

These are conditional on a recognizable rendered class at an inferred address.
Pose error and reader error contribute to disagreements. The real-rendered arm
gives a scale for those errors under genuine observations. Generated observations
have different pose and reader errors. The comparison illustrates the risk of
letting a rendering resample what an earlier observation established. A branch
can retain its observed anchors and layer action-conditioned changes and newly
imagined places over them. Testing the consistency and use of those changes is
separate from this audit.

Reproduce with `.venv/bin/python -m artifacts.experiments.20261123_place_memory.dream_write_audit`.
