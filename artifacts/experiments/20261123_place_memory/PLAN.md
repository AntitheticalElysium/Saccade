# Does P1.5 remember places beyond the frame? (2026-11-23, decision 6)

**Why.** Decision 6 (memory beyond the frame and the episode) has a location-keyed episodic store designed since
10-23, to be built on a measured trigger. The trigger is the world failing to recall places it has seen. October
measured the environment's side:
- 56% of cells entering view are revisits (median gap 20 steps, p90 177);
- terrain doesn't change on revisits;
- neighbours predict re-entering terrain 77% of the time, and memory plus neighbours 91% (`OPTION3_NOTES.md`, S5).

S2 measured revisit agreement inside self-fed dreams (0.68 / 0.60 at gaps 1-16 / 17-64, against 0.94 / 0.88 for the
simulator). That reading had no first-sighting comparison and no neighbour control, so it can't separate memory from
guessing from context. This plan measures what the trained world recalls from real history.

**Setup.**
- Frozen P1.5 checkpoint 10. DEV episodes with ε < 0.99, the first 120 by index.
- **Starts:** frames t >= 8 where the player moves at t (so cells enter view at t + 1), up to 40 per episode, seed
  20261123.
- **One-step dreams** from the real history with the recorded action and recorded fixation (S2's `rollout`, H = 1,
  4 draws per start).
- **Cells:** those entering view at t + 1 (their world cell was out of view at t), where S2's terrain reader on the
  real frame t + 1 reads terrain (not an entity or the player).
- **Agreement:** the reader's terrain class on the dreamed frame equals its class on the real frame.

**Cell groups:**
- **first sighting:** never in view earlier in the episode;
- **revisit**, by the gap since it was last in view: 2-16, 17-64, 65-256, more than 256;
- **context:** "predictable" when the cell's true class equals that of its neighbour toward the player (in view at
  both frames), "surprising" otherwise.

**Claim.** P1.5 recalls revisited terrain: within the surprising context, revisits agree more often than first
sightings, more so at short gaps.

**Competing explanation.** Revisits are easier cells (e.g. near the start, more grass). The separating control is the
comparison within context and within the true class: a class-matched difference, weighted by revisit counts.

**Readings, fixed before results** (episode-bootstrap 95% intervals):
1. agreement by group × context;
2. the class-matched revisit-minus-first-sighting difference in the surprising context, by gap;
3. the same in the predictable context.

**Limits.**
- One world, one step, real history.
- The terrain reader's own errors (it agrees 0.94 on simulator frames at short gaps in S2).
- Revisit status from simulator positions: the model must infer them.

**Run:** `JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_place_memory.place_memory`
