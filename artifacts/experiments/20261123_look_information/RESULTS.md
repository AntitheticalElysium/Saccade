# What does a look add to the agent's knowledge? Results (2026-11-23)

Plan and code: `PLAN.md`, `look_information.py` (commit 873ce384). Outputs: `summary.json`, `rows.npz`.
- Frozen P1.5 checkpoint 10; 160 DEV episodes.
- 5,659 acquired-crop cases: day 3,332, night 2,327.
- The comparison is each crop against the same window rendered at the periphery's acuity, which is what the agent
  already had.

| Case | Creature | Entity-free map | HUD |
|---|---:|---:|---:|
| Belief change, absolute (relative to the belief's norm), day | 0.976 (5.8%) | 0.413 (2.5%) | 0.637 (3.8%) |
| Belief change, night | 0.739 (4.4%) | 0.336 (2.0%) | 0.791 (4.7%) |
| Predictive gain on the next crop at the same place, day (nats per RGB component) | -0.0052 [-0.0080, -0.0029] | -0.0006 | -0.0006 |
| Predictive gain, night | -0.0007 [-0.0015, 0.0000] | +0.0002 | -0.0004 |
| Immediate foveal surprise at t, day / night | 2.00 / 2.94 | 1.69 / 2.83 | 1.28 / 2.02 |

95% episode-bootstrap intervals for belief change are within about ±10% of each mean (summary.json).

## Reading

- **Belief change separates creatures from static, day and night.** The ratio is 2.4x by day and 2.2x at night,
  where pixel surprise couldn't separate them (lift 1.20 against 1.21). The corrected belief moves more for what a
  crop shows than for sensor noise.
- **Predictive gain is zero everywhere.** A crop at t doesn't improve P1.5's prediction of the next crop at the same
  world place, against an immediate surprise of 1.3-2.9 nats. That fits the known fact that fine evidence barely
  reaches what the belief carries forward (F1.6: the global 256-D projection loses it).
  - The belief moves at t, but the predicted belief and the trans-saccadic head don't use the change.
  - This is the declared competing explanation: the predictive definition reads the memory's failure to keep what
    was seen, not the eye.
- **What it means for the eye's information drive:**
  - **Belief change works with today's system:** it has the right selectivity, it is cheap (one extra belief step
    with the periphery-acuity crop), and any eye could compute it.
  - **Predictive gain is the more principled target** (information that improves predictions of real evidence), and
    sensor noise can't earn it. It needs a memory that keeps glimpse content. So the drive couples to the memory
    design: memory beyond the frame, and the world track.
- **Limits:**
  - One checkpoint; real frames only.
  - Belief change can be large for a crop that moves the belief without informing it. The relevance drive and the
    uniform floor counterweigh that, and predictive gain is the check once measurable.
  - The next crop is scored at the window shifted by the player's move, and skipped where the shift leaves the grid.

## Fixed-state static control of the belief-change bonus (11-23)

Rescored all 49 windows on the archived four renderer-noise repeats per physical scene. Used
frozen P1.5 checkpoint 10 and held the episode history, current recorded periphery, lighting,
record read, reward, and terminal flag fixed. Only the crop was taken from each noise repeat.
Each repeat supplies its own full-resolution crop and the matched blurred version at the same
window. The bonus is the L2 distance between the resulting two corrected beliefs.

| Period | Scenes | Mean fine-minus-blur bonus | Mean belief displacement when only the full-resolution crop's static changes | Highest-bonus window switches, repeat pairs |
|---|---:|---:|---:|---:|
| Dark | 8 | 0.517 | 0.253 | 11/48 |
| Night onset | 8 | 0.619 | 0.088 | 0/48 |

In the matching fixed-periphery TSP likelihood control, the highest-NLL window switched on
26/48 dark repeat pairs and 0/48 at night onset. Belief change is less sensitive by this
ranking check, and static alone still changes its reward. The two displacement columns
describe different comparisons and their ratio is not a noise decomposition. These are
hindsight score rankings on eight scenes per period, with correlated repeat pairs within
each scene. This does not measure a trained eye or the benefit of the crop to a decision.

Reproduce with
`JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_look_information.belief_static_audit`.
