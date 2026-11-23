# What does P1.5's belief keep about the map beyond the view? (2026-11-23, decision 6)

**Why.** The user asked whether Mamba remembers what its heads consider important, and whether it remembers distance
and position. Decision 6 turns on it: if the belief already keeps relevant places, the place memory's job is
different.

**Claim (the user's hypothesis).**
- The corrected belief keeps out-of-view map content selectively:
  - classes that bear on reward or survival (water, tree, lava, crafting table, zombies, cows) are decodable above
    what the current view gives;
  - classes no head asks about (sand, path) are not.
- Water is kept better when the agent is thirsty.

**Competing explanations and their controls.**
1. **Decoding comes from the current view's context** (biome cues near the edge), not memory.
   - Control: the same probes on the current coarse tokens and lighting, and on those plus Mamba's read of the record
     (everything Mamba received at that frame except the glimpse).
   - Memory is the belief's margin over that control, and how it changes with time since the place was seen.
2. **Retention is generic, not selective.** Compare the margin across classes.

**Setup.**
- Frozen P1.5 checkpoint 10. DEV episodes with ε < 0.99, 360 drawn with seed 20261124, split by episode: 220 to fit,
  40 for early stopping, 100 to test.
- Frames t >= 16, every third frame.
- **Map targets per class:**
  - the nearest world cell of that class that was in view at an earlier frame, is out of view now, and was last seen
    as that class, within 24 cells;
  - its direction (8 sectors), log distance and age (frames since last seen).
- **Mob targets:** the nearest active zombie or cow, out of view now, that was in view within the last 32 frames. Its
  true current direction (8 sectors).
- **Self-position:** displacement from the episode start (every episode starts at the same place), as dx and dy.
- **Thirst:** water direction separately where the drink value is 4 or less, and where it is 7 or more.

**Probes.**
- One hidden layer of 256 (GELU), inputs standardised, Adam.
- Early stopping on the validation episodes.
- Cross-entropy for direction, Huber for distances and displacement.
- Inputs: the belief (256); the control, current coarse tokens and lighting (1,304); the control plus the record read
  (1,560).

**Readings, fixed before results** (episode-bootstrap 95% intervals over test episodes):
1. direction accuracy per class and input, against the majority-class rate; the belief-minus-control difference,
   paired;
2. the same by age: 1-16, 17-64, 65 or more frames;
3. log-distance and displacement R², by input;
4. water direction accuracy, thirsty against not.

**Limits.**
- One checkpoint.
- Decoding shows information present, not that the world or a decision uses it.
- Targets come from simulator labels; the probes are diagnostic.

**Run:** `JAX_PLATFORMS=cpu MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261123_belief_map.belief_map`

## Positive controls (added 11-23 after the first reading, before rerunning)

The first reading found nothing decodable in the belief, so the probe pipeline itself needs checking. Two quantities
the belief is known to carry:
- the drink value (10 classes); the belief reads the HUD, with energy exact at 0.52 earlier;
- a zombie adjacent to the player (AP 0.92 on 11-22), scored by balanced accuracy.

If the belief probe decodes these well, the map null is real. If not, the probe is the problem.
