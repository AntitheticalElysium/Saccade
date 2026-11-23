# Does the world use a place memory? Yes, strongly, with correct addresses (2026-11-23)

PLAN.md's test: P1.5 plus a per-cell read of the last-seen coarse token at the cell's world address. Addresses come
from simulator positions: a declared substitution that isolates the read from localisation. The fine-tune is
denoiser-only, G1 only, 4,000 updates, by test 3's recipe.

The evaluation: 120 DEV episodes, 40 movement starts each, one-step dreams from real history, 4 draws, S2's terrain
reader, 35,170 entering cells, episode-bootstrap 95% intervals.

## Reading 1: revisits against first sightings (surprising context: the visible neighbour doesn't give the cell away)

| Update | First sightings | Revisit 2-16 | 17-64 | 65-256 | 257+ | Class-matched gain at 2-16 | at 17-64 |
|---|---|---|---|---|---|---|---|
| 0 (P1.5) | 0.341 | 0.349 | 0.330 | 0.318 | 0.396 | +0.003 [-0.015, +0.020] | -0.013 |
| 2,000 | 0.333 | 0.911 | 0.904 | 0.881 | 0.856 | +0.570 [+0.548, +0.590] | +0.569 |
| 4,000 | 0.329 | 0.927 | 0.924 | 0.916 | 0.870 | +0.585 [+0.564, +0.605] | +0.586 [+0.562, +0.611] |

- Cell counts: 4,425 first sightings, then 2,529 / 1,497 / 828 / 142 revisits by gap.
- **Bar "used" (+0.20 at 2-16, +0.10 at 17-64): passed by a wide margin.**
- The gain holds beyond 256 steps (0.870 against 0.329).
- **The predictable context** (the neighbour gives the cell away): revisits are 0.978-0.980 at 4,000 updates, against
  0.826 for first sightings.

## Reading 2: the swap (same starts, view, action and history; only the entering cells' entries change)

| Variant, update 4,000 | Surprising 2-16 | 17-64 | Predictable 2-16 |
|---|---|---|---|
| Correct entry | 0.927 | 0.924 | 0.978 |
| Empty (no entry) | 0.310 | 0.304 | 0.827 |
| Wrong address (3 cells away along rows) | 0.307 | 0.294 | 0.691 |

- Correct beats empty: +0.617 at 2-16.
- "Wrong below empty" holds in the predictable context (0.691 against 0.827), not in the surprising one (0.307 against
  0.310). Copying the wrong cell's class would also land near the context rate, so the PLAN's other half of the
  reading decides it: whether the drawing takes the wrong entry's class above its base rate.

**The class reading** (`identity_4000.json`). The cells are surprising revisits whose wrong-address entry is present
and holds another class. The cell count is 2,635 over 119 episodes, or 1,105 over 116 when the wrong entry is
age-matched (same gap band as the correct entry, Codex).

| Share of draws | Taking the wrong entry's class | Taking the true class |
|---|---|---|
| Correct entry | 0.038 | 0.898 |
| Empty | 0.376 (the base rate) | 0.225 |
| Wrong entry | **0.935** | 0.015 |
| Wrong entry, age-matched | **0.910** (base 0.349) | 0.024 |

- Wrong minus empty: +0.558 [+0.531, +0.588], and +0.561 [+0.526, +0.599] age-matched.
- **The world draws what it remembers, not merely that a memory exists.** The swap's "used" reading passes.

## Guards

- **First sightings:** surprising 0.329 against P1.5's 0.341, and predictable 0.826 against 0.829. Both are within
  0.03.
- **G1 during the fine-tune** falls from 0.00142 (update 100) to about 0.0012 (updates 3,000-4,000). It doesn't rise.
- **Determinism:** the 4,000-update evaluation, rerun from the same checkpoint and seeds, reproduced every agreement
  number exactly.

## What it establishes, and what it doesn't

- **Given the right old observation at the right cell, the denoiser uses it, and follows its content.** P1.5's revisit
  failure (no recall at all) is repaired: 0.93 against 0.33 on hard revisits, and the gain holds past 256 steps.
- **It doesn't establish:**
  - that Saccade can find the right entry for itself: the addresses here are the simulator's, and that is the next
    test;
  - anything about dreams, fine entries, the agent's read, or a world trained with the memory from scratch.
- **Limits:**
  - one seed;
  - a denoiser-only fine-tune on frozen encoders: stored tokens are detached, and the encoders never learn to make
    recallable tokens;
  - the coarse half only.
- This is the ceiling for the inferred-address test: same cells, same reader, same protocol.
