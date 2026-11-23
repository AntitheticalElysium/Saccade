# What does P1.5's belief keep about the map beyond the view? Results (2026-11-23)

Plan and code: `PLAN.md`, `belief_map.py` (commits 5249bab0, the mob fix, and the positive controls). Output:
`belief_map.json`.
- Frozen P1.5 checkpoint 10. 360 DEV episodes (220 fit, 40 validation, 100 test), every third frame: 40,314
  frames.
- Probes: one hidden layer, early-stopped. Rerunning reproduces the first reading exactly.

## Positive controls: the probe works

| Test-set accuracy | Majority | Belief | Current view | View + record read |
|---|---:|---:|---:|---:|
| Drink value (10 classes) | 0.131 | 0.439 | 1.000 | 1.000 |
| Zombie adjacent (balanced accuracy) | 0.500 | 0.835 | 0.942 | 0.913 |

## Direction to the nearest out-of-view place last seen as each class (8 sectors)

| Class | Test frames | Majority | Belief | Current view | View + record | Belief - view [95%] |
|---|---:|---:|---:|---:|---:|---:|
| Water | 9,820 | 0.222 | 0.194 | 0.317 | 0.314 | -0.123 [-0.150, -0.096] |
| Tree | 10,366 | 0.146 | 0.139 | 0.204 | 0.205 | -0.066 [-0.081, -0.050] |
| Lava | 5,973 | 0.195 | 0.148 | 0.195 | 0.176 | -0.047 [-0.093, -0.004] |
| Crafting table | 8,862 | 0.127 | 0.130 | 0.158 | 0.157 | -0.027 [-0.056, -0.004] |
| Stone | 10,427 | 0.225 | 0.240 | 0.351 | 0.341 | -0.111 [-0.139, -0.087] |
| Sand | 10,616 | 0.172 | 0.184 | 0.306 | 0.303 | -0.122 [-0.147, -0.098] |
| Path | 9,744 | 0.217 | 0.214 | 0.299 | 0.303 | -0.085 [-0.110, -0.059] |
| Zombie (out of view, seen within 32 frames) | 1,756 | 0.169 | 0.129 | 0.167 | 0.175 | -0.039 [-0.078, +0.002] |
| Cow (same) | 4,824 | 0.159 | 0.123 | 0.199 | 0.206 | -0.076 [-0.097, -0.054] |

- **By age:** the current view's advantage is largest when the place left view recently (water 0.427 at 1-16
  frames), because the view still shows the edge it left from.
  - The belief stays at the majority rate at every age (water 0.209 / 0.202 / 0.158; majority 0.232 / 0.233 /
    0.191).
- **Thirst:** water direction is no better when thirsty (0.213) than not (0.138), and both are at or below the
  rate the current view gives.
- **Log distance:** R² from the belief is 0.13 for water, against 0.30 for the current view, and near zero or
  negative elsewhere.
- **Displacement from the episode start:** R² is negative for every input, so no path integration is decodable.

## Reading

- **P1.5's belief keeps no spatial memory beyond the view.** It doesn't know the direction to any place, resource,
  hazard or creature once it is out of view. Neither the decision-relevant ones (water, lava, the table, zombies) nor
  the others (sand, path), and not even water when thirsty. It doesn't know where it is.
- **It does keep what its heads ask for:** the HUD (drink 0.44) and an adjacent zombie (0.84). So the user's
  hypothesis holds in a narrow form. The heads are myopic: reward and continuation at the next step, and forecasts up
  to 8 steps. Nothing asks the belief where anything is, so it keeps nothing about where.
- **The precedent for what would teach it:** blind navigation agents' recurrent memories form "selective and task
  dependent" maps when the task needs them (Wijmans et al. 2023). Networks trained to path-integrate grow grid-like
  location codes (Banino et al. 2018). In our system, a value or policy that needs to reach water would be the
  teacher. Today none exists, and goal gradients into the belief are on hold.

## Limits

- One checkpoint.
- Decoding probes show information present, not its use.
- The control probes read 1,304 current-view numbers against the belief's 256.
