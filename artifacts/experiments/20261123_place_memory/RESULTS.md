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
