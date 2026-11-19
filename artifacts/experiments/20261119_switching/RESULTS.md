# Switching arm `sw_rec` (record route, p = 0.5, 4,000 updates from P1.5): stage 1 readings

S2 stage 1 on the same 4,000 starts, simulator futures and frozen readers as P1.5 (`20261119_s2/gates_sw_rec_*.json`,
`model_sw_rec_*.npz`). Stage 1b and the clock checks at 4,000 are in `sw_rec.s2b_*`, `sw_rec.check1*` (pending at the
time of writing).

## Mobs

| All mobs per frame | Step 0 | 1 | 4 | 8 | 16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Simulator | 1.075 | 1.088 | 1.103 | 1.135 | 1.196 |
| P1.5 | 1.075 | 0.971 | 0.844 | 0.739 | 0.603 |
| 2,000 updates | 1.075 | 1.018 | 0.983 | 0.964 | 0.950 |
| 4,000 updates | 1.075 | 0.949 | 0.800 | 0.673 | 0.515 |

| 4-step persistence (simulator) | P1.5 | 2,000 | 4,000 |
| :--- | ---: | ---: | ---: |
| Zombies (0.980) | 0.911 | 0.882 | 0.894 |
| Cows (0.959) | 0.806 | 0.723 | 0.769 |
| Skeletons (0.852) | 0.586 | 0.587 | 0.398 |
| All, lags 1-4 (0.975-0.922) | 0.910-0.791 | 0.871-0.728 | 0.891-0.749 |

- **2,000: less decline, by churn.** Mobs are drawn into the interior from nowhere: cows by day 0.0180 per step against
  the simulator's 0.0042 (P1.5 0.0052), zombies at night 0.055 against 0.018, skeletons by day 0.0059 against 0.0003.
  They also vanish more: cows by day 0.029 against 0.0011 (P1.5 0.016). Skeletons overshoot (+27%), and persistence is
  lower than P1.5's.
- **4,000: worse than P1.5 throughout.** Counts are lower from step 1 on, persistence is lower, and edge arrivals are
  fewer (cows by day 0.0187 against the simulator's 0.0454; P1.5 0.0217).
- **The guards break at 4,000.**
  - Energy drop curves: max deviation 0.089 overall and 0.205 in the 99+-steps stratum (P1.5: 0.007 and 0.004, the clock
    that held at every depth).
  - Overdue food: 0.111 (P1.5 0.005). N5: max belief-HUD gap 0.058 (P1.5 0.013).
  - Multivariate energy score: 7.79 (2,000: 7.72; P1.5 7.62; simulator 5.85).
- **By the declared rule (less step-1-to-16 decline and better 4-step persistence, guards intact) the arm fails at both
  reads.** Not extended to 6,000; the matched control as planned (run only after a pass) is moot.

## Confound (mine): the fine-tune re-warmed the learning rate tenfold

`switch_ft.py` trains with a fresh AdamW at 1e-4 after 100 warm-up updates, which is P1.5's peak rate. P1.5 ended its
cosine schedule at 1e-5 for the denoiser (`min_lr_ratio` 0.1; checkpoint 10's optimizer groups: 1e-5, encoder 3e-6). The
held-out one-step monitor on real inputs rose from 0.001752 to 0.001855 at 2,000 and 0.001812 at 4,000. A tenfold re-warm
can itself move a converged model, and the non-monotone path from 2,000 to 4,000 is consistent with that. So these
readings do not separate switching from the fine-tune recipe. The no-substitute control at the same rate would show
whether the recipe alone does this; a rerun at P1.5's final rate (1e-5) is the fine-tune the arm should have been.
