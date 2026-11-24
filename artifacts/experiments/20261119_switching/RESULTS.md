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

# Rerun `sw_rec2` (P1.5's own optimizer continued at 1e-5; otherwise as `sw_rec`): stage 1, 1b and checks 17 / 15

Readings from `compare.py sw_rec_2000 sw_rec_4000 sw_rec2_2000 sw_rec2_4000`, `20261119_s2/gates_1b_sw_rec2_4000.json`,
`20261116_p15/schedule_sw_rec2_4000.json` and `catalogue_sw_rec2_4000.json`. P1.5's checks come from `schedule.json` and
`catalogue.json`, read on 11-19 by the same code.

## The recipe confound is resolved

The held-out one-step monitor stays flat: 0.0017525 → 0.0017571 (2,000) → 0.0017567 (4,000). The first arm went to
0.001855 / 0.001812. At 4,000:

| Guard at 4,000 | P1.5 | `sw_rec` (re-warmed) | `sw_rec2` (continued) |
| :--- | ---: | ---: | ---: |
| Energy curve, max deviation (all) | 0.007 | 0.089 | 0.002 |
| Energy curve, 99+ steps | 0.004 | 0.205 | 0.001 |
| N5, belief-HUD gap | 0.013 | 0.058 | 0.015 |

So the first arm's collapse at 4,000 came from the tenfold re-warm, not from switching.

## Mobs: more counts, not more persistence

| All mobs per frame | Step 0 | 1 | 4 | 8 | 16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Simulator | 1.075 | 1.088 | 1.103 | 1.135 | 1.196 |
| P1.5 | 1.075 | 0.971 | 0.844 | 0.739 | 0.603 |
| `sw_rec2` 2,000 | 1.075 | 1.000 | 0.919 | 0.840 | 0.729 |
| `sw_rec2` 4,000 | 1.075 | 0.992 | 0.902 | 0.823 | 0.718 |

- **Gains by kind, at step 16 (4,000):** cows 0.363 → 0.482 (simulator 0.768) and arrows 0.028 → 0.043. Zombies
  0.176 → 0.157 and skeletons 0.037 → 0.036 gain nothing.

| 4-step persistence (simulator) | P1.5 | `sw_rec2` 2,000 | `sw_rec2` 4,000 |
| :--- | ---: | ---: | ---: |
| Zombies (0.980) | 0.911 | 0.877 | 0.864 |
| Cows (0.959) | 0.806 | 0.779 | 0.799 |
| Skeletons (0.852) | 0.586 | 0.579 | 0.498 |
| Arrows (0.346) | 0.129 | 0.103 | 0.144 |

- **The S3 blocker now also fires for zombies:** in stage 1 at both reads, and in stage 1b's competent random plans.
- **The count gain is arrivals.** The table is per step at 4,000:

| Event per step | Simulator | P1.5 | `sw_rec2` 4,000 |
| :--- | ---: | ---: | ---: |
| Cows arriving at the edge, day | 0.0454 | 0.0217 | 0.0328 |
| Zombies arriving at the edge, night | 0.0386 | 0.0213 | 0.0233 |
| Cows appearing in the interior, day | 0.0042 | 0.0052 | 0.0121 |
| Cows appearing in the interior, night | 0.0217 | gate does not reject | 0.0416 |
| Cows vanishing, day | 0.0011 | 0.016 | 0.020 |
| Zombies vanishing, night | 0.0095 | 0.020 | 0.027 |

  - Edge arrivals move toward the simulator.
  - Interior appearances also rise, and so do vanishings.
  - This is the first arm's direction at 2,000 (churn), milder.

## Clocks: energy holds, hunger regresses by 4,000

- **Check 17, on-schedule pairs (due minus delayed, drawn):**
  - Energy: 0.995 → 0.986.
  - Hunger: 0.873 [0.849, 0.896] → 0.781 [0.752, 0.811].
  - The memory effect stays at 0 (the clocks are still drawn from the record).
- **S2 time-to-event curves, max deviation (P1.5 → 2,000 → 4,000):**
  - Overdue food: 0.005 → 0.007 → 0.032.
  - Food, 20-98 steps: 0.176 → 0.174 → 0.246.
  - Drink, 20-98 steps: 0.166 → 0.151 → 0.215.

  These are intact at 2,000 and worse at 4,000.
- **Check 15:**
  - Food drops drawn 0.488 → 0.413, drink drops 0.577 → 0.512, and the after-reset rows also fall by about 0.07. The
    CIs overlap; the direction is the same in all four rows.
  - Energy, recovery, damage, food up, drink up and the inventory rows: unchanged within the CIs.
- **Return per future** is 0.174 against the simulator's 0.156 (P1.5 0.146), now over.
- **Unchanged:** DO holds (stage 1b), internal consistency and revisits.

## Verdict by the declared rule

**Fail.**
- Less decline: yes.
- 4-step persistence better than P1.5: no.
- Interior spawns not above P1.5's: no.
- Clocks no worse: no (overdue food and on-schedule hunger regress).
- N5: equal.
- Return: worse.

The direction is the first arm's (spawns and lower persistence), now at the right rate. By the plan, this bounds
one-step switching on the record as a fine-tune.

The rule judges the package against P1.5: switching plus 4,000 extra denoiser-only G1 updates. The hunger regression,
the zombie persistence drop and the churn could belong to either part. The matched control (PLAN.md, "Control after
the fail") separates them.
