# Switching dissection, stage A (events.py on S2's saved grids; `events.json`, `events.log`)

The data are 4,000 starts x 8 futures per world, read by the same reader with the same plans. The tables give cells per
1,000 transitions, pooled over day and night. SEs come from an episode bootstrap paired against P1.5.

Abbreviations:
- P1.5: the base world.
- R2k / R4k: the rerun `sw_rec2` at 2,000 / 4,000 updates.
- A1: the first, re-warmed arm `sw_rec` at 2,000.

## 1. The extra appearances are spawns, not jumps (candidate 5 refuted)

| Cows, per 1,000 transitions | Step | Simulator | P1.5 | R4k | A1 |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Interior spawn (no vanishing within 3 cells) | 1 | 6.3 | 7.0 | 13.4 | 19.6 |
| Interior spawn | 9-16 | 6.7 | 7.9 | 17.1 | 26.3 |
| Interior jump (a vanishing within 3 cells) | 1 | 1.3 | 1.5 | 2.1 | 2.8 |
| Interior jump | 9-16 | 1.3 | 0.8 | 1.8 | 2.6 |
| Edge arrival | 1 | 50.1 | 21.0 | 32.8 | 44.7 |
| Edge arrival | 9-16 | 55.2 | 22.6 | 33.9 | 49.8 |
| Unpaired vanishing | 1 | 3.4 | 37.0 | 37.5 | 48.7 |
| Unpaired vanishing | 9-16 | 3.4 | 16.0 | 22.4 | 31.9 |

- **R4k:** spawns are about 2x P1.5 (+6.4 at step 1, +8.5 to +9.2 deeper, SE 0.5-0.6). Jumps grow by only +0.5.
- **Zombies:** spawns +1.8 to +2.3 per 1,000 against P1.5's 3.5-4.8 (simulator 3.2-4.0). Jumps +0.2 to +0.5.
- **Skeletons:** spawns +0.9 to +1.1.

## 2. The change is present at step 1, from a real history

At step 1 the record and the context are built from real frames only. R4k already differs from P1.5 there:
- **Cows:** interior spawns 13.4 against 7.0, and edge arrivals 32.8 against 21.0.
- **Start-zombie hazard:** 0.084 against 0.070 (+0.014, SE 0.003).

| Hazard per step of the start's own mobs | Step 1 | 2-4 | 5-8 | 9-16 |
| :--- | ---: | ---: | ---: | ---: |
| Zombies: simulator | 0.014 | 0.008 | 0.005 | 0.003 |
| Zombies: P1.5 | 0.070 | 0.029 | 0.021 | 0.017 |
| Zombies: R4k | 0.084 | 0.046 | 0.037 | 0.030 |
| Cows: simulator | 0.012 | 0.006 | 0.005 | 0.005 |
| Cows: P1.5 | 0.095 | 0.065 | 0.060 | 0.052 |
| Cows: R4k | 0.098 | 0.070 | 0.064 | 0.059 |
| Skeletons: simulator | 0.025 | 0.013 | 0.010 | 0.018 |
| Skeletons: P1.5 | 0.228 | 0.147 | 0.107 | 0.129 |
| Skeletons: R4k | 0.264 | 0.177 | 0.148 | 0.197 |

- **R4k against P1.5, zombies:** +0.014 / +0.017 / +0.016 / +0.013 (SE 0.001-0.003), so 50-75% faster.
- **Cows:** +0.004 to +0.008 (SE about 0.002).
- **Skeletons:** +0.030 to +0.068 (SE 0.006-0.019).
- **R2k:** the zombie excess is +0.008 to +0.010, so it grows with training.

The excess hazard is roughly constant with depth: a one-step property, not a compounding one. Nothing shows the
intended effect, which would be a lower hazard at depth from robustness to self-fed records. Vanishings at depth rise
instead (cows +3.5 to +6.4 per 1,000).

## 3. What the base world does at step 1 (the fade itself)

From a real history, P1.5 already:
- vanishes cows at 11x the simulator's rate (37.0 against 3.4 per 1,000 transitions);
- draws 0.42x its edge arrivals;
- loses start cows at 0.095 per step against 0.012, and 5-6% per step after that (about 10x).

The route test (`20261119_s2/routes_readers.json`) shows that, with a real record, P1.5 redraws the mobs it lost:
- record_real holds 0.93-0.99 mobs per frame through step 16;
- free running falls to 0.55.

So the fade is a large one-step loss rate. The real record hides it by restoring the lost mobs; the record of its own
frames cannot.

## 4. Arrivals and spawns move together

| Cows, per 1,000 transitions | Simulator | P1.5 | R4k | A1 |
| :--- | ---: | ---: | ---: | ---: |
| Edge arrivals | 50-55 | 21-23 | 33-35 | 45-50 |
| Interior spawns | 6-7 | 7-9 | 13-18 | 20-26 |

- Both rise together across the three worlds, as if moved by one setting: how readily the denoiser draws a mob where
  the record holds no evidence of one.
- P1.5 under-draws at the edge (newly revealed cells) but is about right in the interior (cells seen empty).
- A1 nearly reaches the simulator's arrivals and triples interior spawns.
- A world must treat "not yet seen" differently from "seen empty"; one setting cannot do both.

## Reading against the candidates (PLAN.md)

- **(5) Motion: refuted** (section 1).
- **(2) Mismatch** does not explain the harm. The changes appear at step 1 with a real record (section 2). It may still
  explain the absence of the intended gain; that is open.
- **(1) Shrinkage:** all four predictions hold.
  - The change is there at step 1.
  - The extra appearances are spawns.
  - The start's own mobs die faster.
  - The conditioning-driven held-out loss rises (level 0 +2.5%, level 1 +1.6%).

  This is a description so far, not a measured mechanism. Stage B measures how much the denoiser relies on the record
  directly.
- **(3) Dose** and **(4) Recipe:** not addressed by these data.
