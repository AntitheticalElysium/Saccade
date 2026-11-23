# Can history change a prediction immediately? The thirst period (2026-11-23, decision 7)

**Why.** Decision 7's first architecture test, approved by the user 11-23. The water run measures how fast the
default changes. This one asks whether evidence in the history can change the prediction at once, with fixed weights,
once training has seen variation (Codex: "thirst tells us whether history can change a prediction immediately").

In P1.5 the drink digit drops every 21 awake steps and has never varied. The world draws drops on schedule from the
record (on-schedule pairs 0.995).

**The rule** (Craftax-Classic `update_player_intrinsics`):
- the thirst counter grows by 1 per awake step and 0.5 asleep;
- above 20 the drink digit drops by one (unless it is 0) and the counter resets;
- drinking doesn't reset it.

The in-simulator patch replaces both `new_thirst > 20` comparisons with `> p - 1`, so the awake period is p. It is the
patch of `20261122_rule_change/clock_curriculum_eda.py`; with p = 21 it must reproduce the stored frames.

**Claim.** After a denoiser fine-tune on episodes whose thirst period varies by episode, the world draws the next drop
when the observed interval says it is due, including at periods it never trained on.

**Competing explanations:**
- it learns the mix of periods (a coin at every lag);
- it keeps the native habit (a drop at 21).

**The separating reading.** On histories matched for the time since the last drop and the current digit, with only
the earlier interval differing between periods, is the drop drawn at the episode's own period?

**Training data.**
- All 8,069 TRAIN episodes, replayed exactly from their recorded actions with a period drawn per episode (seed
  20261123).
- p = 21 with probability 0.3. Otherwise p is uniform over 12-30, excluding 16, 21 and 25.
- **Held out: 16 and 25.**
- The recorded actions were chosen under the native rule.

**Fine-tune: stage 1's and test 3's recipe.**
- P1.5 checkpoint 10; the denoiser only, G1, AdamW 1e-4 with 100 warm-up updates, clip 1;
- 64-step chunks, 16 fresh streams over the re-simulated episodes; carry, cache and record reset at each episode
  start;
- encoders and Mamba frozen. This is the record route: the denoiser reads the per-cell record, whose 64-frame time
  constant holds the last two drops;
- **2,000 updates.**

**Evaluation.**
- **Episodes:** 60 DEV episodes (seed 20261123) re-simulated under p = 16 and 25 (held out), 21 (native), and 13 and
  29 (trained).
- **Units:** frames t with:
  - the agent awake since the drop before the last one, so the observed interval is the period;
  - at least two drops already seen in the episode;
  - drink digit at least 1;
  - no drink at t + 1;
  - τ (steps since the last drop) with τ + 1 in {p - 2, ..., p + 2} or τ + 1 = 21.
- **The draw:** the world's native 4-step sample of frame t + 1 from real history, 4 draws (fast_channel's sampler),
  read by S2's HUD reader.
- **Reading:** P(drawn drop) by period and τ + 1. A drop is the drink digit one below the current one.
- **Arms:** P1.5 (update 0) and the fine-tune at 2,000 updates, on the same units and seeds.
- **The reader floor:** the reader's digit on the real frame, per period. Events it misreads are excluded.

**Bars, fixed before results** (fine-tune at 2,000 updates):
- **In-context inference works:** at both held-out periods, P(drop | τ + 1 = p) is at least 0.50.
  - For p = 25, P(drop | τ + 1 = 21) is also at most 0.20: the native habit gone.
  - Partial: 0.20-0.50 at the held-out p.
- **Native preserved:** P(drop | τ + 1 = 21) at p = 21 is within 0.05 of P1.5's.
- **Trained periods (13, 29):** reported. Recognition within the trained family is the weaker claim (Raventós et al.).

**Limits:**
- one seed;
- the record route only (Mamba frozen);
- one clock;
- recorded actions chosen for the native rule.

**Run:**
- `python -m artifacts.experiments.20261123_thirst_period.thirst simulate` (CPU);
- `... thirst smoke`;
- `... thirst train`;
- `... thirst eval 0` and `... thirst eval 2000`.
