# How fast do the weights absorb a permanent rule change? Results (2026-11-24)

**Verdict: not absorbed.** Under decision 4's life schedule, after 1,554 real +2 drinks lived and 834 updates, the
world draws +2 on 1.3% of drinks on fresh histories. It was 0.08% for P1.5; the PLAN's bar for "absorbed" is 50%.
No reading came close. The return to +1 therefore has nothing to undo.

The run: `water.py run`, one seed, exit 0. Readings are in `run/reading_*.json`, with per-event draws in
`run/reading_*_units.json` and the update log in `run/log.jsonl`.

## The readings

The +2 share of drawn drinks, on reader-correct events. The panel is 150 DEV episodes, 8 draws per event, read by
S2's HUD reader.

| Reading | Updates | Real +2 drinks lived | +2 drink frames trained | Fresh histories (B_first, 149) | After recent +2 evidence (451) | After older +2 evidence (516) | Old rule, A: +2 / +1 (2,163) | P1 monitor G1 |
|---|---|---|---|---|---|---|---|---|
| P1.5 (`20261122_rule_change`) | 0 | 0 | 0 | 0.08% | 1.80% | 0.17% | 0.50% / 78.5% | 0.00175 |
| 200 | 200 | 350 | 386 | 0.50% | 4.35% | 0.75% | 1.20% / 79.8% | 0.00190 |
| 400 | 400 | 742 | 933 | 1.01% | 3.41% | 1.04% | 0.82% / 62.1% | 0.00179 |
| End of +2 | 834 | 1,554 | 2,621 | **1.26%** | 5.54% | 1.16% | 1.31% / 71.1% | 0.00178 |
| Return, 100 | 934 | 1,554 | 2,928 | 1.43% | 5.82% | 1.57% | 1.14% / 63.7% | 0.00191 |
| Return, 250 | 1,084 | 1,554 | 3,201 | 1.17% | 5.82% | 1.50% | 1.20% / 76.5% | 0.00181 |

**How the counts are formed.** "+2 drink frames trained" is the PLAN's count over all streams. It is stream 0's
online drinks (`distinct_plus2`) plus streams 1-15's replays of completed life episodes (`replayed_plus2`; the code
counts streams 1-15 only under that name).

**The reader floor** is at least 0.95 in A (0.986) and B_first (0.993). The evidence groups are reader-limited by
the PLAN's rule: 0.848 after recent evidence, 0.901 after older evidence. Their shares are reader-correct events only.

## What it shows

- **The default barely moved.**
  - On fresh histories the +2 share rose from 0.08% to about 1.2-1.4% and stopped rising after update 400.
  - After recent +2 evidence in the same episode it is about 5.5%, against P1.5's 1.80% on the same events.
    That is a small in-context effect, still far from following the evidence.
- **The old rule wasn't overwritten.** Under the old rule, the +2 share stayed at 0.8-1.3%.
- **The +1 share swings on its own.** Under the old rule it moves between 62% and 80% from reading to reading, with
  the +2 share flat. The swing is between +1 and no increment drawn, so it's reading-to-reading variation of the
  fine-tune, not the new rule leaking in.
  - The PLAN's return criterion (A's +1 back within 0.05 of 78.5%) is met at return 250 (76.5%) but missed at
    return 100 (63.7%).
  - With nothing acquired, the return reading measures only that variation.
- **Forgetting is small.** The P1 monitor rose from 0.00175 to between 0.00178 and 0.00191, at most +9%, at the same
  time as the learning-rate rise from 1e-5 to 1e-4. The PLAN's rule-A-only continuation runs only "if forgetting
  appears". This much isn't treated as forgetting, and it wasn't run.

## Why: the competing explanations

The PLAN names two: the old replay keeps teaching +1 (dilution), or the weights have lost plasticity. This run doesn't
separate them; the PLAN's changed-data-only comparison would. It is proposed, not run.

**A rough mixture estimate** (not a measurement; it assumes the corpus drinks at the life's rate, 0.0073 per step):
- 834 updates × 15 replay streams × 256 steps is 3.2 M replayed steps. About 0.15 M of those are life rows (1,067 +2
  drinks / 0.0073).
- That leaves about 3.06 M corpus rows: roughly 22,000 +1 drinks against 2,621 +2 drinks, so +2 was about 10% of the
  drinks trained on.
- The weights draw +2 on about 1.3% of fresh histories, below even that mixture.
- **Why this matters:** with a uniform replay of all past experience and no input that tells the new worlds from the
  old, the best the weights can do is the mixture. That is a calibrated coin, as decision 7's table predicts for a
  change they can't tell apart. The changed-data-only comparison would show whether the shortfall below the mixture
  is plasticity.

## Limits

- One seed.
- Perception frozen (declared in the PLAN).
- The recorded actions were chosen under the old rule.
- The life is 859 episodes (212,433 real steps). A robot's lifetime would be longer, but its replay would dilute the
  same way.
