# Can history change a prediction immediately? The thirst period: results (2026-11-24)

**Verdict: no.** After 2,000 updates of the denoiser on episodes whose thirst period varies by episode, the world does
not draw the drop when the observed interval says it is due. This holds at the held-out periods and also at the
trained ones. The fine-tune also halved the native drop. What it learned is the PLAN's first competing explanation:
the mix of periods, a low coin at every lag.

The run: `thirst.py train`, then `eval 0` and `eval 2000`, one seed, exit 0. Readings are in `run/eval_{0,2000}.json`,
with per-event draws in `run/eval_{0,2000}_units.json`. Episode-bootstrap intervals come from `bootstrap.py`
(`bootstrap.json`). The training log is `run/train_log.json`.

## The bars

The table gives P(drawn drop) at the due lag (τ + 1 = p), on reader-correct events, with episode-bootstrap 95%
intervals. The draws are 60 DEV episodes, 4 draws per event.

| Period | Role | Events (episodes) | P1.5 | Fine-tune, 2,000 | Bar | Result |
|---|---|---|---|---|---|---|
| 16 | held out | 272 (55) | 0.020 [0.006, 0.038] | 0.035 [0.023, 0.050] | ≥ 0.50 | **fails** |
| 25 | held out | 205 (47) | 0.112 [0.091, 0.134] | 0.076 [0.055, 0.097] | ≥ 0.50 | **fails** |
| 25, at lag 21 | native habit | 220 (50) | 0.278 [0.238, 0.321] | 0.106 [0.075, 0.133] | ≤ 0.20 | met only because every lag is flat |
| 21 | native | 267 (53) | 0.535 [0.479, 0.603] | 0.248 [0.213, 0.294] | within 0.05 of P1.5 | **fails** (−0.29) |
| 13 | trained | 265 (59) | 0.011 [0.002, 0.024] | 0.018 [0.010, 0.027] | reported | not learned |
| 29 | trained | 152 (44) | 0.099 [0.067, 0.146] | 0.097 [0.065, 0.136] | reported | not learned |

**The reader floor** is at least 0.94 at every lag except p = 29's lags 30-31 (0.86-0.88).

## What it shows

- **The due lag isn't singled out at any period other than the native one.** After the fine-tune, P(drawn drop) at
  every reported lag:
  - p = 16: 0.034-0.10, with the due lag at 0.035;
  - p = 13: 0.018-0.118, with the due lag lowest;
  - p = 25: 0.042-0.106, with the due lag (0.076) below lag 21 (0.106);
  - p = 29: 0.053-0.168, with the due lag (0.097) below lag 27 (0.168).

  At p = 25 the due lag sits above lags 23-24 (0.042). That is the "overdue" pattern P1.5 already shows: at p = 25,
  P1.5 draws 0.112 at lag 25 against 0.05 at lags 23-24. The world isn't timing the drop from the observed interval,
  even within the trained family, so this is not a generalization failure.
- **The native peak was diluted, not replaced.** At p = 21 the drop at lag 21 fell from 0.535 to 0.248. It is still
  the only peak (other lags 0.03-0.08). That is what training on a mixture where p = 21 is 30% of episodes produces
  if the interval isn't read.
- **The loss didn't move.** The training G1 stayed between 0.0014 and 0.0016 from update 100 to 2,000. One drop per
  p frames, in one HUD cell, is a very small share of the loss. The currency study (`20261114_currency_eda`) put
  events at about 1% of a likelihood loss. So the fine-tune had almost nothing pulling it toward the timing.

## What it doesn't show

- Whether the record holds the interval. The PLAN assumes it does: the 64-frame time constant spans the last two drops.
  This run didn't read the record.
- Whether a longer fine-tune, a fine-tune that lets the Mamba learn, or a loss that prices the drop would learn it.
  None of those were run.

**Limits:**
- one seed;
- the record route only (Mamba frozen);
- 2,000 updates;
- one clock;
- recorded actions chosen under the native rule.

## For decision 7

Together with the water revision (not absorbed after 834 updates), neither route changes the prediction:
- **The slow route:** a permanent rule change barely moves the default.
- **The immediate route:** observed evidence in the history doesn't move it either, even after training on varied
  periods.

Under the current losses, the world neither adapts its weights quickly nor reads a changed regularity from context.
