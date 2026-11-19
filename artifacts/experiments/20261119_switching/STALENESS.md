# Can an epoch-stale cache stand in for fresh substitutes? (11-19; user: what do papers say, is it dramatic?)

## Literature

- **One update stale:** JEDI caches its one-step estimate and uses it as the next update's conditioning
  (`2605.13013`, Algorithm 1). That works because its updates step through time one frame at a time; ours predict a
  64-frame chunk at once from new data, so the previous update never covered the frames we need.
- **About one epoch stale, used as model inputs or targets:**
  - XBM ([Wang et al. 2020](https://ar5iv.labs.arxiv.org/html/1912.06798)): features "drift exceptionally slow". Drift
    D = |f(x; theta_t) - f(x; theta_{t-dt})|^2 is large early and stable after about 3K iterations; the memory starts
    after 1K warm-up iterations; a memory of the whole training set (up to an epoch stale) still helped (R@1 77.4
    against 69.8 at 1%).
  - Memory banks against MoCo ([He et al. 2020](https://arxiv.org/pdf/1911.05722)): epoch-stale keys "are less
    consistent"; MoCo's smoothly varying encoder beat the memory bank by 2.6 points (60.6 against 58.0). Staleness costs
    when stored values are compared against fresh ones.
  - R2D2 (Kapturowski et al. 2019): stored recurrent states from older networks suffer "representational drift";
    burn-in and stored states mitigate it.
  - MuZero Reanalyse: recomputing stale targets with the latest network gives better targets.
  - GNNAutoScale ([Fey et al. 2021](https://arxiv.org/abs/2106.05609)): stale historical embeddings used as inputs to
    the current forward; the approximation error is bounded by the history's staleness times the learned function's
    Lipschitz constants.
- **The pattern:** staleness is cheap once representations drift slowly (late training, low learning rate), costly
  early, and costlier where the consumer is sensitive (high Lipschitz constant). Codex's noise test showed our record
  read is sensitive.

## Measurement on our checkpoints (`staleness.py`, `staleness.json`)

P1.5 checkpoints are 2.56M frames apart, about one pass over the 2.43M-frame training split: the cache's staleness.
32 fixed training chunks, identical levels and noise at every checkpoint. Stale error = a checkpoint-k substitute's
error against checkpoint k + 1's tokens; fresh error = checkpoint k + 1's own.

| Pair | Low signal (the most generated-like substitute): stale / fresh error | Error cosine | High signal: stale / fresh |
| :--- | :--- | :--- | :--- |
| 1 -> 2 | 13.0x | 0.25 | 172x |
| 3 -> 4 | 2.6x | 0.41 | 32x |
| 4 -> 5 | 1.7x | 0.54 | 22x |
| 5 -> 6 | 1.27x | 0.70 | 13x |
| 7 -> 8 | 1.11x | 0.77 | 6.3x |
| 9 -> 10 | 1.03x | 0.88 | 2.4x |

- **The drift is the encoder's, not the denoiser's.** At every stage the estimates move by about as much as the tokens
  themselves (drift / token drift 1.05-1.7): the token space shifts under the jointly trained encoder, and the
  denoiser's predictions move with it.
- **Early (the 64-step half, checkpoints 1-5) a cached substitute is mostly encoder drift:** 1.7-13x the fresh error
  at low signal, its error pointing elsewhere (cosine 0.25-0.54).
- **Late (the 256-step half, checkpoints 6-10) it is close to fresh:** 1.03-1.13x at low signal, cosine 0.70-0.88.
  At high signal the ratio stays large (2.4-13x), but on a fresh error of about 0.00013 per element: the stale value
  is the old token, a tiny absolute difference.
- **With a frozen encoder (P2, CURRICULUM) the dominant source disappears.**

## What it implies

- **From scratch:** a cache is usable in the second half and costly in the first. A hybrid takes fresh substitutes in
  the 64-step half (where the extra pass is cheaper per update) and the cache in the 256-step half (where it is close
  to fresh). It saves about the 256-step half's extra pass (my estimate: about 1.2 h of the 3), not all of it.
- **Starting switching from the cache alone would mean starting it at about checkpoint 5.** Fresh substitutes can
  start once the estimates are meaningful (the low-signal error is already at its final level, about 0.01, by
  checkpoint 2).
- **As a P2 fine-tune (encoder frozen) a cache would be nearly free of staleness**, but a 2,000-6,000-update fine-tune
  does not need one.
