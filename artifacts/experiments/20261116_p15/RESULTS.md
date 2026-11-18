# P1.5 results (run 2026-11-16/17, dissection 11-19)

Run: `saccade-p15`, F6.1's recipe with the spatial history (MLP read), route by role, the harmoniser at its fixed
point, the gaze forecast and Mamba-3 (PLAN.md), 25.6M frames, one world seed. The monitor loss fell at every
checkpoint, so checkpoint 10 was selected (`run/ckpt_10.pt`, SHA256 `5b849b2a...`). The queue reached its readings on
the fourth attempt:
1. A device mismatch in my adapter check (CPU-only test could not see it), fixed by Codex.
2. A transient Triton ptxas error.
3. A host OOM in S1: the record states encode keeps add ~2.3 GB.

Codex's retry unit then completed all 16 stages (`readings/STATUS.json`). The adapter check passed on the real
checkpoint (`eval_checks.json`). Gemini ran checks 15 and 17 on 11-17 without saving code. Both are rerun from
`dissect.py` (committed before reading, 25c4ff32):
- Check 17 reproduces Gemini's to within 0.001.
- Check 15's clock rows reproduce it.
- Check 15's inventory rows were void in Gemini's run: true change 0.000 at the measured cells, while P1.5's tokens
  change there by 12-21 (`inventory_tokens.py`). They are corrected here.

## 1. Summary

1. **The pixel clocks are drawn, from the record.**
   - Natural frames, drawn at the event (quiet frames): awake energy 0.916 (0.000), drink 0.580 (0.045), food 0.493
     (0.025), asleep rise 0.667 (0.009). F6.1: 0.029 / 0.034 / 0.042 / 0.053.
   - On-schedule pairs, due / delayed: energy 0.995 / 0.000, hunger 0.873 / 0.000. F6.1: 0.017 / 0.016 and
     0.075 / 0.074.
   - The 2 x 2 swap puts the timing in the record the denoiser reads: effect 0.99 / 0.86. The belief's effect is 0.000.
2. **Waking is drawn on time.**
   - First sleeps: distance to the true wake times 0.07 steps (F6.1 5.41), awake by step 12 0.985 (simulator 1.0,
     F6.1 0.545), late or missing 0.02 (0.91).
   - Repeat sleeps: late or missing 0.32 (0.875).
3. **Broad gains:**
   - Reward calibration NLL 0.0064 (0.0121; below the pair-memory reference 0.0068).
   - TSP 2.011 (2.242).
   - Coal 0->1 0.725 (0.351) and iron 0->1 0.363 (0.144).
   - Free recursion at step 16: 0.795 (0.643; truth 1.27).
   - Entities persisting 0.837 (0.806; F4 0.840).
   - Fewer belief-driven increments: wood under NOOP with a DO belief 0.22 (0.69).
4. **The encoder under route by role.**
   - Reward sends no gradient into it (budget).
   - Late token drift 0.00107: a sixth of F6.1's 0.0067, below F4's 0.0030.
   - Token variance 0.361 (F6.1 0.26, F4 0.38).
   - Effective rank unchanged: 4.91 (4.88; F4 5.96).
5. **Costs:**
   - The belief holds less of the HUD than F6.1's while the tokens hold it as well: energy exact 0.52 against 0.71,
     food 0.32 against 0.38. Read from the current percept: 0.92 / 0.91 against 0.87 / 0.86.
   - Zombie hits are drawn more often both ways. Asleep beside a ready zombie, death 0.844 (0.784; true 0.992).
     Beside a cooling one, false waking 0.089 (0.029), damage 0.077 (0.063), death 0.050 (0.022; true 0.008 /
     0.005 / 0.000).
   - Stone 2->3 0.683 (0.835).

## 2. Clocks

| Natural frames (check 15, DEV fold 4, F6.1's rows and seeds) | F6.1 event / quiet | P1.5 event / quiet |
| :--- | :--- | :--- |
| Food / drink / awake energy drop | 0.042 / 0.025, 0.034 / 0.047, 0.029 / 0.011 | 0.493 / 0.025, 0.580 / 0.045, 0.916 / 0.000 |
| The same after a reset | 0.026, 0.033, 0.006 | 0.477, 0.478, 0.883 |
| Asleep rise / recovery / damage | 0.053, 0.482, 0.701 | 0.667, 0.537, 0.655 |
| Food +1 / drink +1 | 0.069, 0.572 | 0.370 (quiet 0.083), 0.804 |
| Stone +1 / coal +1 / iron +1 / sapling +1 | 0.601, 0.202, 0.119, 0.036 | 0.741, 0.397, 0.239, 0.043 |
| Placing stone / sapling; tools | 0.753, 0.802; 0.74-0.85 | 0.865, 0.741; 0.73-0.83 |

| Pairs (check 17), due / delayed | F6.1 | P1.5 | Record effect | Memory effect |
| :--- | :--- | :--- | :--- | :--- |
| Energy, on schedule (due step 62) | 0.017 / 0.016 | 0.995 / 0.000 | 0.99 | 0.000 |
| Hunger, on schedule | 0.075 / 0.074 | 0.873 / 0.000 | 0.87 | 0.00 |
| Energy, old (edited due branch) | 0.007 / 0.007 | 0.031 / 0.000 | 0.03 | 0.000 |
| Hunger, old | 0.089 / 0.089 | 0.202 / 0.000 | 0.20 | 0.00 |

The old pairs edit the due branch's drop off the episode's own schedule. P1.5 draws the unedited schedule almost
always and the edited one rarely; F6.1's recency fine-tune drew both (0.79 / 0.98 at 6,000 updates). What P1.5 times
the drop from beyond the frames since the last change is not measured.

Elsewhere: the natural thirst pair 0.203 / 0.008 (F6.1 0.047 / 0.047). Sleep (the sleep bank, 32 scenes x 8 futures):
- Wait ratio to energy 9: 1.10 (2.13).
- Wait ratio to waking: 1.09 (1.94).
- Drawing energy 9 is followed by waking in 256 of 256 transitions.

The belief's own reads (linear probes on DEV) hold the reward-priced clocks more (recovery 0.807 against 0.710,
asleep rise 0.622 against 0.471, need damage 0.441 against 0.256) and the awake pixel clocks barely (food 0.048 against
0.040): the denoiser reads those from the record, not from Mamba.

## 3. Standard readout, guards and the encoder

- **Generated-vs-real classifier: 0.920 (0.991).** Its power controls fell too (mismatched actions 0.79 against 0.94,
  frozen time 0.88 against 0.91), so part of the drop is a weaker classifier on P1.5's conditioning features.
  Generated frames now sit between frozen time and shuffled history (0.977); F6.1's were above all three.
- **Monitor G1: 0.00175 (0.00383).** Not like-for-like: the record is an extra input to the denoiser.
- **Interface (generated unlock against recorded 1.0):**
  - Higher: EAT_COW 0.61 (0.36), DEFEAT_ZOMBIE 0.71 (0.61), COLLECT_COAL 0.75 (0.63), COLLECT_IRON 0.40 (0.29),
    COLLECT_DRINK 0.94 (0.87).
  - Lower: COLLECT_DIAMOND 0.001 (0.025).
  - Risky class: P(termination) 0.115 (0.105; recorded 0.128).
- **Mining pairs:** coal DO/DO 0.104 (0.067); wood DO/DO 0.916 (0.954), NOOP/DO 0.22 (0.69). Movement 64 / 64.
- **Budget, checkpoint 10 (weighted gradient norms):**
  - Encoder: TSP 2.09 (0.40), G1 0.143 (0.086), reconstruction 0.155 (0.032); every reward and forecast head 0.
  - Mamba: reward forecast 4.74 at distance 1, reactive reward 3.5, gaze forecast 0.30, G1 0.044.
  - Harmoniser sigma: G1-G3 0.04, G4 0.72, G5 4.36.
- **Belief contents (S1.2, exact HUD from b_t):** health 0.99, food 0.32, drink 0.41, energy 0.52, the inventory
  0.49-0.98. F6.1: 1.00, 0.38, 0.43, 0.71, 0.55-0.98. Achievement AP 0.874 (0.900). The current-percept control
  reads them at 0.87-0.96 in both worlds.

## 4. What this does not establish

- One world seed and five changes bundled. Within the run only the clocks' source is attributed, by check 17's
  swap. The encoder gains (TSP, coal / iron, drift, variance) fit route by role, but nothing here isolates it from
  the harmoniser's weights or the record.
- Why the belief holds less of the HUD: route by role (Mamba shaped only by its heads) and the record carrying the
  clocks both predict it; neither is tested.
- The danger shift is measured on the curves' populations only. The zombie's cooldown is a hidden timer on a
  moving entity, which a per-cell record does not follow; that is a hypothesis.
- The old pairs' low rates are an observation; their cause is open.
