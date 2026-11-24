# How fast do the weights absorb a permanent rule change? (2026-11-23, decision 7)

**Why.** Decision 7 asks whether Saccade needs a fast channel. The user (11-23): "we shouldn't build something like
this until we know how much of a problem it is". The frozen world doesn't revise from evidence within an episode
(`20261122_rule_change`). How fast the slow channel, the weights, absorbs a lasting change under the planned life
schedule has never been measured. This is AGENT.md's untested "revision" row. It measures a property of the system
under its agreed schedule; it selects no architecture.

**Claim.** Under decision 4's life schedule, P1.5's world comes to draw +2 after a drink once it has lived some
number of real +2 drinks. The number is what we measure.

**Competing explanations for a slow result:**
- the old replay keeps teaching +1 on similar inputs (dilution);
- the weights have lost plasticity.

The changed-data-only comparison (Codex) separates them. It is run only if absorption stalls, and is proposed, not
chained.

**The life (decision 4, agreed):**
- **Start:** P1.5 checkpoint 10's whole training state:
  - weights, AdamW moments, the fixation chooser and its optimizer, the streams, carry, cache, record and RNG;
  - P1's own update: harmoniser groups, bootstrap rows, forecast windows, gaze forecast, the chooser's online
    learning, 16 streams × 256 steps, 8 micro-batches, clip 1.0.
- **Replay ratio 16.**
  - Stream 0 is the online stream: it walks the life in order, 256 real steps per update.
  - Streams 1-15 draw episodes uniformly, with replacement, from all real experience so far: the corpus (P1.5's
    TRAIN split) plus the life episodes stream 0 has completed.
- **Learning rate:** the life's constant 1e-4 (decision 4), times each group's scale. It ramps linearly over 100
  updates from P1.5's final 1e-5, so the moments see no jump.
- **Phase +2:** 859 DEV episodes outside the evaluation panel, in a fixed random order (seed 20261123), replayed
  exactly from their recorded actions with water giving +2. That is about 833 updates and 212,433 real steps,
  with 1,554 real +2 drinks (simulated 11-23; a drink at 8 only reaches the cap of 9).
  - These are new worlds, disjoint by source world from the panel (Codex).
  - The recorded actions were chosen under the old rule.
- **Phase return:** 250 updates of TRAIN episodes as stored (+1, a fixed random order), unannounced.
- **One deviation from P1, declared:** the encoders (`model.perception`) are frozen.
  - S2's HUD reader reads P1.5's encoder tokens, and its loader asserts that the encoder is unchanged. Training the
    encoders would make every reading invalid.
  - The world's rule lives in the belief, the denoiser and the heads, which all learn.

**Readings, at updates 200 and 400, at the end of phase +2, and 100 and 250 updates into the return:**
1. **The rule-change panel** (`20261122_rule_change/fast_channel.py`'s protocol and grouping):
   - 150 DEV episodes, replayed under +2 and +1, cached once;
   - 8 native 4-step draws per drink event, read by S2's HUD reader;
   - shares on reader-correct events.
   - **The primary reading:** the share of +2 on fresh histories (B_first: no +2 earlier in the episode). It is the
     default changing (Codex). P1.5 draws +2 on 0.08% of these.
   - The evidence groups: in-context use, reported alongside.
   - Group A's +1 share: interference with the old rule. It is 78.5% for P1.5.
2. **Counts at each reading (Codex):**
   - distinct real +2 drinks lived (stream 0);
   - replayed +2 drink frames seen in training (all streams, including stream 0's own);
   - updates and real steps.
3. **Forgetting:** P1's fixed monitor (G1 flow term on the monitor split). It is 0.00175 for P1.5 at checkpoint 10.
4. **The reader floor per group.** If it falls below 0.95 in a group, that group is reported as reader-limited.

**Interpretation, fixed before results:**
- **Absorbed:** B_first's +2 share is at least 0.50. Report the first reading that reaches it, with its counts. If
  none does: not absorbed within 1,554 real +2 drinks and about 833 updates.
- **Return:**
  - A's +1 share comes back to within 0.05 of 78.5% by a return reading, read against how fully +2 had been
    acquired at the end of phase +2 (Codex);
  - B_first's +2 share at the same readings.
- **One seed.** The P1 monitor is the only forgetting reading. A rule-A-only continuation, to separate the
  learning-rate rise from the rule, is run only if forgetting appears.

**Run:**
- `python -m artifacts.experiments.20261123_water_revision.water simulate` (CPU: replays);
- `... water smoke` (5 updates and one reading on 3 episodes);
- `... water run`.
