# Does one observation of a changed rule change the world's next prediction? (user 11-22: "what do you mean that we already have both channels? Are you sure? Is this something we've tested? Run whatever you need")

## Why

I said (11-22) that our design "has both channels" for fast adaptation:
- **replay plus dreaming** (store an experience at once, absorb it slowly);
- **in-context memory** (the record and the belief carry recent observations with no weight change).

Neither has been tested on a changed rule.
- **The record** is a fixed per-cell trace, reset at every episode start (`Record.scan`'s `first`).
- **The clock tests** showed the denoiser reads timing from it under the rules it was trained on.
- **Nothing** has asked whether an observation that contradicts the trained rules changes what the world predicts
  next.

This test measures the in-context (fast) channel on the deployed world, with weights frozen. The weight channel is a
separate, larger experiment; it is proposed in `saccade/design/AGENT.md`, not run here.

## The rule change

**Rule B:** drinking water raises the drink value by 2 instead of 1. It is applied after Craftax-Classic's own step:
when the drink value rose, it is set to min(9, old + 2).
- Thirst still lowers the value every 20 thirst steps.
- Nothing else changes, so frames differ only through the drink digit and what follows from it.

**Episodes:** DEV episodes with at least two drink events, replayed exactly. Each uses its stored actions with the
collector's own random keys (`artifacts/eda/replay.py`), under rule B, and under the original rule A as a check:
- the rule-A replay must reproduce the stored frames exactly;
- a rule-B episode ends at its own death, if that comes earlier.

## The measurement

**Model:** P1.5 checkpoint 10, frozen. It reads each episode's real frames (A and B separately) as in every DEV
reading: the monitor's fixation prior, the record and the belief over the whole episode.

**Units:** each drink event at frame t (the drink value rose from frame t-1 to t) with t >= 8.
- For B, only events whose previous value is at most 7, so a +2 is visible.
- For A, previous value at most 8.

**Draws:** 8 per event, from the world's native 4-step sample of frame t given the real past (the deletion panel's
sampler and seeds). S2's HUD reader reads the drink digit. The drawn increment is the drawn digit minus the real digit
at t-1.

**Groups:**
- A, all events (the trained rule);
- B, the episode's first event (no evidence of the new rule yet);
- B, later events whose previous +2 was within the last 8 frames (visible to the denoiser's context);
- B, later events whose previous +2 was earlier (visible only through the record and the belief).

## Readings (declared before running)

- **Reader floor:** the reader reads the real B frame's drink digit correctly. Below 0.95, the test is unreadable.
- **The fast channel adapts:** P(+2 | B, earlier evidence) - P(+2 | B, first event) >= 0.20, in either recency group.
- **It doesn't:** that difference is below 0.05 in both groups, and P(+2 | B, earlier evidence) < 0.10.
- **Between:** partial.

**Also reported:**
- P(+1) under B, which measures sticking to the trained rule;
- P(+2) under A;
- the share of draws that leave the digit unchanged.

**What each outcome means.**
- **Adapts:** the deployed memory already revises predictions in context from one observation, with no weight change.
  Then the fast channel exists.
- **Doesn't:** we have memory, not adaptation. A fast channel would have to be built. The candidates are learning to
  adapt across many rule variants (AdA, Nagabandi et al.) or fast weight updates from replay (the slow channel made
  fast).

**Limits:** one checkpoint, one rule change, one kind of event. Actions were recorded under rule A, so B episodes are
off-policy, which is legitimate for a prediction test.
