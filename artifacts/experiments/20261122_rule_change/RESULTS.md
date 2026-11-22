# Does one observation of a changed rule change the world's next prediction? No

P1.5 checkpoint 10, frozen.
- **Episodes:** 150 DEV episodes with at least two drink events, replayed exactly under rule B (water gives +2 drink)
  and rule A.
- **Replay check:** the rule-A replay reproduces the stored frames to within 1 grey level.
- **Draws:** 8 native 4-step draws per drink event, read by S2's HUD reader.

| Group | Events | Reader floor | Drawn +2 | Drawn +1 | Unchanged | Other |
|---|---|---|---|---|---|---|
| A (trained rule) | 2,193 | 0.986 | 0.5% | 77.5% | 20.2% | 1.8% |
| B, first drink of the episode | 150 | 0.993 | 0.1% | 84.2% | 15.7% | 0.1% |
| B, a +2 seen within the last 8 frames | 532 | 0.848 | 1.5% | 77.3% | 16.4% | 4.7% |
| B, a +2 seen earlier in the episode | 573 | 0.901 | 0.2% | 67.3% | 31.0% | 1.6% |

**The reader floor is a reader weakness, not the rule change.**
- The reader reads a real 9 in the drink cell as a 3: 74 of 119 and 47 of 90 events in the two evidence groups, and
  28 of 65 under rule A.
- Rule B reaches 9 more often (7 + 2), so its floor is lower.
- Two evidence groups fall below the declared 0.95 floor. So, beyond the declared reading, the shares are recomputed
  on the events the reader reads correctly:

| Group (reader-correct events) | Events | Drawn +2 | Drawn +1 | Unchanged |
|---|---|---|---|---|
| A | 2,163 | 0.50% | 78.5% | 20.2% |
| B, first | 149 | 0.08% | 84.7% | 15.1% |
| B, recent evidence | 451 | 1.80% | 80.1% | 14.6% |
| B, older evidence | 516 | 0.17% | 68.9% | 29.9% |

By the number of earlier +2 drinks seen in the same episode (reader-correct B events):

| Earlier +2 observations | 0 | 1 | 2-3 | 4-7 | 8+ |
|---|---|---|---|---|---|
| Events | 149 | 138 | 204 | 230 | 395 |
| Drawn +2 | 0.08% | 0.27% | 1.84% | 0.98% | 0.66% |
| Drawn +1 | 84.7% | 76.9% | 81.8% | 73.6% | 69.6% |

## Reading

**By the declared reading, the fast channel doesn't adapt.**
- P(+2 | evidence) - P(+2 | first) is 0.014 (recent) and 0.001 (older), against a bar of 0.05.
- P(+2 | evidence) is at most 1.8%, against 0.10.
- After eight or more observations of the new rule in the same episode, the world still draws the trained +1 in 70%
  of draws and +2 in 0.7%.

**What this establishes:**
- The deployed world's memory (the record, the belief, the 8-frame context) carries what it saw, but its predictions
  don't revise from an observed rule change.
- We have memory, not adaptation. The world was trained on one rule set, so nothing ever asked it to infer rules from
  context. This is the measured form of "we already have both channels": the in-context channel isn't there.
- A side fact: under its own trained rule, the world leaves the drink digit unchanged in 20% of draws after a drink,
  the increment under-drawing seen before.

**What it doesn't establish:**
- whether in-context adaptation can be learnt, with training across rule variants as AdA and Nagabandi et al. do;
- how fast the weight channel absorbs the rule (proposed in saccade/design/AGENT.md).

**Limits:** one checkpoint, one rule change (a value change on a HUD digit), off-policy rule-B episodes. The reader's
weakness on 9 is reader-specific.
