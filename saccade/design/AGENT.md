# The agent (living note, opened 2026-11-22)

This note records the 11-22 design discussion that changed what Saccade is. Decisions land in DECISIONS.md, facts in
READING.md as usual; this file holds the picture and the open questions. Nothing here is implemented yet. Where it
conflicts with STEP_SPEC.md (world frozen during imagination, offline only), this note is the newer intent, pending the
decisions listed at the end.

## What Saccade is (user, 11-22)

"There is no actor: the entire system is the agent." The world model is not a simulator with a player inside it. It is
the agent's imagination and its understanding of the world's rules. The belief is what matters for its goals. The
agent improves through a loop:
- **living:** learning about the world from expensive, limited real experience, looking where it matters;
- **dreaming:** practising cheaply what it has learnt.

**Endgoal, in the user's words:** "something you can pretrain for a bit and then let it learn in the real world like
an actual biological system".

**Selling point:**
- an agent that adapts to new situations about as fast as a person, and efficiently;
- it needs no retraining or new code;
- it keeps getting better as it lives, and carries what it learns into new tasks.

**The eye** is an extension of the system, not a separate entity with its own claim.

**Progress is judged by the agent, not by the world's fidelity.** World measurements are for locating failures. The
world must be accurate where the agent goes, never optimistic, and consistent under any sequence of looks.

## Faculties

| Faculty | In Saccade | What teaches it |
|---|---|---|
| Senses | Periphery and lighting, always on; the eye, a 3x3-cell full-resolution crop where the agent looks | Predicting what the eye will see; its own reconstruction; reward heads |
| Imagination and understanding | Denoiser + record | Predicting everything it perceives, from all real experience |
| What matters | Belief (Mamba) | Its goals: reward and continuation, now and forecast. Open: the value of its decisions too (below) |
| Choice | One policy reading the belief, with two outputs: the motor action and where to look next | Its goals, in dreams and in life; the reflexive eye (surprise, contingency) is its early default |
| Practice | Dreams: the same faculties running on imagined input, branched from real states, short at first | — |
| Experience | Replay: the pretraining corpus plus its own life, interleaved | — |

**Two modes:**
- **Awake:** act in the real world, store full frames and looks, update the record and belief.
- **Dreaming:** imagine from real states and practise. The world draws the coarse frame, then the crop where the eye
  looks; reward and termination come from the world's state, not from the look.

Every faculty keeps learning in both modes. Freezing a part is something an experiment does to measure, not part of
the agent.

## Adaptation: how one experience changes the agent

Two timescales, as in the brain's complementary learning systems:
- **Fast:** something in memory changes what the agent predicts and does, within the episode, with no weight change.
- **Slow:** replay and dreaming turn experience into lasting change in the weights, interleaved with old experience so
  old knowledge survives.

**What exists (measured 11-22):**
- **The record** is a fixed per-cell trace that resets at every episode start, so in-context change cannot outlast an
  episode.
- **The deployed world does not revise its predictions from an observed rule change** (`artifacts/experiments/
  20261122_rule_change/RESULTS.md`).
  - Rule B: water gives +2 drink. In 150 replayed DEV episodes, the frozen P1.5 world draws +2 in 0.08% of draws
    at the first drink and 0.2-1.8% after the new rule has been seen in the same episode.
  - Even after 8 or more observations it is 0.7%, and the trained +1 is still drawn 70% of the time.
  - We have memory (record, belief, context), not adaptation: nothing ever asked the world to infer rules from
    context.
- **The slow channel is untested:** how many updates, and how much replay, it takes to absorb a new rule without
  losing old ones.
- **The two channels depend on each other.** Interleaving old and new experience protects old knowledge. But if the
  agent can't tell from context which rule applies, interleaving rule-A and rule-B experience can only teach an
  average of the two. So recognising the current situation in context (the fast channel) is what lets the slow channel
  keep both, and lets an unannounced return to A be recognised. This is reasoning, not yet measured.

## How to judge it (proposal, not decided)

The agent's lifetime, not module comparisons:
- **speed:** real experience until it is competent in a new situation;
- **learning to learn:** that amount falls across the lifetime;
- **retention:** earlier situations stay solved;
- **never stops:** improvement continues, plasticity holds;
- **economy:** computation per real step.

**Codex's protocol (11-22):** rule A -> rule B -> an unannounced return to A -> a held-out related variant, all inside
Craftax-Classic with the same senses and actions.

References:
- a fresh copy of the same pretrained agent started directly on the later variant (what living through the earlier
  situations bought);
- the same continuing agent with and without dreaming, at the same real-step budget, compute reported.

## What counts as a new situation

- **New rules inside Classic:** controlled, the same screen and actions. Preferred first.
- **Full Craftax:** the natural open-ended world, but it changes the senses and actions too: 130 x 110 pixels, a
  9 x 11 view of 10-pixel blocks, 4 inventory rows and 43 actions, against Classic's 63 x 63, 7 x 9 view, 7-pixel
  blocks, 2 rows and 17 actions (checked 11-22). Later, for a broader test.
- **New goals over the same world:** cheap, but they barely test learning new rules.
- **New maps only:** too weak.

## Precedents (read at the source, 11-22)

**The loop:**
- Dyna (Sutton & Barto 9.2-9.3; optimistic model errors are corrected only online);
- Ha & Schmidhuber's iterative training procedure;
- Dreamer and DreamerV3 (online, from scratch);
- DayDreamer (real robots, walking in an hour);
- Dreamer 4 (offline only, by choice);
- MBPO ("short model-generated rollouts branched from real data");
- AlphaGo (imitation, then self-play).

**Fast adaptation:**
- AdA ("as quickly as humans", in memory after large-scale meta-RL; weights fixed afterwards);
- EMPA (human learning efficiency on 90 games, with hand-built theory programs);
- Nagabandi et al. 2018 (a meta-learned dynamics prior "rapidly adapted to the local context").

**Continual learning:**
- Abel et al. 2023 ("the best agents never stop learning");
- Continual-Dreamer (world-model replay for continual RL);
- Dohare et al. (loss of plasticity; continual backpropagation);
- the primacy bias (Nikishin et al.);
- RLPD (keep offline data in replay); Cal-QL (offline policies "behave poorly during fine-tuning").

**Sensing policies with world models:**
- MOSER (a view-conditional world model + a sensory and a motor policy; intrinsic rewards). The closest precedent.
- TaskSense; SUGARL.

**Brain:**
- complementary learning systems (McClelland, McNaughton & O'Reilly 1995);
- sleep replay (Wilson & McNaughton 1994) and awake replay (Foster & Wilson 2006);
- goal-directed sequences (Pfeiffer & Foster 2013); replay by utility (Mattar & Daw 2018);
- sleep improves a skill 20% without practice (Walker 2002); task dreams predict improvement (Wamsley 2010);
- eye movements in imagery and dreams (Brandt & Stark 1997; Laeng et al. 2014; Senzai & Scanziani 2022; Andrillon et
  al. 2015);
- the world as an outside memory (O'Regan 1992); task-driven, just-in-time gaze (Hayhoe & Ballard 2005).

**Human reference:** Crafter's leaderboard lists humans at 50.5% with "Life experience" and 0 interaction, against
21.8% for the best listed 1M-step agent (Lake et al. 2017: rapid learning needs causal models and learning-to-learn).

## Open decisions, in order

1. What counts as a new situation, and the lifetime judge.
2. How one experience changes the agent quickly (fast and slow channels), informed by the measurements above.
3. The choice faculty: one policy with two outputs, its inputs, how it learns awake and dreaming.
4. Whether goals (the value of decisions) shape the belief; the denoiser stays general.
5. The rules of dreaming: starts, horizon, outcomes from the world's state, look consistency.
6. The life cycle: schedule, replay mix, plasticity.
7. Knowing what it doesn't know, and curiosity.
8. Acuity.
