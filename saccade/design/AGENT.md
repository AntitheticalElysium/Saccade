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

## What a continual learner needs, and what Saccade has (11-22; the user: architecture first, data later)

Varying the pretraining world is a judge and an accelerant, not the mechanism. The design question is what the agent
must contain to learn continually, adapt quickly, keep what it knows and keep improving. From the literature, read at
the source:

| Need | Evidence | Saccade today |
|---|---|---|
| Keep the ability to learn | Networks trained continually lose plasticity (Dohare et al.); the primacy bias (Nikishin et al.) | Training anneals the learning rate to a tenth by the end; nothing preserves plasticity |
| Don't forget | Interleave old and new (CLS); slow learning on important weights (EWC, Kirkpatrick et al. 2017); generative replay, "inspired by the generative nature of hippocampus" (Shin et al. 2017), and replay of internal representations (van de Ven et al. 2020) | Training samples the corpus; no consolidation process. The world model could be its own generative replay: dreams that rehearse old situations |
| Learn from one experience | Episodic memory: Neural Episodic Control "is able to rapidly assimilate new experiences and act upon them" (Pritzel et al. 2017); the hippocampus | The 10-23 design's location-keyed episodic store (DECISIONS, "Agreed"), deferred as increment I1. The record is a per-episode trace that resets |
| Use evidence in context | Working memory; a slow learner trains a fast in-context learner (Wang et al. 2018) | Belief + record + recent frames. Works for hidden states that varied (clock phases), not for never-varied rules |
| Notice change; attach learning to situations | Latent-cause inference (Gershman et al.) | Not built ("the model doesn't know" was agreed 10-23, never built) |
| Evidence sets the update size | Learning rates follow volatility (Behrens); one-shot switching under causal uncertainty (Lee) | Not built: every experience moves the weights the same |
| Assign credit across a life | Replay instead of backpropagation through all of time | Gradients are truncated at chunk boundaries. A thirst interval and its next drop share a 64-frame chunk only ~33% of the time (Codex) |
| Consolidate offline | Sleep replay; replay ordered by utility (Mattar & Daw) | Not built: no sleep phase, no prioritised replay or dreams |
| Seek information | The eye; curiosity (Plan2Explore) | The eye exists; curiosity doesn't |

**What this implies, on three timescales, as in the brain:**
- **Working memory** (activity): belief, record, recent frames. Exists.
- **Episodic memory** (one-shot, survives episode resets, retrieved by situation or place): the deferred store. Central
  to adapting in a few experiences, and to carrying knowledge across deaths.
- **Knowledge** (slow weights of world, belief and choice): kept plastic, consolidated in sleep from replayed episodes
  and generated dreams, protected by interleaving.
- **Control:** noticing change, setting update size from evidence, curiosity. Missing.

**Build order (the user's 10-26 principle):** design the whole system first, build minimal, then add designed
components on measured triggers. The first proof the concept works at all is closing the loop: an agent that lives,
dreams and improves.

## Closing the loop: the minimal agent (design started 11-22)

The first proof is an agent that lives, dreams and improves. Its decisions, in order:
1. **The choice faculty:** where it lives, what it reads, the order of looking and acting within a step, and which
   gradients it sends. (Below.)
2. **Dreaming:** start states, horizon, outcomes from the world's state, how returns are formed.
3. **Learning in dreams and in life:** the imitation head start, the policy objective, the value.
4. **The awake/asleep cycle:** what runs when, the replay mix, plasticity.

The episodic store, change detection and evidence-set update size are designed in from the start (their interfaces),
and built when a measurement calls for them.

### Decision 1: the choice faculty

**Precedent, Dreamer 4, read at the source (2509.24527, section 3.3):**
- Agent tokens are inserted "as an additional modality into the world model transformer". They "attend to themselves
  and all other modalities, [but] no other modalities can attend back to the agent tokens. This is crucial for
  avoiding causal confusion of the world model".
- Phase 2 trains policy and reward heads on the agent tokens with multi-token prediction (L = 8), while "we continue
  to apply the video prediction loss".
- Phase 3 is reinforcement learning in imagination with PMPO (alpha 0.5), a reverse KL to a frozen copy of the policy
  (beta 0.3), and a value head on lambda-returns (gamma 0.997). It updates only the policy and value heads; finetuning
  the whole transformer "provides small additional benefits at higher computational cost".

**What the choice must know, and where it is today (measured):**
- **The belief holds the gist:**
  - a zombie adjacent, AP 0.921;
  - its cooldown, 0.952;
  - but not which cell it is in: 0.116, against 0.99 from the current frame.
- **It holds the HUD less well than the tokens:** energy exact 0.52 in the belief, against 0.92 from the percept.
- **The whole screen enters Mamba through one 256-number projection,** measured as lossy for spatial detail.
- **The hidden clock phases live in the record** (swap effect 0.99), and Mamba doesn't carry them (0.000).

**Options:**
- **A. The belief only** (STEP_SPEC section 5; m0.yaml p3.features). Simplest, but it reads through the 256-number
  bottleneck, which lacks spatial detail and the clocks the world uses.
- **B. The belief plus the current percept,** a separate network. It recovers current detail, but not the record or
  the world's understanding, and it is an actor in all but name.
- **C. Agent tokens inside the world (Dreamer 4).** The choice is a readout of the same computation the world uses
  to imagine:
  - it reads the current frame's tokens, the recent frames, the record (through each cell's read) and the belief
    (through the memory tokens);
  - one-way attention keeps the world's predictions from depending on the agent's own choices;
  - our denoiser already has one-way groups (physical queries never read glimpse keys), and every real or dreamed
    frame already gets one clean context pass, so reading it adds little cost.

**Laid out again 11-22 (the user: "stop treating Saccade as actor + model"; "10-24 isn't a rule"). Undecided.**

*1. Should the agent's choices read the world model's internal understanding?*
- **My "two sources of truth" objection was wrong.** Agent tokens are not a second memory; they persist nothing. They
  are the choice computation placed where it can see:
  - memory: the belief, through the memory tokens, and the record, through each cell's read;
  - evidence: the current tokens and recent frames;
  - understanding: the denoiser's contextual computation of what the world is doing.
- **What the denoiser holds that the belief lacks (measured):**
  - the hidden clocks: drawn from the record, swap effect 0.99, against 0.000 for Mamba;
  - each mob's cell: belief 0.116, frame 0.99;
  - the action-conditioned rules.
- **Neuroscience: one substrate for perceiving and imagining.**
  - "neural representations of imagined and perceived stimuli are similar in the visual, parietal, and frontal
    cortex" (Dijkstra et al. 2019);
  - perception uses the generative model's predictions (Rao & Ballard 1999).
  - "The denoiser is only for imagination" is the actor + model view.
- **World-model precedents:**
  - the actor reads the world's own latent state (Dreamer 1-3); planning in the world's latent (TD-MPC2);
  - agent tokens inside the world (Dreamer 4);
  - DIAMOND's agent reads generated frames, not the diffusion model's internals.
- **Practicalities:**
  - Dreamer 4's agent tokens read slightly corrupted context too, so the context pass's mild noise is not
    disqualifying (my concession to Codex was too quick).
  - Factorised attention lets agent tokens in the physical stream choose the LOOK from the coarse frame before the
    glimpse, and a second set choose the MOTOR action after it, within one pass.
  - Cost: unmeasured. The P1.6 world-eye's +12 h came from a separate sequential estimate pass, not from this.
- **Current lean:** agent tokens inside the world (C) are the one-agent design. Belief plus percept (B) is the cheaper
  test that may miss the hidden timing.

*2. Which parts should the agent's goals shape, and how?*
- **10-24 was the user's hedged lean**, "I'd lean against making the actor stage influence the dynamics ... It might
  actually be something we do later down the line", not a rule.
- **Positions, verified:**
  - **Prediction-only world, behaviour on top.** In DreamerV3's reference code, actor-critic gradients through
    imagined features are stopped (ac_grads False). But reward prediction AND a value loss on real replay do shape the
    representation (reward_grad True; repval_loss and repval_grad True; scale 0.3).
  - **Goals shape the shared network with prediction kept:** Dreamer 4, phase 2.
  - **End-to-end goal-shaped:** MuZero; TD-MPC2's decoder-free latent.
- **Continual-learning side:**
  - Task-general world models adapt to new tasks fast (Plan2Explore).
  - Task-specific models are simpler, but the class shrinks to the true model as tasks multiply (Grimm et al. 2020).
  - Keeping the imagination engine general protects later tasks.
- **Brain side:**
  - Learning a task reshapes even V1 "to adjust its processing to task requirements" (Poort et al. 2015).
  - The mechanism proposed is not replacing prediction with a task objective but gating plasticity: a global reward
    signal times attentional feedback that "limits plasticity to those units ... crucial for the stimulus-response
    mapping" (AGREL, Roelfsema & van Ooyen 2005).
- **So there are three options, not two:** G0 detached; G1 goal gradients with prediction kept; G2 goal-gated
  predictive learning (reward and attention set where and how much the world learns, while it learns prediction).
- **Current lean:**
  - G1 for the belief, on real experience (DreamerV3's replay-value precedent);
  - G2 for the denoiser in time (the privileged route of 11-15 is G2);
  - G0 until then.
- Codex: goal-shaping the belief changes the world's conditioning (memory tokens), so prediction must keep training
  alongside it.

*3. Where does the head start come from, realistically?*
- **What a deployed agent actually has:**
  - abundant observation without action labels (video, others acting);
  - a little of its own interaction;
  - sometimes demonstrations on its own body, which are expensive teleoperation;
  - its own life.
- **Practice:**
  - Dreamer 4 learns most of its world "from diverse unlabeled videos" with action conditioning from little labeled
    data;
  - R3M pretrains on human video (Ego4D) and then learns tasks "given just 20 demonstrations";
  - Open X-Embodiment pools teleoperation from 22 robots;
  - DayDreamer learns on the robot from scratch in hours.
  - VPT trains an inverse dynamics model on a little labeled data to label "a huge unlabeled source", yielding a
    behavioural prior for "hard-exploration tasks that are impossible to learn from scratch via reinforcement
    learning".
- **Biology:**
  - children "copy others' actions with high fidelity", from observation, without the demonstrator's motor commands;
  - imitation at birth is contested (Oostenbroek et al. 2016).
- **So:** imitation from labeled motor footage (our corpus's actions) is the comfortable option, realistic only where
  teleoperation exists. The faithful path:
  - the world learnt mostly from observation;
  - its own actions' effects learnt from a little own interaction;
  - behaviour learnt on the job, by living and dreaming;
  - imitation, if any, by watching others and inferring their actions with the agent's own world model
    (inverse dynamics, VPT's route).
- **Design consequence:** labeled-demonstration imitation is a permitted accelerator, never a requirement.

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
- **The two channels depend on each other.** Corrected 11-22 (Codex): unlabelled mixed experience doesn't teach an
  average. A generative model trained on rule-A and rule-B drinks with no way to tell them apart learns a calibrated
  coin (+1 or +2). It is still right only at the base rate. Recognising from context which rule holds is what would
  let it be right in each.

### What the research says about the two channels (read at the source, 11-22)
- **How much one experience should move a belief is itself learnt.**
  - People "assess volatility in an optimal manner", and their learning rates follow it (Behrens et al. 2007).
  - Uncertainty about a causal relationship "induces rapid changes in the rate of learning", switching between
    incremental and one-shot learning (Lee, O'Doherty & Shimojo 2015).
  - In a world that has never varied, a single surprising outcome should move a learner very little.
- **Fast learning sticks when it fits existing structure.** Rats trained on a flavour-place schema learned new
  associations "for only one trial" and consolidated them rapidly (Tse et al. 2007). Neocortical learning "can be
  rapid for information that is consistent with known structure" (Kumaran, Hassabis & McClelland 2016, CLS updated).
- **Slow learning builds the fast learner.** Dopamine-driven plasticity trains the prefrontal cortex "to operate as
  its own free-standing learning system" in its activity (Wang et al. 2018). This is learning to learn, as in Harlow's
  learning sets (Harlow 1949, not read; the phenomenon is confirmed in Murray & Gaffan 2006).
- **Machine precedents:** AdA and Nagabandi et al. get in-context adaptation by training across many variants, and it
  works within the family of variation they trained on.

**For Saccade:**
- The frozen world behaved as a learner raised in a world that never changed should: its experience gave it no
  volatility and no schema in which drink value is a variable.
- A quantity that never varied is indistinguishable from a law, so in-context adaptation to a rule needs some prior
  experience of variation, or prior structure that treats the quantity as a variable.
- Open: how much variation, of what kind, and whether variation in some quantities teaches "things vary" in general.
- The record is sensory memory: it recorded the +2 jumps faithfully and shouldn't change its nature. What should move
  is the belief and the world's use of it, by the amount the agent's learnt volatility warrants.

## Three kinds of learning a new rule (11-22; what we expect, stated before any fine-tune)

A changed rule can be learnt in three different ways. They test different faculties, and each fine-tune must say
which one it tests. Latent-cause theory (Gershman et al. 2010, 2017) frames the difference: old memories are modified
when new observations are inferred to come from the same cause, and new memories are formed when a new cause is
inferred. Extinction "does not destroy the original learning, but instead generates new learning that is especially
context-dependent" (Bouton 2004).

| Kind | Example | What the agent should do | What our world does today (measured) |
|---|---|---|---|
| Revision: the rule changed for good | Water now gives +2, everywhere | Update the old rule. Animals keep the old one, tied to its context (renewal); networks overwrite by default | Untested. A rule-B-only fine-tune measures speed and interference |
| Differentiation: a new rule tied to a visible context | Water gives +2 at night | Learn a conditional rule beside the old one | Craftax already has such rules: zombie damage 7 asleep against 2 awake; hunger and thirst at half speed asleep; health recovery doubled asleep; zombie spawning 0.02 + 0.1 (1 - light)^2. Learnt slowly and partly: asleep beside a ready zombie, P1.5 draws death 0.844 (true 0.992) |
| Inference: a hidden cause, inferred from evidence | In some places or episodes water gives +2, unannounced | Infer from a few observations which version holds here, and keep both | Works for hidden STATES that varied in training: clock phases (on-schedule pairs 0.995 / 0.000, carried by the record), cooldowns. Fails for a rule PARAMETER that never varied (water yield: 20261122_rule_change) |

**Why these differ:**
- Revision tests plasticity and interference.
- Differentiation tests conditional learning without interference.
- Inference is the fast channel, and the selling point ("adapts to a new situation") depends mostly on it.
- A visible context (sleep, light) makes a rule a fixed function of what is seen: the slow channel learns it, and
  nothing needs inferring. So a day/night rule cannot measure the fast channel.
- The fast channel needs something hidden that varies.
- In Craftax-Classic the hidden things that vary are states (phases, cooldowns, mob health), and our world infers
  those. No rule parameter varies across episodes. Of the ones that could, day length and spawn chances are
  already parameters of the environment.

**Requirements this implies (blueprint, not modules yet):**
- The agent needs some way to attach new learning to a situation (a latent cause or context), rather than only
  overwriting or only averaging.
- How much an experience moves it should depend on surprise and on its sense of volatility (Behrens; Lee).
- Its representations should share structure across things (consumables, recipes, creatures), so that one
  experience fitting known structure can count (Tse; Lake).
- Each requirement comes with a probe. A module is built only when a measurement shows the gap on the running loop.

## Where variation comes from: the three options scoped (11-22; parked as the future judge)

**Evidence availability** (`artifacts/experiments/20261122_rule_change/variation_eda.py`, DEV labels, 1,009
expert-mixed episodes, median length 212 steps). For each rule: the share of episodes that see the event revealing its
value, and the median step of the first and second sighting.

| Rule | How its value shows | Seen in episodes | First / second sighting | Native parameter? |
|---|---|---|---|---|
| Thirst / hunger / fatigue period (20 / 25 / 30) | Interval between HUD drops | 100% / 100% / 100% | 21, 42 / 26, 57 / 31, 80 | No |
| Zombie damage awake (2) | Health drop beside a zombie | 95% | 112, 155 | No |
| Table cost (2 wood) | Inventory drop on placing | 76% | 28, 99 | No |
| Zombie cooldown (5) | Repeat hits by one zombie | 78% | 148, 166 | No |
| Night onset (day length 300) | Lighting | 71% | 148, 448 | Yes |
| Zombie damage asleep (7) | Health drop beside a zombie | 68% | 194, 272 | No |
| Cow / zombie health (3 / 5) | Hits until it dies | 56% / 54% | 92, 175 / 106, 196 | Yes |
| Water yield (+1) | Drink digit after drinking | 48% | 117, 208 | No |
| Cow / plant food (+6 / +4) | Food digit after eating | 23% / 15% | 293 / 152 | No |
| Spawn chances | Statistical; fresh spawns are never seen in view (Codex) | — | 29-199 trials for 80% (Codex) | Yes |

**The options:**
- **Native parameters only** (day length, mob health, spawn chances): least hand-made, but few. Spawn rates give very
  weak evidence, and day length matters mostly near dusk (Codex). Too narrow to teach that things vary.
- **A family of hidden rules varied per episode** (native where they exist, written by us otherwise): this is domain
  randomization. The robotics precedents:
  - Peng et al. 2018: randomizing dynamics yields "policies that are capable of adapting to very different dynamics";
  - OpenAI 2019: memory models trained on randomized environments "show clear signs of emergent meta-learning at test
    time";
  - Tobin et al. 2017: "the real world may appear to the model as just another variation".

  It works with our senses, actions and corpus: exact replay of recorded actions under changed rules is verified, and
  so is the in-logic patch (Codex). It spans several evidence channels: timers, creatures, yields, lighting.
- **Craftax-Classic, then full Craftax:** natural novelty that fits Classic's schemas, but it changes the screen (130 x
  110, 9 x 11 view, 10-pixel blocks, 4 inventory rows) and the actions (43). There is no expert corpus for it, and the
  Craftax authors report existing methods "fail to make material progress". Best kept as the later lifetime test of
  transfer, once perception is geometry-general.

**Proposal (11-22, undecided):**
- The family of hidden rules is where the fast channel would come from.
- Start with the rules whose evidence comes earliest and most often, and whose channel the world already uses for
  inference:
  - the clock periods. The world already infers clock phases in context; inferring the period from the interval
    between drops is the next step on the same channel;
  - then creature rules (damage, health, cooldown);
  - then yields and day length.
- **Hold water yield out entirely, as the transfer test.** If varying other rules teaches the agent to adapt to the
  never-varied water +2, then "things vary" generalised. The rule-change benchmark and its matched histories already
  exist.

**Benchmark notes from Codex (11-22, kept for later):**
- On ordinary replays, elapsed time plus the current drink digit identify the regime 81.8% of the time, so the test
  must vary the earlier interval with the current evidence held fixed.
- Start with thirst alone (930 DEV episodes with a clean interval, completed at a median step of 42), and test a
  held-out thirst period before cross-family water.
- Replayed actions were chosen for the old rule.

**Open:** the record and the belief reset at every episode start. Rules that persist across a lifetime of episodes can
only be carried by the weights unless some memory survives the reset.

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
