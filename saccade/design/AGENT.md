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

**What agentic Saccade needs, in four parts (the user, 11-23: "we might not necessarily want to overcomplicate
this"; minimal first, extended on the failure modes we hit, as Saccade itself was built).**

**"Minimal" (the user, 11-23, correcting my "first loop").** The architecture has everything it needs to excel, and is
extended only when a new problem appears. It is not an intentionally stripped-down version. The whole system is built
and run. If a change touches the world, P1 is rerun, rather than ad-hoc half-truth tests on a frozen world. The
exception is minimal changes, where running the existing world with the agent has value of its own. So there are no
staged "first loop" variants (frozen world, uniform gaze); those were mine and are withdrawn.
1. **Choosing looks and actions from the system itself.**
   - Agent tokens inside the world.
   - The separate chooser and the P1.6 world-eye retire as separate parts (the user dislikes the chooser being
     separate). Their job moves into the agent's own look tokens. The look is **targeted from the start**, in
     watching, in life and in dreams: it is one of the architecture's main points. A small uniform floor stays for
     exploration, as the chooser kept 0.2 / 49.
     - **Withdrawn: my "uniform in life and in dreams".** It was test hygiene, a fixed controller for a clean
       comparison. It contradicted the 11-22 agreement ("the eye belongs to the agent, and its goal-directed attention
       decides what the non-selective world learns in detail") and CURRICULUM ("part of the actor's action during
       imagination").
     - What the old eyes bought the world was mixed: P1-F2's learned gaze against the same world under random gaze
       gained coal, iron, cows and skeletons, and lost wood, stone, zombies and waking. Under P1.6's world-eye "the eye
       looked more at mobs, and the world gained nothing on mobs from it" (P1.6 RESULTS).
     - My first argument, that the old eyes "sample almost uniformly" (window entropy 5.58 of 5.61 bits), was wrong
       (Codex, 11-23, `20261119_s2/RESULTS.md`, coverage audit). Entropy is broad over frames; per frame the chooser
       was selective:
       - entities covered 56% against 20% under uniform windows;
       - the health cell after a drop 38% against 4%.
       - Kept as a reference for the new eye.
     - Uniform windows give an interior cell up to 9/49 coverage, a corner 1/49.
   - **What teaches the look (gaze has no demonstrations).** The look learns from the system's own signals, emergent
     rather than top-down:
     - **the world's surprise and the action's contingency**, inherited from the chooser (0.8 softmax(0.5 C + 0.5 S) +
       0.2 / 49). This is the curriculum the user expects it gives the world, kept as the look's intrinsic objective
       instead of as a separate network. Whether it beats uniform looks for the world's training is unmeasured on the
       same recipe; P1-F2 against the same world under random gaze was mixed;
     - **the decision value of a crop:** which crop improves the next decision, measured on real recorded frames by
       revealing different crops of the same frame to the same choice faculty (Codex).
       - The decision-value EDA (`artifacts/experiments/20261123_gaze_decision_value/RESULTS.md`, 11-23) found none
         detectable: full resolution minus the same window at periphery acuity is -0.002 nats per decision, night
         +0.0004, under P1.5's old chooser -0.002. The instrument is weak (a one-epoch probe, 0.45 nats of action
         information), so this is a lower bound.
       - With three earlier measurements (the coarse view holds every HUD value and the health digit exactly; the
         best look saves 0.25-0.6% of next-frame loss), **Craftax-Classic at 3x3 acuity gives the eye little
         information work.** The full system's own read will show whether decision value is there. If not, the look's
         signal here is the world's surprise, and the eye's value is night creatures and economy (dreams generate
         only the looked-at crop).
       - **The user (11-23): known for a long time, not surprising.** We stay with Craftax-Classic and test the eye in
         full Craftax later, which is the true out-of-distribution test anyway. The user's hunch is that the eye will
         still be useful, even in Classic. Acuity stays as it is: no lowering just to make the fovea useful (11-21);
     - **dream returns,** once look schedules can't change later outcomes. The step order already guarantees the
       step's own outcome.
   - **Precedents:**
     - glimpse policies learnt from task reward (Mnih et al. 2014);
     - people fixate near-optimally to gain information in search (Najemnik & Geisler 2005);
     - information seeking ranges "from automatic biases toward novelty or surprise, to systematic search for learning
       progress and information gain" (Gottlieb et al. 2013);
     - infants may learn gaze following because looking where a caregiver looks yields interesting information, a
       reinforcement account (Triesch et al. 2006, via a 2021 summary);
     - eye movements in REM follow the dreamt gaze (Senzai & Scanziani 2022).
2. **Turning new experience into something kept.** Three different speeds:
   - **in context:** memory that changes predictions with no weight change. The record and belief hold the evidence,
     but the world doesn't use it (measured); that needs training across variation;
   - **in the weights within the life:** online updates and replay of the new real experience;
   - **after one exposure:** an episodic store, later.
   - "How fast is fast" picks among these.
3. **Learning continually.** What trains what is mostly set:
   - the world by prediction on real experience: the corpus while watching, its own transitions while living. The
     hypothesis is that a world good enough at P1, including on its own outputs, needs no rollout training (P2 is a
     possible later stage; Decision 2, sub-decision 5);
   - the choice and its value by imitation, and by dreams through their own weights;
   - the look by its signals (part 1);
   - goal shaping on hold.

   Open: the learning rate (no annealing to zero), plasticity tools on a measured trigger, and surprise-driven
   learning later.
4. **Acting sparsely, practising long offline.** Mostly schedules, with two jobs offline:
   - replaying real experience for the world;
   - dreaming for the choice, the look and the value.

   Dreams start from coherent snapshots.

The decisions below fill in these four parts. The first proof is an agent that lives, dreams and improves. In order:
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
- **The belief only** (STEP_SPEC section 5; m0.yaml p3.features). Simplest, but it reads through the 256-number
  bottleneck, which lacks spatial detail and the clocks the world uses.
- **The belief plus the current percept,** a separate network. It recovers current detail, but not the record or
  the world's understanding, and it is an actor in all but name.
- **Agent tokens inside the world (Dreamer 4).** The choice is a readout of the same computation the world uses
  to imagine:
  - it reads the current frame's tokens, the recent frames, the record (through each cell's read) and the belief
    (through the memory tokens);
  - one-way attention keeps the world's predictions from depending on the agent's own choices;
  - our denoiser already has one-way groups (physical queries never read glimpse keys), and every real or dreamed
    frame already gets one clean context pass, so reading it adds little cost.

**Laid out again 11-22 (the user: "stop treating Saccade as actor + model"; "10-24 isn't a rule"). Sub-question 1
agreed; 2 and 3 researched again (below).**

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
  - Look then act shares one pass's computation, split at the crop (Codex; corrected 11-22 against
    saccade/model.py `sample`):
    - physical tokens never read glimpse keys;
    - the sampler already generates the periphery and lighting first, sets the fixation between the stages
      (`choose(p, v)`), and keeps the physical keys and values for the glimpse steps;
    - the completed-frame context pass runs after the crop. Splitting it the same way puts look tokens in its physical
      half, before the crop, and motor tokens after.
  - Cost: unmeasured. The P1.6 world-eye's +12 h came from a separate sequential estimate pass, not from this.
- **Agreed (user, 11-22): agent tokens inside the world.** The look becomes one of the agent's two outputs from
  the start, chosen by the same computation that picks the motor action, and the glimpse informs that action. The
  P1.6 world-eye's separate estimate pass goes.
- **Ordering:** look then act, inside the split pass above. (My 11-22 "two passes per real step" overstated the cost.)
- **Costs to measure:**
  - time per real step and memory at 6 GB;
  - one-way attention verified: the world's outputs are unchanged when the agent tokens change;
  - prediction quality unchanged.
- **Codex (11-22), not yet answered before this note:**
  - **Information versus computation.** Codex would start from belief + acquired spatial evidence + a record read, then
    test what denoiser activations add on that same interface: comparing the belief alone against agent tokens changes
    the information and the architecture at once. The user chose agent tokens. Once they work, the clean reading is
    agent tokens attending to the denoiser's inputs only versus to its activations, with the same weights and heads.
  - **Efficiency.** Measure the decision gain and the extra cost while awake. Reading the denoiser may pay mostly on
    rare, consequential decisions, so the question is how often the agent needs that computation to act well. The
    brain arbitrates between costly model-based and cheap habitual control by their uncertainty (Daw, Niv & Dayan
    2005).
  - **The look has no demonstrations.** Craftax players saw the whole screen, so the corpus has no gaze to copy. The
    look learns from the system's own signals, targeted from the start (part 1 above: the chooser's surprise and
    contingency, the decision value of a crop, and dream returns once look schedules can't change later outcomes). In P1.5's dream, reward and termination are sampled from features that include the generated crop
    (s2b.py `outcome.features`), which is where the 0.12 pp / 5 pp inconsistency enters.
- **Build checks (Codex 11-22; I reran the four existing sampler tests it cites, test_m0.py: physical outputs ignore
  the glimpse and fixation, glimpse-only step equals the full step, streaming equals chunked, cascade equals recomputed
  history: all pass):**
  - swap the current crop with the pre-look inputs held fixed: the look's logits must be identical, the motor logits
    may change;
  - swap the recorded motor label: the logits predicting that action must be identical. A training batch holds the
    crop and the label together, so imitation can score well on information the awake agent only has later;
  - the motor choice imitates the corpus under the agent's own looks. The crop-swap check above keeps the imitation
    honest whichever window was seen.

*2. Which parts should the agent's goals shape, and how? (Researched again 11-22. ON HOLD, user 11-22: "put this on
hold until we understand the rest of the system better".)*
- **What it is, plainly (the user asked, 11-23).** Every part learns from some loss:
  - the world (encoder, denoiser, record) learns to predict what it will see;
  - the choice and value learn what to do and how good situations are.

  Goal shaping is whether the losses *about the goal* also reach the shared parts that model the world. Those losses
  are reward prediction, the value, imitation of actions, and returns in dreams; the shared parts are the belief and
  the denoiser. If they do, the agent's model of the world bends toward what matters for its current goal.
  - **The upside:** a focused representation. The brain does reallocate: V1 is reshaped by a task (Poort 2015), and
    dopamine paired with a tone enlarges its map (Bao 2001). Ours: the reward-forecast heads taught the belief the
    clocks.
  - **The downside:**
    - predictions bent toward one goal transfer worse to the next;
    - imagined returns make the world optimistic;
    - value targets that never stop moving cost plasticity.
  - **Today:** P1.5 already does a little. Its reward and continuation heads train Mamba (route_by_role, 98% of the
    belief's gradient).
- **The default while on hold** is today's: reward and continuation train Mamba; the choice's and the value's losses
  stay in their own weights; the denoiser learns by prediction on real experience. Codex agrees with prediction-only
  updates for the denoiser at first.
- **Naming:** my earlier labels G0/G1/G2 collided with the loss groups (G1 next percept, G2 grounding, G3
  consequences). They are dropped; the question is laid out by route.
- **10-24 was the user's hedged lean**, "I'd lean against making the actor stage influence the dynamics ... It might
  actually be something we do later down the line", not a rule.
- **What P1.5 already does (saccade/train.py, losses.route_by_role):**
  - the reward, continuation, forecast and gaze heads train Mamba and its input maps through a second pass over
    detached percepts, never the encoder;
  - so the belief is already goal-shaped on real experience. In P1-F6 the reward forecast's weighted gradient into
    the belief was 4.54, against G1's 0.031 (cosine to the total 0.997);
  - it is also the only loss that taught the belief clocks (recovery, cooldown; P1-F5);
  - the denoiser is trained by prediction alone.
- **Six routes by which a goal can shape what the agent learns:**
  1. **Behaviour: what is experienced.** Inherent in a living agent. Dyna corrects model errors where the agent goes
     (Sutton & Barto 9.3).
  2. **Gaze: which evidence enters, and what is learnt in detail.**
     - Brain: with identical stimuli, practising one attribute doesn't improve another (Ahissar & Hochstein 1993).
     - Ours: P1-F2's gaze learns.
     - The risk: looking must not change outcomes. On P1.5, two passive looks (map vs HUD) change the one-step
       positive-reward probability by 0.12 pp on average, but by 5.0 pp at one branch (71.6% -> 76.6%, episode 546
       frame 61; Codex, recomputed from the saved draws). A gaze trained on dream returns could learn to look where
       the dream is kind.
  3. **Practice selection: what is replayed and dreamt.**
     - Field: prioritised replay (Schaul et al. 2016), which "introduces bias because it changes this distribution in
       an uncontrolled fashion" and corrects it with importance weights.
     - Brain: replay by utility (Mattar & Daw 2018). Sleep favouring memories expected to matter (Wilhelm et al. 2011)
       did not replicate (Ashton & Cairney 2021).
     - Ours: outcome-selected terminal tails installed an action-prior shortcut, and broad hazard-choice sampling
       fixed it (08-29).
  4. **Goal targets from real experience into shared modules.**
     - Field:
       - DreamerV3's reference code: reward and a value on real replay train the representation (scale 0.3);
       - Dreamer 4 phase 2: imitation + reward on the agent tokens, video loss kept;
       - MuZero / TD-MPC2: goal targets only.
     - Cautions:
       - a decoupled encoder "matches or outperforms end-to-end RL" (Stooke et al. 2021);
       - reward-only latents warp (Visualizing MuZero);
       - non-stationary targets cause "capacity loss" (Lyle et al. 2022), and value targets never stop moving in a
         life.
     - Brain:
       - task learning reshapes V1 (Poort et al. 2015);
       - dopamine paired with a tone enlarges that tone's cortical map and shrinks its neighbours' (Bao, Chan &
         Merzenich 2001);
       - only the feature paired with task targets is learnt, even when subliminal (Seitz & Watanabe 2003).
       - Goals allocate representation; none of these makes predictions favour good outcomes.
  5. **Outcome-weighted prediction: how much the world learns from each moment.**
     - Field: VaGraM rescales the model's error by the value gradient (Voelcker et al. 2022).
     - Brain: three-factor rules (Fremaux & Gerstner 2015).
     - Ours: rare-event doses shifted priors (the 10-02 consequence-learning study).
  6. **Imagined returns into the world: what the world predicts.**
     - Stopped in DreamerV3 (ac_grads False).
     - Stopped in DayDreamer, stated as the reason, not ablated: "this would lead to incorrect and overly optimistic
       model predictions".
     - Dreamer 4 phase 3 trains heads only; fine-tuning the whole transformer gave "small additional benefits" in its
       one-off offline setting.
     - MnM trains model and policy on one bound, and its learned dynamics turn out "optimistic" (Eysenbach et al.
       2022).
     - Ours: every decision-critical error we know makes imagination safer than reality.
- **Two principles that follow:**
  1. **Goals shape shared modules only through targets the world didn't produce.**
     - Real rewards, real terminations, demonstrated actions and values fitted on real experience can't make the world
       believe something false about the world.
     - An imagined return is the world's own output. Training the world on it teaches it to tell the choice what it
       wants to hear.
  2. **A goal may set how much the world learns from a moment, using only what is known before the outcome.**
     - With a weight w(x) on the context, the per-context minimiser of a proper loss is still p(y|x). Each prediction
       stays calibrated; only capacity moves.
     - With a weight w(x, y) that depends on the outcome, the minimiser becomes w(x, y) p(y|x) / Z:
       - weighting by realised reward is optimism by construction (Codex);
       - weighting by surprise over-predicts rare outcomes (PER's bias; the 10-02 consequence-learning study).
     - Allowed: where it looked, the value of the situation, the uncertainty before the outcome. The 11-15 privileged
       glimpse route (attention as precision that gates learning) is of this kind.
- **Codex's corrections to both principles (11-22, checked, conceded), for when this resumes:**
  - Principle 1 is too broad. Shortcut forcing already trains the denoiser on its own detached predictions
    (model.py `shortcut_loss`, the bootstrap target), which teaches it to reproduce a predictive computation more
    cheaply. The line is the incentive, not who produced the target. Codex's version: grounded prediction trains the
    denoiser; imagined returns train the choice and its value.
  - Principle 2 must say "information the predictor conditions on", not "known before the outcome". If the selector
    sees a cue the predictor's input lacks, w(x, z) with z unseen gives E[w p(y|x, z) | x] / E[w | x], not p(y|x).
    This bites with the eye: a collector that sees a danger cue the foveated input missed would change the world's
    apparent damage rate even when it selects before the hit.
  - The P1-F6 share (reward forecast 4.54 vs G1 0.031 into the belief) shows dominance on the measured chunk, and the
    token forecast's tiny dose. Whether that dominance hurt prediction is untested: the matched continuation
    (20261115_g1_share_control) was stopped before any reading. A stopped gradient at the denoiser still leaves a
    route through Mamba's changing state.
- **Proposed position (undecided):**
  - routes 1-3 from the start, with outcome-consistent looks before any gaze is trained on dream returns (Codex), and
    the world's own training data selected by context, a broad base kept;
  - route 4 into the belief: keep reward and continuation; add the choice's imitation loss. Do this after a first
    imitation run with the world frozen (Codex: "can a choice head learn to act well from the world and memory we
    already have?"), judged by real decisions and by dreamed consequences under matched histories. A value on real
    replay is left to decision 3, because of capacity loss;
  - route 4 into the denoiser, at first: no. The agent tokens get their own weights. They read the denoiser's
    activations detached and the belief's memory tokens attached, so the imitation gradient reaches the belief and
    stops at the denoiser. Reason, from our own system: where goal and prediction losses already share a module (the
    belief), the goal losses took about 98% of its gradient (P1-F6; whether that hurt prediction is untested, above).
    The denoiser is the module whose job is the world's rules. This differs from Dreamer 4 phase 2 (shared weights);
    revisit if the choice lacks something the denoiser's features could give it;
  - route 5 with context weights only; route 6 not at all (dreams train heads only).
- **Codex:** goal-shaping the belief changes the world's conditioning (memory tokens), so prediction must keep training
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
- **Revised 11-22 (the user: a robot isn't deployed untrained for its task; we must tell and show it what we want).**
  I withdraw "labeled-demonstration imitation is a permitted accelerator, never a requirement". A deployed agent must
  be told what to do, and the channels are few:
  - a reward, designed or learnt;
  - demonstrations (show);
  - language (tell);
  - goal examples;
  - corrections (coach);
  - preferences.
  DayDreamer is the exception: hand-designed rewards per task (a five-term walking reward, sparse grasp rewards), one
  task per run.
- **Imitation has two separable roles:**
  - **specification,** what to achieve, needed in some form;
  - **competence,** how to do it, optional but large (VPT).
- **Deployed practice combines showing, coaching and own experience:**
  - pi*0.6 / RECAP (2511.14759): demonstrations, autonomous rollouts and expert corrections during execution, with an
    advantage-conditioned policy. On the hardest tasks it more than doubles throughput and roughly halves failures;
  - Sirius (Liu et al., RSS 2023): shared human-robot autonomy, and the policy improved from deployment data (+27%
    success on real hardware);
  - LWD (2605.00416): fleet offline-to-online learning.
- **Humans are built to be taught:**
  - natural pedagogy transmits "cognitively opaque" knowledge that observation alone would make hard to acquire (Csibra
    & Gergely 2009);
  - 3-4-year-olds copy demonstrated steps even when a transparent box shows them useless; chimpanzees drop them
    (Horner & Whiten 2005).
  - Teaching is a natural channel, not a crutch.
- **The phases are not stages.** One loop with three sources whose mix shifts over a life:
  - others' experience: watching, demonstrations, instructions;
  - own experience: acting, looking, being corrected;
  - imagined experience: dreaming.
  The world learns from all real experience. The choice learns from whichever target is present: a demonstrated action,
  its own outcome, an imagined return. The architecture is the same throughout; only the mix changes (early life
  mostly watching, deployment mostly own experience, idle time dreaming).
- **Watching without motor labels** (video, others acting) still needs inferred actions: VPT's inverse dynamics,
  through the agent's own world model.
- **For the first Craftax loop:**
  - watching is the corpus with its actions (motor labels kept, Codex; realistic where teleoperation exists);
  - own experience is live Craftax;
  - specification is the reward plus the demonstrations.

### What it is for: use cases and what they imply (11-22)

- **Home and service robots:** every home is a new world (layout, objects, people, habits). The owner teaches by
  showing and correcting. No dangerous trials. Learning stays on the device, which keeps it private. Idle and charging
  time is when it dreams.
- **Fleets (warehouses, factories):** many bodies sharing experience, frequent re-tasking (LWD).
- **Remote and field robots (planetary, underwater, disaster sites):**
  - no live teaching: Mars is up to about 20 minutes away each way, so flight software "must be resident on the in
    space asset" (NASA mission-operations reports, via search; the exact report for each quote not pinned);
  - damage and wear: DayDreamer's stated limitation is wear "that may require human intervention or repair"; with a
    prior, a damaged hexapod recovers in under two minutes (Cully et al. 2015);
  - so competence must be in place before deployment;
  - irreversible outcomes must never be under-predicted.
- **Assistive devices:** personalisation to one body over time. Human-in-the-loop exoskeleton optimisation cut walking
  metabolic cost by about 24% (Zhang et al. 2017).
- **Games and virtual agents:** cheap, safe experience. Craftax sits here.

**What they share, so what Saccade needs:**
- **Real experience is the scarce resource**, so imagination and selective sensing are the levers (the intelligence
  loop).
- **A specification channel,** mostly demonstrations and corrections, with rewards learnt from them.
- **No resets.** One continuous life needs memory that survives episodes (the episodic store). Craftax's episode reset
  is a convenience, not the design.
- **Safety:** errors on irreversible outcomes must never be optimistic. Ours are optimistic: deaths and hits are
  under-drawn.
- **Economy:** compute and energy on the device. In the brain, signalling takes a large share of the energy budget,
  which favours energy-efficient codes (Attwell & Laughlin 2001). The eye is Saccade's version.
- **A changing body and world:** DayDreamer's XArm lost performance at sunrise and recovered in about 5 hours, "faster
  than it would be to train from scratch", "with no change to the learning algorithm".

### DayDreamer, compared (2206.14176, read at the source 11-22)

**The same as Saccade:**
- online learning on the real system;
- a world model learnt from a growing replay buffer, and behaviour learnt in imagination;
- a learner thread and an actor thread in parallel;
- the world model untouched by the actor-critic's gradients;
- adaptation to pushes within 10 minutes.

**Different:**
- No head start: "without simulators or demonstrations", from scratch for each task, with hand-designed rewards.
- One task per run, and no continual-learning machinery. It notes "promise for using Dreamer in continual learning
  settings".
- The whole observation every step; no active sensing.
- An RSSM with reconstruction: short memory and no episodic store.
- A world model plus a separate actor-critic.

**So:** it proves the loop runs on real hardware in hours. Saccade's claim is the parts it lacks:
- starting from watching;
- selective sensing;
- fast adaptation within one life;
- keeping what it learnt.

### Decision 2: dreaming (opened 11-22)

**The user (11-22): don't build the dream design around P1.5's deficits.** The fading mobs and the day coin are a
world-quality problem with measured causes, to be solved separately ("just a matter of building a good enough
world"). The design below assumes a good enough world. P1.5's measurements locate it, nothing more.

**What it is for:** practice for the choice and its value, branched from real moments, so that scarce real experience
goes further. **The judge:** the same continuing agent with and without dreaming, at the same real-interaction budget,
with compute reported (Codex; "How to judge it" below).

**Precedents, read at the source:**
- **DreamerV3 reference code:**
  - imagination starts from every replayed state (`imag_last: 0`);
  - 15 steps (`imag_length`), λ 0.95, a 333-step horizon (γ ≈ 0.997), continuation predicted (`contdisc`);
  - actor-critic gradients stop at the world (`ac_grads: False`).
- **Dreamer 4:**
  - "Imagined rollouts start from contexts of the dataset";
  - "we start only one rollout from each context, prioritizing data diversity";
  - rewards come from the reward head, values from λ-returns with γ 0.997, and the transformer is frozen;
  - the main text gives no horizon.
- **DayDreamer:** H = 16. **IRIS:** starts from "a real observation sampled from past experience", H = 20, and stops at
  a predicted episode end.
- **MBPO:** "short model-generated rollouts branched from real data"; short rollouts with ensembles are "sufficient to
  prevent model exploitation"; it separates the task horizon from the model horizon.
- **Ha & Schmidhuber:** they raise a temperature to make the dream "noisier and more uncertain", so that the
  controller can't exploit its imperfections.
- **Dreamer** never generates observations in imagination: the reward comes from the sampled latent state, and "the
  decoder ... is not needed while learning behaviors from latent rollouts" (DayDreamer).

**Brain:**
- replay starts where the animal is and runs toward remembered goals (Pfeiffer & Foster 2013), ordered by gain times
  need (Mattar & Daw 2018);
- eye movements in REM sleep follow the dreamt gaze (Senzai & Scanziani 2022; Andrillon et al. 2015): the eye acts in
  dreams too.

**What P1.5's dreams do today (measured, 20261119_s2):**
- **First-step consequences match the simulator:**
  - damage beside a zombie 0.170 / 0.169;
  - death 0.0069 / 0.0069;
  - unlock 0.017 / 0.017;
  - return per 16-step future 0.146 / 0.156.
- **Mobs fade on the world's own history:**
  - 0.94 / 0.83 / 0.71 / 0.55 mobs per frame at steps 1 / 4 / 8 / 16, against the simulator's 1.03-1.14;
  - cows are -11% at step 1;
  - the denoiser's read of the record of its own generated frames carries this loss.
  - The error is optimistic: danger disappears.
- **Zombie cooldown is wrong at depth:** 3.6x false hits while cooling, too few when ready.
- **Coal and iron follow DO at depth 1** only 0.34 / 0.22, against 0.88 / 0.92.
- **Decision value (Codex, 764 DO/NOOP starts):**
  - choosing the first action by 8 dreamed futures is slightly worse than always DO at every horizon (-0.011 at 16
    steps);
  - the simulator's own futures gain +0.008 [0.003, 0.014];
  - at one step the dream ties 74% of the time.
  - A narrow cohort, but no decision value yet.
- **Passive looks** move the one-step reward probability by 0.12 pp on average, and by 5.0 pp at one branch.

**Sub-decisions, in order (each discussed on its own):**
1. **What a dreamed step commits to. AGREED (user, 11-22).** In life, the consequences of the previous action exist
   before the agent looks; looking only reveals them. The order for a dreamed step:
   - draw the coarse view and lighting;
   - draw the outcome of the previous action (reward, termination) from what is committed so far, never from the look;
   - choose the look;
   - draw the glimpse given everything committed, the outcome included.

   This factorises the same joint as reality (state, then outcome, then evidence at a look), so the outcome's
   distribution can't depend on the look. Dreamer is the extreme case: its outcomes come from a committed latent, and
   observations are never generated. Today the outcome is read from features that include the generated crop.
   - **The two supporting checks (`artifacts/experiments/20261122_dream_commitment/RESULTS.md`).**
     - In a perfect world both orders factorise the same joint. What differs is which property holds by construction:
       - committing first makes the outcome exactly independent of the look;
       - reading the crop (today) makes the evidence agree exactly, while look-independence rests on the model's
         quality.
       - An eye trained on dream returns will find any incoherence.
     - **The committed coarse view holds the damage outcome exactly.** An exact renderer-glyph read of P1.5's coarse
       health token is right on all 43,115 DEV frames, catching all 962 drops with no false drops
       (`coarse_health_codebook.py`), as the 11-04 acuity finding said.
       - My first reading ("the coarse view does not"; "even a perfect world contradicts 4.7-11.5%") measured the
         weak readers' limits, the probe's AP of 0.966 and the coarse reader's 9.9% miss. It was corrected 11-23.
     - **P1.5's dreams today:**
       - the committed p tracks recorded outcomes with AP 0.950 (0.998 on the real successor);
       - the crop contradicts 3.6% of committed drops on the balanced panel, but **15.7%** reweighted to the event's
         real frequency, with five starts carrying 82% of it (Codex, verified).
       - This is model error, unresolved, and not a deployment rate.
   - **DECIDED (user, 11-23): the glimpse reads nothing new; no event code.**
     - The glimpse is already drawn given the committed coarse view, which holds the outcome. So the remaining
       contradictions are model error, which training and a better world reduce: the world-quality track.
     - An input would not enforce agreement anyway. Only the information structure and a measurement do.
     - Conditioning would be needed only where a committed outcome depends on detail that the committed state can't
       carry and the eye can reveal: two independent draws of one uncertain event, which no training can align. That
       is measured absent for Craftax damage. It is a trigger to measure for other senses and bodies, not a module.
     - Still true:
       - never the task's score (Codex: 39 of 962 DEV drops coincide with an unlock and have nonnegative reward, and a
         score changes with the goal while the scene doesn't);
       - the judge is agreement among the committed event, the coarse view and the fine evidence (Codex).
   - **Is incoherence a big problem?** Mostly not.
     - The step order already guarantees that a look can't change the step's outcome.
     - What remains is noise, not a bias toward good outcomes. The one second-order risk: a contradicting glimpse
       carried into the next step dilutes the committed consequence there, which thins the bad tail slightly.
     - The brain's dreams are far from coherent, and they still teach:
       - "memory consolidation in sleep is not necessarily associated with the veridical reiteration of an experience"
         (Wamsley et al. 2010, where maze dreams were fragments and predicted improvement);
       - dreams are "sparse, hallucinatory, and narrative" (Hoel 2021, who argues their strangeness helps
         generalisation);
       - Ha & Schmidhuber made dreams noisier on purpose to stop the controller exploiting them.
     - What matters is a systematic bias an optimiser can exploit, so two readings in the loop:
       - dreamed against real returns from the same starts;
       - whether later outcomes depend on the look schedule.
     - The outcome heads read the committed state (pre-look). In the full run they are trained that way from the
       start.
   - **Multi-step:** under a fixed action sequence, different look schedules must give the same distribution of later
     physical outcomes, averaged over what the looks could reveal (Codex). Looks may legitimately change the belief and
     what it predicts, but not the agent's luck.
   - **Codex:** keep "what happened" apart from "how the current goal scores it", for retasking. Craftax scores
     with the game's reward. A reward head that reads only committed state can be swapped for a new goal's.
2. **Where dreams start:** real moments from replay, uniform at first. The 08-29 lesson favours a broad base, and
   starts chosen by what the value conditions on leave it calibrated. Prioritised starts come later, on a measured
   trigger. A start needs the real belief, record and context cache at that moment.
3. **Horizon:** a value carries what lies beyond it. The general reason to keep it short holds even for a good world.
   Model error grows with how far the learning policy has moved from the data ("the model error increases with the
   divergence between the current policy π and the data-collecting policy", MBPO). A continual learner's world is
   always wrong somewhere new. Precedents use 15-20 steps.
4. **Returns:** λ-returns from dreamed rewards and continuation, bootstrapped by a value head (DreamerV3, Dreamer 4).
   Where the value is trained (dreams only, or real replay too) decides how much dream optimism it inherits.
5. **What learns from dreams: the choice, the look and the value. The world learns from real experience** (the user
   asked, 11-23, why the world doesn't evolve through dreams). I first answered that P2 was the canonical route, and
   was wrong. I hadn't reread what happened to P2 (corrected 11-23, user: "don't cite something as canonical unless you
   read everything about it"). "The world learns from dreams" means three different things:
   - **Its self-fed rollouts scored against what really followed (P2, CURRICULUM 2): a possible later stage, not the
     plan.** What was established 11-19 (`20261119_p2_decisions/RESEARCH.md`, sections 5-6):
     - **The sampling world models in our sources don't train on their own rollouts:** Dreamer 1-4, DIAMOND, IRIS,
       DRAMA, Horizon Imagination, LeWM. Deterministic latent models do, because their recorded targets stay valid:
       SPR, V-JEPA 2-AC, MuZero, TD-MPC2.
     - **The reasons:**
       - a sampled branch diverges, so the recording is no valid target for it (Huszar 2015);
       - a valid target is distributional, and so expensive and noisy;
       - one-step accuracy suffices if it holds on the model's own outputs (On Training in Imagination, Lemma 1).
     - **P2's score "never labels a transition wrong".** Only the 2 retained steps can blame the frame that lost a
       mob, and "nothing stops the later frames being pushed to add mobs back (phantoms)". The switching arm then
       showed that failure for real (the user, 11-19 21:53: "didn't we establish that P2 as we proposed wouldn't
       work?").
     - **The mob loss traced to the sampling chain:** the day coin, a calibrated hedge drained by the K = 4 chain
       (switching dissection). It is a P1-level problem.
     - **So the hypothesis (the user's, and the field's): a world good enough at P1, including on its own outputs,
       avoids P2.** P2 stays a possible stage if a measured failure calls for it, under its own contract:
       precommitted actions and a fixation schedule shared by the samples and the recorded target (Codex;
       `20261102_p1f15_readiness/P2_DATA.md`). Also self-generated training precedents: Self Forcing (Huang et al.
       2025), DaD (Venkatraman et al. 2015).
     - **What a life adds is new one-step evidence.** The awake agent's own real transitions train the world's P1
       losses online. It is not a free P2 target, since reality follows only the actions actually taken (Codex).
   - **Its dreams as if they were real experience. No.** They carry no new information, and indiscriminate training
     on generated data makes "tails of the original distribution disappear" (Shumailov et al. 2024). Our world's
     errors already sit in the tails (deaths and hits under-drawn), so this would make it more optimistic.
     - The exceptions, already or later: shortcut forcing's self-consistency (in use), and generative replay against
       forgetting, a consolidation decision. The brain's sleep replays real experience; NREM replay "can abstract
       rules from corpuses of learned information" (Lewis, Knoblich & Poe 2018).
   - **Imagined returns into the world:** goal shaping, on hold.
   - The choice, the look and the value learn from dreams through their own weights.
6. **The eye in dreams: targeted** (user 11-23). The agent chooses the look, and the look decides which glimpse is
   drawn, so the eye practises in dreams as in REM sleep. It trains on dream returns once look schedules can't change
   later outcomes. Before that it learns from its other signals (part 1).

The sub-decisions depend on one another, so they are discussed together (user, 11-22). The step contract anchors them.

**The dream settings for the full system (11-23; the user has no strong view; precedent defaults, declared in the
yaml).** No staged variants. The judge is the whole agent in real Craftax, with dreamed against real action
differences read in the run (Codex).

| Sub-decision | Default | Source |
|---|---|---|
| Starts | Real moments, uniform over replay, one rollout per start, each a coherent snapshot rebuilt from its stored real history | Dreamer 4: "one rollout from each context, prioritizing data diversity"; DreamerV3 starts from every replayed state |
| Horizon | 15 steps; a committed termination stops the return; the value bootstraps at the cutoff | DreamerV3 `imag_length: 15`; DayDreamer 16; IRIS 20. With λ 0.95 the full 15-step return carries 0.95^14 = 49% of the target's weight (Codex), so late-dream errors reach the first decision. Read them in the run |
| Returns | λ 0.95, γ 0.997, returns and advantages scaled by their 5th-95th percentile range | DreamerV3 (`horizon: 333`, `retnorm: perc 5-95`); Dreamer 4 γ 0.997 |
| Value | Trained on dreamed λ-returns *and* on real-replay λ-returns, through its own weights while goal shaping is on hold. Checked against real continuations of the evaluated policy | DreamerV3 `repl_loss`, `repval` 0.3. Codex: DreamerV3's replay target bootstraps from the imagined return, so it can carry dream optimism back |
| What learns | From dreams: the choice, the look and the value. The world from real experience (P2 only on a measured trigger) | Sub-decision 5 |
| Policy objective | Decision 3 (Dreamer 4's PMPO with a reverse KL to the imitation policy, or DreamerV3's actor-critic) | |
| The eye | Targeted in watching, life and dreams; trained by the world's surprise and contingency, the decision value of a crop, and dream returns once consistent | Part 1 |

**Parked (user 11-23): thinking when it's worth it.** Recorded in its own section below, to explore after the first
proof.

**Dreams in the whole system (11-22): what the component view would miss.**
- **Practice or deliberation.** Imagination can run in two places:
  - offline, as practice that trains the choice and value (Dyna);
  - at decision time, imagining before acting (TD-MPC2, MuZero).
  
  The brain does both. At maze choice points rats pause, and hippocampal sequences sweep ahead along the options;
  "as behaviour automates, these sweeps transition from going in both directions to going only in one direction, and
  then vanish" (Redish 2016). The pausing happens under uncertainty, and control passes between costly model-based
  and cheap habitual systems by their uncertainty (Daw, Niv & Dayan 2005). This is Codex's efficiency question: think
  only when a decision needs it. It changes the awake loop's compute. **First proof: offline practice only** (Codex,
  11-23). Deliberation is priced once imagined action comparisons show decision value.
- **Dreams never enter real memory as real.**
  - A dream runs on copies of the belief, the record and the context cache, and writes back only learning: the
    choice and value weights.
  - The future episodic store must not hold dreams, or must tag them.
  - The world is not trained on its own dreams as if they were real experience. Indiscriminate training on generated
    data makes "tails of the original distribution disappear" (Shumailov et al. 2024), and rare events are what
    decisions hinge on.
  - People face the same hazard: imagining an event inflates confidence that it happened (Garry et al. 1996); keeping
    perceived and imagined apart is reality monitoring (Johnson & Raye 1981).
  - Consolidation, if dreams ever train the world, is its own decision.
- **Trusting a dream.** A continual learner's world lags reality wherever something is new: after a rule changes, in
  places the policy has only just reached. There its dreams rehearse the old lesson and pull the choice against what
  real experience is teaching. Dream learning should depend on where the world is reliable. That couples dreaming to
  knowing what the agent doesn't know (uncertainty, change detection) and to the fast channel. If adaptation in
  context worked, a dream started from a recent real history would carry the new evidence and dream the new rule
  before the weights catch up.
- **Replay and dreaming are different jobs.** A dream can't teach the world something it doesn't already predict: it is
  the world's own output. Turning a new real experience into lasting knowledge means replaying the *real* experience
  to the world (the slow channel). The brain's sleep reactivates real experience (Wilson & McNaughton 1994). Dreaming
  from that experience then lets the choice practise around it. An offline period does both: replay of real
  experience for the world, dreams for the choice. How fast the slow channel is depends on how often a new
  experience is replayed, bounded by forgetting.
- **Fresh starts, from a coherent snapshot** (Codex). A start needs the belief, record and context at that moment,
  computed by the same model that rolls the branch forward, which matters once awake learning and dreaming run
  concurrently. The stored real history is what lets the start be rebuilt, or a cached one checked. A learning world
  makes stored states stale. DreamerV3 writes each replayed chunk's latent states back into replay
  after training and reuses them as the next start's context (`replay_context`). Our staleness audit measured stale
  substitutes as poor early in training and near fresh late (`artifacts/experiments/20261119_switching/STALENESS.md`).
- **Schedule.** The awake step must stay bounded for a real-time body: the split pass and nothing more. Dreaming runs
  in idle time or in a parallel learner, as in DayDreamer's learner and actor threads.
- **The eye makes dreams cheap too.** A dreamed step generates the coarse view and one glimpse, not the full frame. The
  dream's cost grows with looks, which the agent chooses. This is a synergy to keep.

### Decision 3: how each part learns, watching, living and dreaming (opened 11-23)

**Precedents, read at the source 11-23:**
- **Dreamer 4, section 3.3** (`third_party/papers/2509.24527v1.pdf`):
  - Imitation and reward are learnt on the agent tokens with multi-token prediction of length 8, and "we continue to
    apply the video prediction loss". No test of the prediction length is reported; it cites the LLM work.
  - "The behavioral cloning loss is applied only on the relevant fraction, while the dynamics loss is applied only on
    the uniform sequences to avoid optimistic generations" (data 50% uniform, 50% task-relevant). This is our
    principle 2 in practice.
  - The policy objective is PMPO, the sign of the advantages R^λ - v (eq. 11): (1 - α) / |D-| Σ_{D-} ln π - α / |D+|
    Σ_{D+} ln π + β / N Σ KL[π ‖ π_prior], with α 0.5 and β 0.3. The KL is reversed "to better constrain the policy to
    the space of reasonable behaviors". There is no return normalisation, and "the three objective terms ... are all
    measured in nats".
  - The value is trained on λ-returns with γ 0.997. Only the heads train, with one rollout per context.
  - "The world model representations outperform the general representations of Gemma 3 for behavioral cloning."
  - The JAX reimplementation (`third_party/sources/edwhu__dreamer4-jax/scripts/train_policy.py`) codes eq. 11 as
    written, with its own horizon of 32.
- **DreamerV3 code** (`third_party/sources/danijar__dreamerv3/dreamerv3/agent.py`, `imag_loss`, `repl_loss`):
  - REINFORCE on advantages scaled by the 5th-95th percentile return range, with entropy 3e-4;
  - a slow critic regulariser (slowreg 1.0);
  - a replay value bootstrapped from the imagined return (Codex's caution).
- **VPT** (Baker et al. 2022, read in the PDF):
  - fine-tuning keeps a KL to the frozen pretrained policy, because without it the agent "obtains only items early in
    the sequence" and later skills "are lost due to catastrophic forgetting";
  - the KL replaces the entropy bonus, because the prior is "much more likely to take sequences of actions that lead
    to interestingly new states";
  - ρ starts at 0.2 and decays by 0.9995 per iteration. That "protects policy skills in early iterations while
    guaranteeing that the policy can eventually maximize the reward function, regardless of how different its
    behavior has to be".
- **Teachers that fade:**
  - in kickstarting, the teacher loss's weight "is allowed to change over time, so that the student can ... potentially
    surpass the teacher" (Schmitt et al. 2018);
  - in Jump-Start RL, the guide policy's role shrinks as the learner improves (Uchendu et al. 2023).
- **Brain:**
  - an actor-critic in the basal ganglia, with dopamine as the critic's prediction error (Joel, Niv & Ruppin 2002;
    Schultz, Dayan & Montague 1997);
  - habits and deliberation arbitrated by uncertainty (Daw, Niv & Dayan 2005);
  - teaching as a natural channel (Csibra & Gergely 2009).

**Our demonstrations (EDA 11-23, S0 labels).**
- The expert is our PPO agent on symbolic observations. Alone (07-27) it scored 91.2, with 20.6 / 22 achievements and
  diamonds in 44% of episodes.
- The corpus mixes it with random actions at ε 0.1, 0.25, 0.5 and 1.0, a quarter of the 8,069 TRAIN episodes each.
  71% of actions are the expert's.
- At ε 0.1, 97% of episodes end in death (median length 478), with 17.6 achievements and diamonds in 9%. At ε 1.0 the
  figures are 2.4 achievements and length 135.
- So imitating the expert-flagged steps learns one strong policy over a broad, noise-perturbed spread of states. The
  expert acted on full symbolic state, so wherever it used something the screen doesn't show, the belief and record
  must carry it.

**Proposed (undecided), part by part, in one system:**
1. **The world** learns by the P1 losses on real experience: the corpus while watching, its own transitions while
   living. Goal shaping is on hold, so it stays as today. Following Dreamer 4, no outcome-selected data reaches the
   world's loss.
2. **The motor choice:**
   - **Watching:** imitation of expert-flagged actions, exogenous ones masked, with multi-token prediction of length
     8 (Dreamer 4's default, untested there). The crop-swap and label-swap checks apply.
   - **Dreams:** PMPO, α 0.5, plus a reverse KL with β 0.3 to an anchor policy (Dreamer 4).
   - **The anchor (revised 11-23 after the research below; the user asked what robots do).**
     - **How systems anchor, read at the source:**
       - **Robots mostly anchor through data:**
         - HIL-SERL keeps demonstrations and human corrections in an off-policy RLPD-style replay; corrections are
           "crucial for enabling the policy to learn from its mistakes" (Luo et al. 2024);
         - ResiP keeps a frozen BC model and learns an RL residual on top (Ankile et al. 2024);
         - AWAC's supervised actor update keeps the policy close to the data (Nair et al. 2020);
         - RECAP conditions on advantages over all its data;
         - WSRL warm-starts and then drops the offline data (Zhou et al. 2024);
         - DPPO fine-tunes the diffusion policy by policy gradient, with no BC regulariser in what I read.
       - **Agents and language models use an explicit reference:**
         - frozen, constant: Dreamer 4, β 0.3;
         - frozen, decaying: VPT, ρ 0.2 x 0.9995 per iteration;
         - fixed: InstructGPT's "per-token KL penalty from the SFT model";
         - moved during training: TR-DPO blends the policy into the reference or replaces it after N steps, +19% win
           rate (Gorbatovski et al. 2024).
     - **What an anchor is for:**
       - guiding exploration where values don't yet discriminate (VPT replaces entropy with it);
       - protecting latent skills until their value is realised (VPT);
       - keeping dream-trained behaviour where the world is accurate (Dreamer 4);
       - limiting the size of each step (trust regions; TR-DPO);
       - safety on real hardware (residuals, corrections).
     - **Proposal, agreeing with Codex and withdrawing my moving copy.** A slowly moving copy of the policy can follow a
       series of individually small, mistaken dream updates. It would carry dream errors into the anchor. So:
       - the reference starts as the frozen imitation policy, at Dreamer 4's β 0.3;
       - its influence fades as the agent shows real competence (VPT's decay, tied to real performance);
       - the reference is replaced by the current policy only when that policy has earned trust in **real**
         experience (Codex), which is TR-DPO's replacement gated by reality instead of a step count;
       - the reference keeps its own copy of the agent-token weights and head. Its logits are checked on fixed
         histories, because the world it reads keeps learning (Codex);
       - the corpus stays in the replay, as the data anchor robots use.
   - **Real experience reaches the motor choice directly as well (the user: "both").** The agent replays its own
     successful decisions, with a value baseline. Self-imitation learning: −log π(a|s) (R − V(s))+, imitating past
     actions "only when such decisions resulted" in a return above the value (Oh et al. 2018). So routine, mistaken and
     lucky actions inside a success aren't copied (Codex). RECAP is the robotics precedent. This gives reality a
     direct route to correct dream errors. Codex's return-order audit matters here: P1.5's single dreams order DO
     against NOOP wrongly 19.9% of the time at 15 steps, and PMPO uses only the sign. In the brain, reverse replay
     rates track reward (Ambrose, Pfeiffer & Foster 2016). Demonstrations and corrections go to plain imitation.
   - **Imitation targets** are masked per target, not per window: only 35.7% of eight-action windows are all expert
     (Codex, TRAIN census).
3. **The value:** dreamed and real-replay λ-returns, through its own weights, with DreamerV3's slow regulariser and a
   held-out check against real continuations.
   - **Dreamer 4 doesn't do this.** It trains the value only on imagined λ-returns, offline, because its dataset's
     returns belong to the demonstrators. The real-replay value is DreamerV3's (`repval` 0.3). In a life the agent's
     own real returns exist, and short dreams put more of the return on the value.
4. **The look (revised 11-23; the chooser retires entirely, formula included).**
   - **Measured** (`artifacts/experiments/20261123_eye_drives/RESULTS.md`):
     - the old chooser's raw surprise finds creatures by day (lift 4.32) and halves at night (2.04), when the static
       draws it;
     - the world's own foveal surprise, the TSP likelihood under the content-blind faithful decoder, can't tell night
       creatures from night static (lift 1.20 against 1.21);
     - contingency is broad (lift 1.2-1.5);
     - night is where the fovea's one known gain lies.
   - **Neuroscience:**
     - gaze follows a priority map combining salience and relevance (Fecteau & Munoz 2006);
     - Bayesian surprise, the belief change, attracts gaze, "irrespectively of how rare" an observation is (Itti &
       Baldi 2009);
     - the basal ganglia orient the eyes to reward (Hikosaka, Nakamura & Nakahara 2006);
     - parietal neurons encode a saccade's expected information gain, independent of reward (Foley et al. 2017);
     - fixations are chosen to reduce uncertainty about task-relevant variables (Sprague & Ballard 2003).
   - **ML precedents:**
     - glimpses learnt from task reward (Mnih et al. 2014);
     - SUGARL needed an intrinsic sensorimotor reward for its gaze policy (Shang & Ryoo 2023).
   - **Proposed: two drives, both from the system's own signals, with no imitation of a formula and no counterfactual
     supervision.**
     1. **Relevance:** the look shares the step's advantage. That means PMPO in dreams once look schedules can't change
        later outcomes, and real advantages, including self-imitation, in life. This is the drive that can find night
        zombies, because they change returns.
     2. **Information:** an intrinsic reward for the look, on **real** glimpses only: the world's foveal surprise at
        the chosen window, the TSP likelihood. This is what the look revealed beyond what the coarse view and belief
        predicted.
        - By day it picks creatures like the old chooser (argmax 0.548 against 0.525). At night it is diluted
          (measured).
        - In dreams it would be the world's surprise at itself, not information, so it isn't used there.
        - Its jobs are a dense early signal for gaze (SUGARL's lesson) and the world's curriculum. The curriculum
          value is unproven: P1-F2 was mixed.
     - **While watching** there is no task advantage. The look learns from drive 2, plus possibly the information its
       chosen glimpse adds to the demonstrated action: likelihood with against without the glimpse, for the chosen
       window only. That is realisable by any eye.
     - **A 0.2 / 49 uniform floor** for exploration.
   - **Dropped:**
     - **counterfactual decision value over all 49 windows.** It needs privileged full frames, which a real eye can't
       see, plus many passes and an oracle-style objective. It stays an evaluation reading;
     - **the NOOP forecast for contingency.**
   - **Trigger:** if the eye still misses night creatures in the run, look at content-aware precision for drive 2.
     The 10-31 pitfall applies: learned content-dependent variance made the pixel decoder digit-blind. Learning
     progress is the other candidate.
   - **Open:** the weighting of the two drives, in nats as Dreamer 4 balances its terms.
5. **Order (the user, 11-23):** watch, then dream and live interleaved, on the fly, with dreaming possible before the
   first real step.
   - Watching comes first, because living is only worth its cost once the world and the imitation are competent.
   - After that, dreaming and living alternate or run in parallel, as DayDreamer's learner and actor threads do.
   - Dreaming before the first real step is what Dreamer 4 shows works offline.
   - The schedule is decision 4's and can change on the fly.
6. **The brain parallel, stated carefully.** The actor-critic model of the basal ganglia is a well-supported
   hypothesis: dopamine firing resembles the critic's temporal-difference error (Schultz, Dayan & Montague 1997;
   reviewed by Joel, Niv & Ruppin 2002), and its anatomical details are debated. Our parts take those computational
   roles:
   - the world predicts, as cortex does;
   - the value is the critic and the choice the actor;
   - the advantage plays dopamine's role;
   - replay of successes resembles reward-modulated reverse replay;
   - the eye combines salience and reward-learnt relevance.

   It is an analogy of roles, not an anatomical identity.

## Parked: thinking when it's worth it (user 11-23, explore after the first proof)

**The idea.** At decision time the agent may imagine before acting, using the denoiser on candidate futures, but only
when the decision is worth the compute. Easy situations get a habit; hard or new ones get thought. This is the same
question as an LLM that thinks longer on hard problems. Looking is the same economics: a glimpse is information bought
at a cost. So one rule could serve the eye and deliberation alike.

**What is known (read at the source unless noted):**
- **The value of computation.** A computation is worth running while its expected improvement of the decision
  exceeds its cost: "continue the available computation with the highest value until all remaining computations have
  negative value" (Russell & Wefald 1991, via a summary; not read in full). People plan in a way broadly consistent
  with resource-rational planning, though not with any single heuristic (Callaway et al. 2022).
- **Learnt control of imagination.**
  - Hamrick et al. 2017: a model-free metacontroller decides how many imagined simulations to run and which model to
    consult. It adapts its computation to difficulty, weighs each model's reliability against its cost, and reaches a
    lower total cost than fixed policies.
  - Thinker (Chung, Anokhin & Krueger 2023) wraps the environment with the learnt world model and adds planning
    actions, so the agent learns to plan itself; state of the art on Sokoban.
- **Learnt halting.**
  - ACT (Graves 2016) learns a halting probability; PonderNet's authors call it "notably unstable".
  - PonderNet (Banino et al. 2021) learns the number of steps end to end, balancing accuracy, cost and
    generalisation.
  - HRM (Wang et al. 2025) halts by Q-learning; the benefit of its halting is contested.
  - TRM (Jolicoeur-Martineau 2025) is a two-layer, 7M-parameter recursive network.
- **LLMs.**
  - Recurrent-depth latent reasoning iterates a recurrent block to arbitrary depth at test time, with per-token
    adaptive compute (Geiping et al. 2025).
  - The failure mode is not knowing when to stop: o1-like models spent 1,953% more tokens on "2 + 3" with little
    gain, and overthinking was more frequent on simple problems (Chen et al. 2024).
- **Brain.**
  - Rats deliberate at choice points under uncertainty, with hippocampal sweeps along the options that vanish as
    behaviour automates (Redish 2016).
  - Control passes between costly goal-directed and cheap habitual systems by their uncertainty (Daw, Niv & Dayan
    2005), or by trading decision speed against accuracy (Keramati, Dezfouli & Piray 2011).

**Saccade's compute knobs, already there:**
- whether the agent tokens read the denoiser's activations at all (Codex's efficiency question);
- how far ahead, and how many candidate actions, to imagine;
- how many denoising steps per imagined frame. Shortcut forcing trains the denoiser for any power-of-two step count up
  to `k_max` (8; K = 4 today), so the sampler is already an iterative refinement whose depth can vary, which is the
  recursive-architecture link;
- where to look, and whether to look.

**The open decision when it resumes:**
- how the agent estimates whether thinking will change its decision: learnt halting, a metacontroller, or
  uncertainty-based arbitration;
- what it pays.

The first proof stays offline practice (Codex). Deliberation is priced once imagined action comparisons show decision
value.

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
  - The changed rule: water gives +2 drink. In 150 replayed DEV episodes, the frozen P1.5 world draws +2 in 0.08% of draws
    at the first drink and 0.2-1.8% after the new rule has been seen in the same episode.
  - Even after 8 or more observations it is 0.7%, and the trained +1 is still drawn 70% of the time.
  - We have memory (record, belief, context), not adaptation: nothing ever asked the world to infer rules from
    context.
- **The slow channel is untested:** how many updates, and how much replay, it takes to absorb a new rule without
  losing old ones.
- **The two channels depend on each other.** Corrected 11-22 (Codex): unlabelled mixed experience doesn't teach an
  average. A generative model trained on +1 and +2 drinks with no way to tell them apart learns a calibrated
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
| Revision: the rule changed for good | Water now gives +2, everywhere | Update the old rule. Animals keep the old one, tied to its context (renewal); networks overwrite by default | Untested. A fine-tune on the changed rule alone measures speed and interference |
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

**Codex's protocol (11-22):** the trained rule -> a changed rule -> an unannounced return to the trained rule -> a
held-out related variant, all inside
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

## The decision map (11-22)

**To close the loop (the first agent that lives, dreams and improves):**
1. **The choice faculty.** Agreed: agent tokens inside the world, look then act in a split pass, with build checks.
2. **Dreaming.**
   - Agreed: the step commits to the outcome before the look.
   - Decided: the glimpse reads nothing new, with no event code (11-23).
   - Proposed together: starts, horizon, returns and value (precedent defaults). The world learns from real
     experience, with P2 a possible later stage. The eye is targeted in dreams. Action differences are read at steps 1,
     4 and 15 inside the run (Codex).
   - Isolation, trust and fresh starts are recorded.
   - Practice first; deliberation is parked (its own section).
3. **Learning, awake and dreaming** (Decision 3, opened 11-23; proposal above, undecided):
   - the world: P1 losses on real data;
   - the motor choice: imitation, then PMPO with a frozen imitation reference whose influence fades with real
     competence and which is replaced only by a policy that earned trust in reality; plus self-imitation of its own
     real successes;
   - the value: dreams and real replay;
   - the look: reward-learnt relevance plus the world's foveal surprise on real glimpses. The chooser retires.
4. **The life cycle:**
   - what runs awake and what asleep;
   - replay contents and mix (the corpus plus its own life);
   - the world learning online without the learning rate annealing away (plasticity, optimizer continuity);
   - refreshing stored states;
   - the 6 GB budget.

**Designed in from the start, built on a measured trigger:**

5. **Knowing what it doesn't know:** uncertainty that sets how far to trust a dream and where to look or go to learn
   (curiosity), and change detection.
6. **Memory beyond the episode:** the episodic store (one-shot, survives resets, keeps dreams out), and what survives
   Craftax's episode resets.
7. **The fast channel:** how adapting in context is learnt (variation in training, scoped above).

**Framing and judging:**

8. **Task specification and retasking:** what happened versus how a goal scores it; how new tasks enter
   (demonstrations, corrections, rewards).
9. **How goals shape learning** (on hold).
10. **The lifetime judge,** and what counts as a new situation.
11. **Acuity.**

**A separate track:** a good enough world. The day coin and the self-feeding fade are world quality, not agent design.
