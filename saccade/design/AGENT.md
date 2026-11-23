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

   The learning rate, the optimizer and plasticity are proposed in Decision 4. Surprise-driven learning comes later.
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
- **Measured 11-23, with a proposed revision of what the look reads (Codex; the user decides).**
  - An eye that reads the belief chooses a chunk's looks one frame at a time (Decision 3).
  - Projected over P1's 25.6 M frames with CUDA graphs (`20261123_step_costs/RESULTS.md`):
    - a look riding the denoiser's physical half costs 30.5 ms per step for 64 streams: about 3.4 h, or 18.2 h for
      the watching run;
    - a light look reading the belief, the current periphery and lighting, and the record read costs 2.8 ms at 32
      streams: 0.6 h, or 15.5 h.
  - **In life the gap mostly closes.** The act tokens run the denoiser's pass every awake step anyway, so a look
    riding its physical half costs little there. The 2.8 h is a watching cost.
  - **What the denoiser adds before the crop is unmeasured.** The light look reads what the world's own crop predictor
    (TSP) reads to predict a crop: the predicted belief, the periphery and the lighting. The evidence for where a crop
    will be informative is within its reach.
  - **Withdrawn, the full eye kept (the user, 11-23: "unless we can prove equivalence; I'd rather we find a way to
    optimize the architecture as a whole"; Codex agrees).**
    - The denoiser also uses the previous action, the spatial record and temporal attention over recent context.
      Whether that improves gaze is open, and cutting it would take evidence away from the agent.
  - **The route instead: make the whole system faster (Codex, measured on real P1.5 updates).**
    - A fused record kernel cut the scan from 41.5 ms to 1.2 ms (state within 4.2e-7). With temporal FlexAttention,
      updates fell from 2.237 s to 2.068 s at 64 frames and from 8.438 s to 8.242 s at 256. Peak reserved memory fell
      from 3.93 to 3.19 GiB.
    - That's about 45 minutes over the run, so the full-eye watching run projects to about 17.5 h.
    - **Next:**
      - the eye and the crop share one physical pass (the eye reads its features, and the crop reuses its keys and
        values);
      - a physical-only kernel for the sampler's four steps, which today compute glimpse tokens on a placeholder;
      - a whole-update profile before any width or attention change. Dreamer 4's grouped-query attention would be a
        world change.
    - Both pass changes preserve the computation in real arithmetic and need a bf16 check on real sequences.
    - A checkpoint at eight passes (about 14 h projected) is judged by the native world and decision checks before
      any stop.
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

**Is decision 2 general? (The user asked 11-23: could it go to robots?)**
- **The design carries over.** All of these depend only on a foveated sensor, a reward and a termination:
  - the outcome committed before the look;
  - real starts as coherent snapshots;
  - the defaults (DreamerV3's, which DayDreamer ran on four real robots with H = 16);
  - the targeted eye in dreams;
  - the world learning from real experience;
  - isolation, trust and fresh starts;
  - a bounded awake step with dreaming in idle time.
- **"The glimpse reads nothing new" is a rule with a trigger, not a law.** It holds because Craftax's coarse view
  carries the outcome (the health digit, exactly). A robot whose outcomes show only below peripheral acuity, such as a
  crack or a liquid level, could fire the trigger.
- **Passive sensing only (Codex).** A digital crop reveals a completed event. A physical camera move that consumes time,
  changes the body's pose or affects contact belongs in the action and transition model. Look invariance is then
  tested at matched physical actions and elapsed time, and the horizon is chosen in units of the robot's control rate.
- **What is Craftax-specific lives in perception and action, not in dreaming:**
  - the fixed 9 x 9 cell grid with 49 discrete windows, which a robot replaces with continuous or finer gaze;
  - the screen-fixed per-cell record, which a moving camera would need to register or replace with a world- or
    object-centred memory;
  - 17 discrete actions. PMPO and MTP work for continuous actions too (MPO's family is continuous control);
  - episodes with resets, against one continuous life.

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

**Agreed 11-23 (the user: "your drive and real experience splits seem correct"), with Codex's corrections applied, part
by part, in one system:**
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
     - **DECIDED (the user, 11-23: "I'll let you make the call"). The anchor in practice:**
       - **What it does:**
         - a dream update minimises PMPO + β KL[π_θ ‖ π_ref] over dreamed states;
         - the advantage-maximising solution of that trade-off is π*(a|s) ∝ π_ref(a|s) exp(A(s,a) / β);
         - so the policy can move far within behaviour the reference takes, but barely toward actions the reference
           almost never takes, however good a dream says they are. That caps dream exploitation;
         - PMPO's sign-based loss is not exactly this closed form, but it strikes the same balance.
       - **Worked example (softened 11-23, Codex; arithmetic verified):**
         - with π_ref [DO 0.70, NOOP 0.20, sleep 0.10], dreamed advantages [-0.2, +0.3, 0] and β 0.3, the target is
           about [0.36, 0.54, 0.10];
         - the KL is a soft preference with no fixed ceiling, and what moves a rare action is its advantage *relative*
           to the others. An action at 0.1% in the reference with advantage +0.5 rises to 0.53% when the others have
           advantage 0, and to 2.73% when they have -0.5;
         - PMPO averages signed log-probability terms separately over positive and negative samples, so its update
           differs from this closed form. β 0.3 stays the starting coefficient, and policy drift from the reference is
           a reading in the run.
       - **The life cycle of the reference:**
         1. When imitation is competent, π_ref is a frozen snapshot of the agent-token weights and the motor head. Its
            action distributions on a fixed set of real histories are recorded.
         2. Dreams train against it at β 0.3 (Dreamer 4), held constant.
         3. Every N real episodes, the current policy is compared with the reference on real return and on a
            retained-skill panel (each achievement's rate).
         4. If it is better on return and loses no skill beyond a tolerance, it becomes the new reference, which is
            snapshotted and re-recorded. This is TR-DPO's replacement gated by reality instead of a step count
            (Codex), and VPT's fading achieved by replacement rather than by decaying β.
         5. The reference's distributions are re-checked on fixed histories after world updates, since it reads a
            learning world (Codex: "the thing to freeze is the reference's behaviour on a fixed history"). On drift
            beyond a tolerance, its copy is distilled back to its recorded outputs. Distillation and the check use
            separate history sets (Codex, 11-23): otherwise a passing check only shows the distillation fitted those
            examples. The check set includes histories where its retained skills matter.
       - **Why this form:**
         - Dreamer 4 for dreams; VPT for skills (the retained-skill panel); robots for verification in reality;
           TR-DPO for references that move;
         - a moving average would follow small, mistaken dream updates, so my moving copy is withdrawn;
         - the corpus stays in the replay, as the data anchor robots use;
         - the look has no imitation reference, so its reference is its own policy at the same snapshots.
   - **Real experience reaches the motor choice directly as well (the user: "both"), with signed advantages.**
     - Recent real experience trains the policy by PMPO on real advantages, both signs: what went better than expected
       is reinforced, what went worse is discouraged.
     - Codex's audit (verified) shows why positive-only self-imitation isn't enough. With an independently estimated
       baseline, the lower-return action still earns positive credit on 14.45% of its outcomes at 15 steps (3.50% of
       the weight). The original self-imitation method itself runs beside an on-policy actor-critic (Oh et al. 2018,
       A2C + SIL).
     - Self-imitation of older successes, −log π (R − V)+, stays an option for retaining rare skills, on a measured
       trigger.
     - In the brain, dopamine signals both better- and worse-than-expected outcomes (Schultz, Dayan & Montague 1997),
       and reverse replay rates track reward (Ambrose, Pfeiffer & Foster 2016).
     - This gives reality a direct route to correct dream errors. P1.5's single dreams order DO against NOOP wrongly
       19.9% of the time at 15 steps, and PMPO acts on the sign.
     - Demonstrations and corrections go to plain imitation.
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
     1. **Relevance:** the look shares the future task return, with a **pre-look baseline** (Codex).
        - A value read by the look tokens before the crop, against the motor's value after it.
        - If the look's baseline had already seen the crop, it would subtract the very gain the look should get
          credit for.
        - **The grounded signal is the future task return minus V_pre** (Codex, 11-23).
          - V_post - V_pre is a cheaper estimate of the same advantage when both values predict the same return under
            the same downstream policy. It is Sprague & Ballard's quantity, learnt by the critic.
          - A single difference can be negative because a good look revealed bad news. So it is only a shortcut,
            trusted as far as it agrees with real returns across repeated looks.
        - In dreams this is PMPO once look schedules can't change later outcomes. In life it is real advantages.
        - The reward of the previous motor action belongs to that action.
        - This is the drive that can find night zombies, because they change returns.
     2. **Information: belief change from the crop actually acquired, on real glimpses only**
        (`artifacts/experiments/20261123_look_information/RESULTS.md`).
        - It is ||b_t(real crop) - b_t(the same window at periphery acuity)||: how far what the look showed moved the
          belief beyond what the periphery had already given. That is Itti & Baldi's surprise as belief change.
        - Measured on P1.5: creatures move the belief 2.4x more than entity-free cells by day, and 2.2x at night.
          Pixel surprise couldn't separate them at night (lift 1.20 against 1.21), and it flips with sensor noise at a
          fixed scene (Codex's static control: the best window changes on 26-31 of 48 dark repeat pairs, verified).
        - **Static still moves it, less.** Codex's static control for belief change: the best window switches on 11 of
          48 dark repeat pairs, against 26 for pixel surprise under the same fixed periphery (0 of 48 at night onset
          for both).
        - **It doesn't repeat the 11-20 failure.** The switching dissection's "belief-change eye" scored a different
          quantity: how far seeing the real frame t shifted the world's zero-signal prediction of frame t + 1. It
          failed, drawn to newly entering terrain (43% of its score) and night static (49%), with mobs at 10%,
          because a mob's next position is hedged. Today's measure, read on those failure classes (third reading,
          `20261123_look_information/RESULTS.md`):
          - entering terrain scores below static (0.365 / 0.227 against 0.408 / 0.340, day / night);
          - creatures score highest by day (0.936), and changed HUD highest at night (1.003, against 0.708 for
            unchanged HUD);
          - unchanged HUD still scores 1.6-2x static, although the coarse view holds its digits. Belief change partly
            rewards structured detail, not only new information.
        - It costs one extra belief step, and any eye can compute it: the periphery-acuity crop comes from the
          periphery.
        - **The principled target is predictive gain:** whether the crop improves the prediction of subsequent real
          evidence (Codex). Sensor noise can't earn it. It read zero on P1.5 for every category.
          - **Softened (Codex, 11-23):** that test gave the predictor the actual next periphery. A zero gain can mean
            redundancy (the next frame already shows it), weak retention, or a readout that barely uses the carried
            belief.
          - A forecast scored before the next frame is revealed would separate these. P1.5's trans-saccadic head has
            no forecast mode (`heads.forecast.tsp` is false), so that needs its own generation pass.
          - Predictive gain becomes the drive's definition once memory demonstrably carries glimpse content forward.
            Until then it is a reading, beside raw foveal likelihood as a diagnostic.
        - In dreams it would be the world's surprise at itself, so it isn't used there.
     - **While watching** there is no task advantage. The look learns from drive 2, plus the information its chosen
       glimpse adds to the demonstrated action: likelihood with against without it, for the chosen window only.
     - **A 0.2 / 49 uniform floor** for exploration.
   - **Contingency retired as a term** (the user asked). Its job was to look where one's actions have effects. That is
     covered:
     - by relevance, for effects that change returns;
     - by information, for effects the agent doesn't yet expect. Once a behaviour's effects are understood, their
       evidence stops moving the belief and the eye moves on, which is the user's point that contingency evolves from
       surprise.

     A measured trigger: if the eye under-samples the faced cell after DO in the run.
   - **Dropped:**
     - **counterfactual decision value over all 49 windows.** It needs privileged full frames, which a real eye can't
       see, plus many passes and an oracle-style objective. It stays an evaluation reading;
     - **the NOOP forecast.**
   - **Weighting (Codex, accepted, revised 11-23):**
     - "nats" don't fix the information drive's scale: averaging a crop's pixel losses or summing them changes it
       arbitrarily;
     - **the information drive is a bounded auxiliary reward, not a sign.** A sign removes the units but keeps the
       risk of reinforcing small, noisy differences: a crop that beats the baseline by a hair gets the full push, and
       static still moves the belief (11 of 48 repeat pairs);
     - so belief change, minus a running baseline, is divided by its running 5th-95th percentile range and clipped.
       The scale is the one DreamerV3 uses for returns (without its floor of 1, which assumes reward units).
       Magnitude survives, so a creature's 2.4x counts for more than a hair's breadth;
     - task advantages keep PMPO's sign form (Dreamer 4) and task-reward units;
     - the two terms balance by one dimensionless weight, judged by what it does to gaze and to real decisions at the
       same sensing budget.
   - **During world learning (the user asked).** There is no separate P1 phase with a different eye in the full
     system. The look is the agent's from the first watched frame:
     - early on, both drives are uninformative, so the look is near-uniform by itself (the floor plus untrained
       drives), which gives the world coverage when it knows least;
     - as the world and belief learn, belief change sharpens the look onto what informs, and the world's fine-detail
       learning (the trans-saccadic and fovea-reconstruction losses) follows the look. This is the curriculum, and the
       11-22 agreement that the eye's attention decides what the world learns in detail;
     - the feedback loop that made P1-F2 mixed (the eye shapes the world's data, the world's state shapes the eye) is
       bounded by the floor, and read in the run (coverage by category, day and night);
     - the alternative, a fixed uniform eye during watching, is not the agent.
   - **What this means for P1: pretraining the world and the eye, then watching adds the motor choice (revised
     11-23, the user's proposal).**
     - **My first answer bundled everything into one run from the first frame. Its only real argument was the eye,
       and the eye doesn't need it.**
       - The act tokens read the world one way, with gradients stopped (Decision 1), so they never change it. Training
         them from the first frame buys the world nothing and spends compute on features that are still moving.
       - The look does change the world, because it picks which crops train the trans-saccadic and fovea losses. But
         its information drive needs no actions, rewards or task: only the belief with and without the fine crop. So
         the eye can learn in pretraining, which is the user's point.
     - **The general schedule, the modern pipeline:**
       1. **Pretraining, the foundation:** perception, record, belief, denoiser, the outcome heads where rewards
          exist, and the look by its information drive.
          - Data: anything that shows the world. On Classic that is the corpus. For a robot it is video, with actions
            where they exist and a learned "no action" embedding otherwise. Dreamer 4 does this and needs actions for
            little of its 2,541 hours: with 100 hours of actions it reports 85% PSNR and 100% SSIM "compared to a
            model trained with all actions". Latent actions are the other route (Genie, LAPO, DreamDojo).
          - Saccade's belief update takes an action, a reward and a termination, so unlabelled data needs that
            missing-data contract (Codex). Classic needs none.
       2. **Watching, the body and the task:** the act tokens and imitation join, the look gains the demonstrated-
          action term, and the world's losses continue.
          - This is Dreamer 4's phase 2 ("finetune world model with task inputs"), V-JEPA 2-AC's post-training on 62 h
            of robot video, DreamDojo's target-robot post-training and Cosmos's post-training.
          - Dreamer 4's imitation policy initialised from the pretrained world beats imitation without it.
       3. **Life:** living and dreaming (Decision 4).
       - On Classic, both stages read the same corpus, as Dreamer 4's phases read the same dataset.
     - **Per pretraining update**, as P1.5 has today:
       1. a no-gradient pass in which the look tokens choose each frame's window (P1.5's chooser does this in
          `Controller.prepare`);
       2. the training pass on those crops, with the look's information drive: one extra belief step with the
          periphery-acuity crop.
     - **That look pass is sequential across the chunk** (Codex). The eye reads the belief, and the belief holds the
       last glimpse, so the look at frame t waits for the crop at t - 1. The chooser could batch the chunk because it
       read only coarse history.
       - Measured on frozen P1.5 (`20261123_step_costs/RESULTS.md`): the streaming calls are latency-bound, about 37
         ms per frame at 1 or 16 streams.
       - A 256-frame look pass costs +122% of P1.5's update at 16 streams, or +44% when it runs 64 streams at once
         (four updates' looks, the look policy lagging by up to three updates, with its probabilities logged).
       - CUDA graphs or compilation could cut the launch overhead; unmeasured.
       - This is the price of an eye with memory, whenever it learns. Every active-vision learner generates its
         glimpse sequence step by step (Mnih et al. 2014; Grimes et al. 2023).
       - A batchable alternative exists: an eye reading only what doesn't depend on its earlier crops (the coarse
         view, the record, the belief at the chunk's start). It is not the eye that lives, so it stays a fallback if
         the cost is prohibitive.
     - **Early on** the look tokens are untrained, so the looks are near-uniform by themselves (with the 0.2 / 49
       floor). The uniform start happens only while the eye knows nothing.
     - **It mirrors infant development** (an analogy, with verified sources):
       - early eye movements are mostly reflexive and subcortical, and voluntary saccades mature later (Johnson 1990;
         the timing is debated);
       - at 7-8 months, infants look away from sequences an ideal learner finds too predictable or too unpredictable
         (Kidd, Piantadosi & Aslin 2012);
       - saccadic latency, looking time and engagement follow surprise, predictability and learning progress (Poli et
         al. 2020). That is an information drive before any task;
       - goal-directed gaze comes with action: 12-month-olds look ahead to an observed action's goal and 6-month-olds
         don't (Falck-Ytter et al. 2006). 3-month-olds given reaching experience start attending to others' goals
         (Sommerville, Woodward & Needham 2005);
       - machine precedent: a camera policy rewarded only by its own predictor's error developed a "behavioral fovea"
         (Grimes et al. 2023).

       So: information first (pretraining), others' goals while watching, its own relevance in life. Newborns also have
       innate salience. Ours starts uniform: emergent rather than built in.
     - **What goes away:** the chooser, its surprise and contingency head, and its separate network.
     - It is a new world run: P1.5 can't be reused, because the eye, the pre-look outcome heads and the learning rate
       schedule (Decision 4) all change.
   - **Does the glimpse have a privileged route? (the user asked, 11-23; raised on 11-15)** It has three routes, and
     two are privileged:
     - **What fine detail is learned: privileged by construction.**
       - Only the looked-at crop carries full-resolution pixels, so the crop losses (trans-saccadic prediction,
         fovea reconstruction) learn fine detail only where the eye looks.
       - That is attention gating learning (Turk-Browne et al. 2005), the 11-15 idea, built in.
     - **The current decision: direct.** The act tokens read that frame's glimpse tokens after the crop (Decision 1).
     - **Memory and the world's dynamics: not privileged.**
       - The denoiser's physical tokens never read glimpse tokens.
       - The belief takes the glimpse through one 256-number projection that keeps little of it (F1.6; predictive
         gain about zero).
       - P1.6's foveal record was barely read.
     - **Why that hasn't mattered:**
       - In Classic at 3x3 the periphery shows nearly everything again next frame, so there is little for memory to
         keep.
       - Where acuity is lower (robots, full Craftax), what a look saw has to be remembered, because looking again
         costs.
     - **In the brain:** memory across saccades is a limited-capacity, long-lasting store holding little detail,
       apparently visual working memory itself, and attention decides what enters it (Irwin 1991; Irwin & Gordon,
       abstracts).
     - **So no new route now.** The memory route belongs to Decision 6. The location-keyed episodic store agreed on
       10-23 is that privileged route, and predictive gain is its judge. It is built when acuity makes it needed.
   - **Trigger:** if the eye still misses night creatures in the run, look at content-aware precision or predictive
     gain. The 10-31 pitfall applies: learned content-dependent variance made the pixel decoder digit-blind.
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

### Decision 4: the life cycle (opened 11-23; agreed 11-23, numbers provisional)

**Status (the user, 11-23: "agreed for all of these").**
- The budget, the segment length and the dream length stay open until measured. They carry provisional values
  from precedent beside the other hyperparameters.
- Each part is confirmed once its evidence is in:
  - **confirmed by precedent and measurement:**
    - one learner with a frozen awake copy and a rebuild at adoption. Codex timed the rebuild on P1.5 at 42 / 167 /
      337 ms for 256 / 1,024 / 2,048 frames, against about 9.2 s of awake GPU time per 256-step segment at one
      stream;
    - uniform replay with an online queue;
    - a constant learning rate, one optimizer for the life, plasticity tools on a trigger;
    - episodes inside one life;
    - the checks;
  - **provisional:**
    - S = 256 real steps;
    - the dream horizon of 15;
    - dream starts per update: one per replayed context, as in Dreamer 4. Starting from every frame, as DreamerV3
      does, measured at about 210 h per million real steps at replay ratio 16 (`20261123_step_costs`);
    - the replay ratio;
    - where watching starts;
    - the life's length: 100 k real steps as the first substantial reading, then decide how far toward 1 M (Codex);
    - how many Craftax environments step together. One stream costs about 10 h of awake GPU per million real steps
      and 16 about 0.6 h, because the calls are latency-bound. Several environments are several episodes of the
      same agent, like a fleet.
- **Codex's additions, adopted:**
  - log which weights generated each real segment, with the look and motor probabilities used, for any off-policy
    correction later;
  - on a continuous robot, a bounded rebuild (the last N frames, or in the background, checked before adoption). On
    P1.5, a rebuild from the last 256 frames moved the positive-reward forecast by up to 1.83 points at frame 2,048;
  - the live fraction (Codex: 6.6% at 10 k real steps and 9.8% at 100 k at ratio 16) is logged separately for world
    training, real advantages and dream starts;
  - P1's peak learning rate is a starting point to verify.

**What it settles:** which weights act and which dream, when each faculty learns and from what data, at what pace,
on one 6 GB GPU. The user framed it on 11-22 as "largely just figuring out training schedules". It has to honour
decisions 1-3:
- one run from the first watched frame;
- the frozen reference, promoted by reality;
- signed real advantages;
- dreams from coherent snapshots;
- a world that never learns from its own dreams.

**Precedents, read at the source 11-23:**
- **DreamerV3 code** (`third_party/sources/danijar__dreamerv3`):
  - one agent object; `run/parallel.py` runs an actor thread and a learner thread on it, so acting always uses the
    newest weights and carries its latent across updates;
  - one update every `batch_size × batch_length / train_ratio` real steps. For Crafter that is 1,024 / 512, so an update
    every 2 steps and each real frame replayed about 512 times, over 1.1 M steps;
  - replay of 5 M steps (more than the run), sampled uniformly, plus an online queue: each newly completed sequence is
    trained on next (`core/replay.py`, `online: True`);
  - `replay_context`: each step's latent is stored at acting time, starts its training sequence, and is overwritten
    with the current model's latent after every update;
  - learning rate 4e-5, constant after 1,000 warm-up updates, never annealed; no weight decay.
- **DayDreamer** (PDF):
  - a learner thread and an actor thread;
  - "there is no training frequency hyperparameter because the decoupled learner optimizes the neural networks in
    parallel with data collection, without rate limiting";
  - FIFO replay of 10^6, learning from step 10^4, learning rate 1e-4.
- **HIL-SERL code** (`examples/train_rlpd.py`):
  - the learner publishes weights every `steps_per_update` updates, and the actor receives them by callback;
  - "50/50 sampling from RLPD, half from demo and half from online experience";
  - human interventions are kept in their own store and used as demonstrations.
- **RLPD:** symmetric 50/50 offline/online batches, "not very sensitive" to the proportion.
- **CLEAR** (PDF):
  - a 50-50 mix of new and replayed experience ("performance does not appear to be very sensitive to this ratio");
    75-25 already removes most forgetting, and all-replay costs performance;
  - behavioural cloning of its own past outputs on replay (a KL on the policy, L2 on the value);
  - reservoir sampling when memory is bounded: storing 1 in 200 experiences changed little.
- **Isele & Cosgun** (PDF):
  - long-term memory by distribution matching, which is reservoir sampling, "consistently the best approach on all
    domains tested";
  - kept with "a small FIFO buffer that constantly refreshes to ensure that the network has the opportunity to train
    on all incoming experiences", which they relate to complementary learning systems.
- **Kessler et al. 2023**, a DreamerV2 world model learning continually:
  - reservoir sampling "robustly mitigates forgetting";
  - 50:50 recent/old "causes inconsistent effects";
  - a larger buffer "prevents forgetting" but makes it "harder to learn new tasks".
- **R2D2** (the paper wasn't readable; its slides were): recurrent states stored from older networks suffer
  "representational drift", a zero start state is much worse, and burn-in helps the start of replayed sequences.
- **MuZero Unplugged:**
  - Reanalyse recomputes targets on stored data with the latest network;
  - "the gain from bootstrapping with the actually observed environment rewards seems to outweigh any harm" from
    being off-policy.
- **IMPALA:** the learner runs ahead of its actors ("policy lag"), corrected by V-trace.
- **Our staleness audit** (`20261119_switching/STALENESS.md`):
  - between P1.5 checkpoints the drift was the jointly trained encoder's;
  - cached states were 1.7-13x the fresh error early, and 1.03-1.13x late, after the cosine schedule had lowered the
    learning rate.
- **Plasticity:**
  - continually trained networks lose the ability to learn; continual backprop reinitialises a few little-used units
    (Dohare et al. 2024);
  - the primacy bias; reset the last layers and keep the replay (Nikishin et al. 2022);
  - resets let the replay ratio scale (D'Oro et al. 2023);
  - in model-based RL, resetting the agent hurts and resetting the world model helps (Qiao et al. 2023);
  - layer norm plus weight decay maintain plasticity (Lyle et al. 2024);
  - shrink and perturb (Ash & Adams 2020);
  - resetting Adam's moments helps at hard target-network updates, little with soft ones (Asadi et al. 2023).
- **Schedules:** a constant learning rate with a short cooldown (under 20%) matches cosine, while cosine needs the
  length in advance (Hägele et al. 2024).
- **Brain:**
  - complementary learning systems: fast hippocampal storage, replayed interleaved into a slow neocortex (McClelland,
    McNaughton & O'Reilly 1995). Replay "allows goal-dependent weighting of experience statistics" (Kumaran, Hassabis
    & McClelland 2016);
  - awake replay "occurs most often during relative immobility" (Carr, Jadhav & Frank 2011);
  - sleep favours consolidation, and waking favours encoding and retrieval (Diekelmann & Born 2010);
  - sleep renormalises synaptic strength, restoring learning capacity (Tononi & Cirelli 2014);
  - the wake-sleep algorithm: waking trains the generative model on real input, sleep trains recognition on its
    fantasies (Hinton, Dayan, Frey & Neal 1995). That is our split: the world learns from reality, the choice from
    dreams.

**Proposed:**

1. **Snapshots: one learner; the awake agent adopts its weights at segment boundaries; dreams always run on the
   current weights.**
   - **Dreams:** there is one set of learning weights. Dreams run on them, starting from states computed within the
     same update: a start is the training stream's state at that frame, as DreamerV3 imagines from its training
     batch. Every dream is a coherent snapshot by construction.
   - **Acting:** the awake agent acts with a frozen copy for a segment of S real steps, then adopts the learner's
     weights.
     - On adoption, the current episode's state (coarse tokens, record, belief, context) is rebuilt by re-encoding
       its stored frames and looks with the new weights.
     - The reason is R2D2's: a state computed by older weights is stale. Our audit puts the drift in the encoder, and
       a constant learning rate (point 4) keeps the encoder moving, unlike P1.5's late, annealed phase.
   - **Why not act on live weights as DreamerV3 does:**
     - its latent is re-anchored by each observation, while our record and belief integrate the whole episode
       through an encoder that keeps learning;
     - a frozen copy per segment also gives each segment's data one known policy, which the signed real advantages
       need.
   - **The reference** (Decision 3) is a third copy, frozen, of the agent tokens and heads.
   - **S, working: 256 real steps,** P1's long chunk, so a segment is one replay chunk.
     - The reading is the difference between the carried and the rebuilt state's action logits at adoption (R2D2's
       discrepancy, in our units).
     - If it stays small, S can grow; if large, adoption needs to be more frequent.
   - **Policy lag:** signed real advantages use only the last few segments, so lag is bounded by construction, and the
     advantages come from the current value. V-trace (IMPALA; CLEAR on replay) is the precedent correction if the
     value's held-out check against real continuations fails.
   - **Robots:** the same contract runs with parallel threads (DreamerV3's `parallel.py`, DayDreamer, HIL-SERL's
     nodes): the learner never stops, and the actor adopts weights at its own boundaries. On Craftax with one GPU the
     segments run serially: act S steps, then the offline updates they buy.
2. **Pretraining, watching, then a life of alternating segments.**
   - **Pretraining and watching** as in Decision 3: the world and the eye first, then the act tokens join with the
     world's losses continuing.
     - Together they run for P1's frame budget, 25.6 M frames (ten passes); under P1.5's recipe the monitor loss fell
       at every checkpoint over that budget.
     - Where watching starts within it is provisional. Dreamer 4 doesn't state its phase lengths, and P1.5's monitor
       loss never flattened, so it can't be "when the world is done".
       - Working value: the last pass (10% of the budget), a placeholder with no precedent or measurement behind it.
       - It is confirmed or extended by the act tokens' held-out imitation loss flattening.
   - **At handover:**
     - the reference is snapshotted;
     - imitation of the corpus stops. The reference's KL becomes the behavioural anchor, as in Dreamer 4's phase 3,
       which has no imitation loss;
     - **trigger (Codex):** if a promotion check's retained-skill panel shows a corpus skill fading, imitation of
       expert-flagged corpus steps resumes at a small weight. Precedents: CLEAR clones its past self on replay, and
       HIL-SERL keeps demonstrations in half of every batch. A corpus kept in the world's replay can still get little
       motor rehearsal;
     - the corpus stays in replay as the world's data anchor;
     - demonstrations or corrections arriving later go to imitation (Decision 3).
   - **Life:** segments of S real steps, each followed by the offline updates it buys:
     - world updates on replay batches;
     - dreams from the same batches' states.
   - **One knob, the replay ratio r:** replayed frames per real step. Dreams scale with it, as in DreamerV3.
   - **Optional:** a dream-only period between handover and the first real step (Dreamer 4 shows offline dreaming
     works). The order can change on the fly.
3. **Replay: keep everything, sample uniformly, train on the new at once.**
   - **What's stored:** every real step's full frame, look, action, reward and termination, plus the corpus's expert
     flag. Disk is the only limit: a 63 × 63 RGB frame is 11.9 kB raw, 12 GB per million steps before compression.
   - **World batches:** uniform over all real experience, corpus included. That is DreamerV3's replay, Kessler's
     reservoir result, Isele's distribution matching and CLEAR's buffer.
     - **Plus an online queue:** each new segment is trained on in the offline phase right after it (DreamerV3's
       `online`, Isele's small FIFO).
     - In the brain, that is replay right after experience (awake replay) and interleaving with old memories
       (complementary learning systems).
     - **How a new segment enters P1's streams.** P1 trains whole episodes as streams with a carried state, and a
       segment can start mid-episode.
       - The stream that takes it starts from the state stored at acting time, as DreamerV3's `replay_context` does.
       - In the serial schedule that state was computed by the weights the learner holds when the offline phase
         begins, so it is fresh at the first update.
       - Training writes the refreshed states back.
       - Uniformly drawn episodes start from their beginning, as in P1.
   - **The corpus's share falls by itself** as life accumulates (half at 2.56 M real steps). There is no mixing
     schedule.
   - **Not 50/50** (RLPD, HIL-SERL): those are actor-critic replay mixes. For a world model, Kessler found 50:50
     recent/old inconsistent and uniform best.
   - **Dream starts follow the world's batch.**
     - Through the online queue, each new segment supplies starts in the update right after it: one stream of the 16
       in P1's batch shape.
     - If the agent's own states are still too rare among starts, RLPD's symmetric split, applied to dream starts
       only, is the precedent fix.
   - **Held out:** fixed history sets for the reference's drift check (distil on one, check on another).
4. **The world keeps learning: a constant learning rate, one optimizer, plasticity tools on a trigger.**
   - **The learning rate is constant after warm-up, for the whole life:** P1's peak 1e-4 after its 1,024,000-frame
     warm-up (DreamerV3: constant, never annealed; DayDreamer 1e-4).
     - Cosine needs the training length in advance, and a life has none.
     - Hägele et al. find a short cooldown supplies the drop cosine gets at its end. So a comparison with the
       cosine-trained P1.5 uses a cooldown branch, while the life itself never cools down.
   - **One AdamW for the whole life, never recreated** (our fine-tune rule).
     - Asadi et al.'s resets help where a hard target network changes the loss abruptly. Our world has no target
       network, so no reset.
     - The value's slow regulariser is a soft target, where their gain was small.
   - **Plasticity:**
     - layer norm and weight decay are already in (the Mamba and denoiser norms; AdamW 0.01 on linear and conv
       weights): Lyle et al.'s combination;
     - no default resets: Qiao et al. found resetting the agent hurts in model-based RL, and resetting the world
       erases the watching;
     - **the reading:** how much the online-queue update lowers the world's loss on a new segment, measured on
       arrival and after;
     - **the trigger:** that gain falls over the life while the loss on arrival stays high. Then shrink-and-perturb
       or continual backprop for the world, and a last-layers reset for the value and policy heads (Nikishin, with
       the replay kept).
     - In the brain, renormalisation in sleep plays this role (Tononi & Cirelli), as an analogy only.
5. **Episodes inside one life.**
   - Craftax resets on death:
     - the weights, replay, optimizer and reference persist;
     - the belief, record and context restart (P1's episode start).
   - What survives a reset beyond the weights is Decision 6, memory beyond the episode.
   - Segments count real steps, not episodes, so a robot without resets runs the same schedule.
6. **Checks, every 10% of the life's real-step budget** (P1's checkpoint cadence):
   - promotion: real return and each achievement's rate, the current policy against the reference, on paired
     evaluation episodes outside the learning budget (DreamerV3 keeps separate evaluation environments);
   - the reference's drift on its held-out histories;
   - dreamed against real action differences at steps 1, 4 and 15;
   - eye coverage by category, day and night;
   - plasticity (point 4);
   - forgetting: the world's loss on the corpus monitor set;
   - adoption staleness (point 1).
7. **The budget on one 6 GB GPU.**
   - **Measured:** P1 trains at about 464 frames/s (25.6 M frames in 15.3 h for P1.5; 16.6 h for P1.6).
   - **At that rate DreamerV3's Crafter schedule is out of reach:** 1.1 M real steps × 512 replays is 563 M frames,
     about 337 h. Our replay ratio will be far lower.
   - **A life of 1 M real steps**, Crafter's standard budget, costs about 0.6 h of world training per unit of r: r 16
     is about 10 h, r 32 about 19 h. That's before dreams and acting.
   - **Unmeasured, and needed before fixing r:**
     - the awake step (split pass, Mamba step, Craftax's CPU step);
     - the dream step (the four-step sampler for the coarse view, one crop, the context pass);
     - the rebuild at adoption.

     These are the first build measurement. r then follows from the time we give a life and the measured costs, not
     from tuning.
   - One GPU job at a time; replay on disk.
8. **How much to dream (the user asked, 11-23).**
   - **Dreamer 4's one rollout per context is cheaper because it dreams less.** That is 240 dreamed steps per 4,096-frame
     update, against DreamerV3's 15 from every replayed frame. Dreamer 4 gives its reason: "prioritizing data
     diversity and reducing memory consumption".
     - DreamerV3 can afford about 7,680 imagined transitions per real step on Crafter because its imagined step is a
       small latent recurrent step that renders nothing.
     - Dreamer 4's, like ours, runs a large generative transformer for every imagined frame.
   - **Measured:** a dreamed step costs at least 2.8 ms per stream here, about 1.5x training on a real frame
     (`20261123_step_costs`, second reading).
   - **Dreaming is cheap relative to reality only where reality is expensive.**
     - A dreamed step is about 3x a Craftax step (0.9 ms on one CPU core), so on Classic reality is cheaper than
       imagination.
     - That is the environment, a fast simulator, not our hardware: a faster GPU speeds dreams, but a hand-written
       simulator stays cheaper than a learned one.
     - DayDreamer's robots act at 20 Hz and 2 Hz, so the same dreamed step is 18x and 180x faster than their real
       steps, runs 256 at once, risks nothing, and can branch from any remembered moment.
     - The benchmark's real-step budget is how Classic stands in for expensive reality. We report wall-clock time
       beside it.
   - **Dreams get their own fixed share of offline time** (the user, 11-23: "do it like humans? A fixed time per awake
     schedule"). They also get their own starts, drawn from replay with the online queue, independent of the world's
     batch.
     - **My earlier reading was withdrawn,** both by the user and by Codex. It would have kept dreaming while the
       policy's dream objective improved, but that can mean the policy is getting better at exploiting the world.
     - **The brain's version is fixed by time awake.**
       - In the two-process model, sleep pressure rises during waking and falls exponentially in sleep (Borbély 1982).
       - Learning modulates it locally: a learning task raises slow-wave activity in the regions it used, and the
         rise correlates with the gain after sleep (Huber et al. 2004).
       - So the default is a fixed offline time per awake time. Modulating it by learning load is a later option.
     - **The judge** is real improvement at a matched interaction budget, with the action-difference checks at depths
       1, 4 and 15 (Codex), and the complete dream update priced, policy and value learning included.
     - **Precedent that more dreaming helps:**
       - Sutton's 1990 Dyna shows learning curves for 0, 10 and 100 planning steps per real step;
       - DreamerV3's README: "Increasing the number of gradient steps further increases data efficiency."
     - **Provisional:** dreams get half of the offline GPU time. At replay ratio 16 that is about 11 dreamed steps per
       real step.

### Decision 5: knowing what it doesn't know (opened 11-23; research and proposal, undecided)

**What it is for: three jobs the life cycle exposes.**
1. **Trusting a dream.** A world that is wrong where the policy goes teaches the policy to exploit it. On 07-06, a
   policy trained in a frozen world scored below random in real Crafter. Reality corrects only where the agent goes.
2. **Curiosity:** where to go and look to learn, beyond what the reference's KL and the eye's information drive give.
3. **Noticing change:** a rule or a place is new, so learning speeds up there and stale dreams stop counting.

On 10-23 this was agreed in principle. On 10-25 it was shelved until the complete model exists, with "keep it cheap
to add later (modular prior heads, ...)".

**Are these jobs canonical, or by-products of our Craftax work? (the user asked, 11-23)**
- **Trusting a dream is model-based RL's central problem, everywhere except in how the Dreamers handle it.**
  - In model-based RL with deep networks, "the learned policy tends to exploit regions where insufficient data is
    available for the model to be learned" (ME-TRPO, Kurutach et al. 2018). Their fix is an ensemble of models.
  - Ha & Schmidhuber's dream agent found an "adversarial policy" (read 11-22).
  - MBPO keeps model rollouts short and branched from real data; MOPO and MOReL make offline models pessimistic
    (read 11-22).
- **The Dreamers rely on structure and world quality, with no runtime uncertainty.**
  - Dreamer 1-3 learn online, so reality keeps correcting the world; imagination is 15 steps, with a critic beyond.
  - Dreamer 4 is offline, and its guards are three:
    - "the dynamics loss is applied only on the uniform sequences to avoid optimistic generations";
    - a reverse KL to the frozen imitation prior "to better constrain the policy to the space of reasonable
      behaviors";
    - a world accurate enough to be "the first world model to accurately predict the object interactions and game
      mechanics" of its block-placing tasks.
- **Predicting false dreams in a generative world model is new.**
  - Hansen & Wang (2606.27326, June 2026, a 350M Dreamer 4 world model, read at the source) write that where an
    autoregressive world model "will fail along a rollout has received comparatively little attention".
  - They find hallucination "is inherently a data coverage issue": it concentrates in low-coverage regions.
- **Curiosity from the world model's uncertainty is canonical,** including in Hafner's own line (Plan2Explore, Sekar
  et al. 2020).
- **Noticing change is not standard in the Dreamers.** It comes from continual learning and neuroscience, and from
  our own selling point (adapting to new situations), not from Craftax.
- **Our design already has most of the Dreamers' guards:**
  - online correction (life);
  - short dreams with a value beyond;
  - the reference's KL;
  - the world learning only from real data;
  - signed real advantages;
  - the dreamed-against-real action-difference checks.

  So uncertainty is not needed by default, and the readings-then-triggers form below fits.

**Three kinds of unpredictability, which prediction error mixes:**
- **Expected:** noise no learning removes, like night static. In the brain, acetylcholine signals "expected
  uncertainty, coming from known unreliability of predictive cues within a context" (Yu & Dayan 2005).
- **Epistemic:** what the world hasn't learned yet. It should drive curiosity and limit trust in dreams.
- **Unexpected:** the rules changed. Norepinephrine signals unexpected uncertainty in the same account.

Raw prediction error can't tell them apart. That is the noisy-TV problem: a curious agent "finds a source of
randomness in the environment and keeps observing it" (Burda et al. 2018; OpenAI's write-up). Our pixel surprise fell
into it with night static (`20261123_eye_drives`).

**Ensembles: precedent, uses, limits (the user asked, 11-23).**
- **The idea:** several predictors trained on the same data agree where data is plentiful and disagree where it is
  thin, so their disagreement measures what hasn't been learned.
  - Query by committee chooses the next example "according to the principle of maximal disagreement", and gains
    information per query where random sampling gains almost none (Seung, Opper & Sompolinsky 1992).
  - Deep ensembles made it standard for deep networks (Lakshminarayanan et al. 2017).
- **Uses in RL:**
  - **exploration:**
    - Bootstrapped DQN has K heads on a shared network, each on its own bootstrap sample, and follows one head per
      episode (Osband et al. 2016). The shared network "can provide significant computational advantages at the
      cost of lower diversity between heads";
    - Plan2Explore uses ensemble disagreement as the reward;
  - **regularisation against exploitation:** ME-TRPO;
  - **horizon weighting:** STEVE (Buckman et al. 2018);
  - **pessimism offline:** MOPO.
- **Limits:**
  - heads on shared features can agree when the shared features missed the same thing (Codex; Osband's own caveat);
  - in latent dynamics models, transitions can be pulled toward well-supported regions, an attractor. Ensemble
    disagreement then "picks up local epistemic uncertainty but does not reliably track global error over long
    latent rollouts", and returns can be overestimated (Biased Dreams, Berger et al. 2026, abstract and summaries).
- **Without an ensemble:** Hansen & Wang's three signals come from the world model itself, "without auxiliary
  networks or labels":
  - flow instability: how much the clean-frame prediction moves across sampler substeps;
  - inter-seed variance: disagreement among independent samples at fixed past and action;
  - the tokenizer round-trip residual.

  Each is divided by how much the scene is moving, which removes the noise-like confound. They reach per-task AUROC
  0.87-0.94 for hallucination events, against 0.80-0.93 for scene motion alone, and they serve as curiosity rewards
  for targeted data collection, adapting the model to unseen tasks with 50 trajectories.
  - Our world is a shortcut-forcing sampler like theirs, so flow instability comes free in every dream, and
    inter-seed variance in any multi-draw reading.
  - Our 07-06 result agrees: the spread of independent samples predicted open-loop error at AUC 0.74 in Memory Maze,
    while Mamba's Δ did not (0.59).
- **In the brain:** no committee of world models, but three close relatives:
  - **sampling:** neural variability may be samples from the posterior, with spontaneous activity part of the
    representation (Fiser, Berkes, Orbán & Lengyel 2010). That is the relative of inter-seed variance;
  - **a population of diverse predictors:** dopamine neurons carry "a diverse set of RPE channels, each of which
    carries a different value prediction, with varying degrees of optimism" (Dabney et al. 2020). Their spread
    codes the return's distribution, closer to expected than epistemic uncertainty;
  - **reliability-based arbitration:** control is divided between model-based and model-free systems "dynamically
    weighted by the degree of reliability in each system", with the reliability signals in prefrontal cortex (Lee,
    Shimojo & O'Doherty 2014; Daw, Niv & Dayan 2005). That is the brain's version of deciding how far to trust
    imagination.

**What the system already computes, with no new module:**
- the world's loss on each new segment on arrival (Decision 4's plasticity reading): surprise at real experience;
- its drop after the online-queue update: learning progress, which "avoids situations in which nothing can be
  learnt" (Oudeyer, Kaplan & Hafner 2007);
- each crop's belief change: the eye's information drive;
- flow instability in every dream, and inter-seed variance wherever several draws are taken.

**Proposed: the world's own signals as readings from the start, uses on a measured trigger, validation before
use.**
1. **Readings, from the start:**
   - surprise on arrival and learning progress for every segment, logged by context (day, night, own life, corpus);
   - flow instability and inter-seed variance, each divided by scene motion (Hansen & Wang).
   - **No ensemble by default.** Outcome-head ensembles stay an evaluation tool, used if the world's own signals fail
     their validation. That follows Hansen & Wang's single-model route and Codex's caution.
2. **Validation before any use (Codex):** a signal earns a use only after it predicts what that use needs.
   - **For dream trust: wrong action orderings on real forks. Run 11-23 (`20261123_dream_trust/RESULTS.md`).**
     - Data: P1.5's archived 764 DO/NOOP forks, 8 dreams per action against 16 simulator futures.
     - **Raw spread fails:** the spread of the dreamed returns scores AUROC 0.48 at depth 15. Spread tracks how
       eventful a start is.
     - **The world's confidence in the comparison passes.** That is the dreamed gap over its own sampling error
       across the draws.
       - It scores 0.85 at depth 4 and 0.76 at depth 15. The simulator's own spread scores 0.76 and 0.59.
       - Confident orderings (abs z >= 2) are right 97% and 94% of the time; unconfident ones 72% and 58%.
     - **Confidently wrong orderings are 8% of the errors at depth 15**, Biased Dreams' false negatives. The checks
       against reality stay.
     - **Not separated:** part of low confidence is the 8 draws' own sampling noise. The use is the same either way.
     - **So the validated signal is a decision's confidence across several dreams.** It costs several draws per
       start, and it fits the evaluation readings at once and dream-trust weighting on its trigger.
   - For change: learning progress on the next real observations.
3. **Uses, each on its trigger:**
   - **Dream trust.**
     - Trigger: dreamed against real action differences show a bias (Decision 2's reading), or dreamed return rises
       while real return stalls.
     - Use: penalise dreamed reward by the validated signal (MOPO), or weight the return's horizons by it (STEVE).
       This is where the project's original uncertainty-set horizons would attach, as a later add-on.
   - **Curiosity.**
     - Trigger: real coverage stalls (achievement discovery flattens, the online queue brings little new).
     - Use: an intrinsic reward of learning progress or the validated signal (Plan2Explore; Hansen & Wang). Never raw
       prediction error.
   - **Change.**
     - Trigger: surprise on arrival stays above its own expectation for a context while learning progress there is
       high. That is Yu & Dayan's unexpected uncertainty.
     - Use: replay that context's new segments more, and discount dreams started before the change. Later, the fast
       channel and the episodic store (decisions 6 and 7).
     - Judge: the hidden-rule benchmark (the thirst period first).
4. **What it isn't:**
   - Mamba's Δ, rejected as an uncertainty signal on 07-06;
   - the pixel decoder's scale, which is blind to content (and the 10-31 pitfall);
   - raw prediction error as curiosity.
5. **Learning faster when uncertain: an extension, built on a measured trigger (the user, 11-23: "something we might
   have to measure once agentic Saccade has its first runs").**
   - **The brain:**
     - learning rates follow volatility (Behrens et al. 2007);
     - an approximately Bayesian delta rule sets the rate from change-point probability (Nassar et al. 2010);
     - acetylcholine and norepinephrine carry expected and unexpected uncertainty (Yu & Dayan 2005).
   - **In machine learning:**
     - Adam normalises step sizes by gradient magnitude, not by uncertainty;
     - IDBD meta-learns per-feature step sizes (Sutton 1992);
     - Degris et al. 2024 argue that RMSProp-like normalisation doesn't move step sizes toward their optimal values in
       continual learning.

     It is an open problem, not a solved component.
   - **What the design already does:**
     - new experience is trained on first (the online queue);
     - the change trigger above replays a changed context more. That raises its learning through data, as the brain's
       novelty-gated replay does (Lisman & Grace 2005), with no step-size knob.
   - **The trigger:** the hidden-rule benchmark shows adaptation too slow even with that replay. Then the candidate is
     per-module step-size adaptation (IDBD-like), on evidence.

### Decision 6: memory beyond the frame and the episode (opened 11-23; research and proposal, undecided)

**What it is for: four jobs.**
1. **Places beyond the frame, within an episode.** The world has to redraw what it has seen, in life (prediction) and
   in dreams. A dream that walks away and comes back should find the same place. Hippocampal amnesics' imagined scenes
   have details but no spatial coherence (Hassabis et al. 2007, READING.md).
2. **What the eye saw:** the glimpse's privileged memory route (Decision 3). It matters little in Classic, where the
   periphery shows it again, and more where acuity is low.
3. **One fact from one experience:** a single event that changes later predictions. This overlaps the fast channel
   (Decision 7).
4. **Across episodes.**
   - Craftax-Classic makes a new world every episode: all 1,009 DEV episodes start from distinct views at the same
     position. Places can't carry over; rules and skills can, through the weights.
   - A hidden rule that persists across episodes would need a memory that survives the reset. RL² keeps its hidden
     state "preserved to the next episode, but not preserved between trials" (Duan et al. 2016).
   - A robot's life has no resets.

**What the environment demands** (October's EDA, simulator replay, `20261023_system_design/OPTION3_NOTES.md`):
- **Re-entries:**
  - 56% of cells entering view are revisits (median gap 20 steps, p90 177);
  - in 16-step windows, entering terrain was last seen never 41%, within 16 steps 27%, at 16-64 14%, at 64-256 12%,
    and beyond 256 5% (M2).
- **Terrain is stable and self-motion is exact:**
  - terrain on revisits essentially never changes (S3: 0-0.07%);
  - self-motion is exactly determined by the action and the visible target cell (S1);
  - an identical view at another position happens on 0.03% of frames (S2).
- **What predicts re-entering terrain:** visible neighbours predict its block 77% of the time, and memory plus
  neighbours 91% (S5).
- **Entities** drift while out of view (S4: zombies median 3 cells at 16-64 steps), so they need a position belief
  that widens with absence.
- **Behaviour versus imagination:**
  - the expert loses nothing measurable when its memory beyond 64 steps is cut (M1: achievements within ±0.15);
  - except that water last seen 64-256 steps before took 20.8 more steps to reach with no memory at t0 = 1000;
  - so competent behaviour barely uses long memory, but imagination's fidelity does (M2, S5).

**What P1.5 does today:**
- Inside self-fed dreams, re-entering cells agree with their last clear sighting 0.68 of the time within 16 steps and
  0.60 at 17-64, against 0.94 / 0.88 for the simulator through the same readers (S2).
- **Measured 11-23 from real history** (`20261123_place_memory/RESULTS.md`; one-step dreams, 35,170 entering terrain
  cells, revisits against first sightings, matched by class):
  - P1.5 doesn't recall places at all, even after a few steps away;
  - where the visible neighbour doesn't give a cell away, revisits are drawn right 0.347 of the time at gaps 2-16,
    against 0.341 for cells never seen (class-matched difference +0.000 [-0.017, +0.019]);
  - where it does, both sit at 0.83.
  - So re-entering terrain is drawn from context alone, and S2's 0.68 was context, not recall.

**Precedents, read at abstract level unless noted:**
- **Video world models with memory:**
  - WorldMem keeps "a memory bank consisting of memory units that store memory frames and states (e.g., poses and
    timestamps)", retrieved by "field-of-view (FOV) overlap and temporal proximity". The generated frames attend to the
    retrieved frames as clean latents inside the diffusion denoiser (Xiao et al. 2025);
  - Context-as-Memory retrieves past frames by camera field-of-view overlap, rule-based (Yu et al. 2025);
  - both say a longer temporal window is too expensive.
- **Agents:**
  - Neural Episodic Control writes every experience into a dictionary that "does not learn when to write", so reads
    change faster than weights (Pritzel et al. 2017);
  - MERLIN finds "memory is not enough; it is critical that the right information be stored in the right format",
    and forms memories by predictive modelling (Wayne et al. 2018);
  - RL² carries memory across episodes;
  - TEM binds a path-integrated location code to content and predicts the view from location (Whittington et al.
    2020, READING.md).
- **Robots:**
  - maps keyed by pose (SLAM);
  - ReMEmbR stores captioned video segments with "timestamps and robot coordinates" in a vector database and queries
    them (Anwar et al. 2024).
- **The brain:**
  - **indexing:** the hippocampus stores an index to the neocortical patterns an experience activated, and "a partial
    cue that activated the index could activate the neocortical patterns" (Teyler & DiScenna 1986; Teyler & Rudy
    2007);
  - **novelty gates entry:** the hippocampus flags information its long-term store lacks, and the resulting dopamine
    enhances LTP (Lisman & Grace 2005);
  - **frames of reference:** long-term spatial memory is allocentric, short-term egocentric, translated by parietal
    and retrosplenial areas (Byrne, Becker & Burgess 2007);
  - **reality monitoring:** whether a memory was perceived or imagined is inferred from its qualities, not tagged
    (Johnson & Raye 1981);
  - **imagination recombines** remembered details (Schacter & Addis 2007).

**Proposed (undecided), building on the 10-23 choice (attention over (location, content) entries):**
1. **The store:**
   - per world cell, what was last seen there: the coarse token, plus the glimpse tokens where the eye looked;
   - stamped with the time;
   - read by location for the cells in view.

   It is WorldMem's form at our scale. A Classic map is 64 × 64 cells, so an episode's store is small enough to keep
   whole.
2. **Location by path integration from the world's own estimate of its displacement.**
   - Craftax makes this exact (S1, S2), and S2's camera-shift reader shows the shift is readable from the periphery.
   - Learned and inferred, never given, so it carries over to robots, where self-motion is noisy (the 10-23 decision:
     "coherent with noisy motion later").
3. **Read path:** the denoiser's cell tokens attend to their retrieved entries, keyed by location relative to the
   current position. This is the world's long-term memory; the record stays the screen-fixed short-term trace (clocks,
   recent change). The two jobs differ, as allocentric and egocentric memory do in the brain.
4. **Write policy:** every real frame writes what it shows (NEC's rule, no learned gate). Novelty-gated writing (Lisman
   & Grace) is a later refinement for capacity on long lives.
5. **Dreams write to a branch-local copy, never merged into real memory** (October's D5; reality monitoring).
6. **Scope:**
   - in Classic the store is cleared at a reset, because the next episode is a new world;
   - what persists across episodes is the weights and the replay;
   - a robot's store persists;
   - cross-episode fast memory for hidden rules is Decision 7's.
7. **Entities:** object records with a last-seen place and time and a widening belief (S4). These are designed in and
   built after the place store.
8. **Built on its trigger, which is met.**
   - The world has no recall beyond the frame (above), while the environment offers it: memory plus neighbours 91%
     against neighbours 77% (S5), terrain stable (S3), more than half of re-entries being revisits (E2).
   - The world is retrained for the agent anyway, so the store joins that run, as the record joined P1.5 on its
     trigger (10-27: "add I1 store / I2 records on measured triggers").
   - **Its judges:**
     - this measurement: the revisit-minus-first-sighting difference in the surprising context, which must rise
       above zero;
     - dreamed revisit agreement against the simulator's (S2);
     - dreamed against real action differences on routes back to remembered resources.
9. **Open for the user: how location is known.**
   - **Alignment of consecutive coarse frames over the candidate shifts.** It is visual odometry, exact in Craftax
     (S1, S2) and the standard robot estimate beside wheel or IMU odometry.
   - **Learned path integration (TEM-like).** It is the brain's way, and general, but it needs a training signal
     for the location code.

   Either way the position is inferred by the agent, never read from the simulator.

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
   The light-eye proposal (11-23) was withdrawn: the full eye stays, and the whole system is optimised instead (fused
   record, FlexAttention, a shared physical pass).
2. **Dreaming.**
   - Agreed: the step commits to the outcome before the look.
   - Decided: the glimpse reads nothing new, with no event code (11-23).
   - Proposed together: starts, horizon, returns and value (precedent defaults). The world learns from real
     experience, with P2 a possible later stage. The eye is targeted in dreams. Action differences are read at steps 1,
     4 and 15 inside the run (Codex).
   - Isolation, trust and fresh starts are recorded.
   - Practice first; deliberation is parked (its own section).
3. **Learning, awake and dreaming** (Decision 3, agreed 11-23):
   - the world: P1 losses on real data;
   - the motor choice: imitation, then PMPO in dreams with a frozen reference (β 0.3) replaced only by a policy that
     earned trust in reality; plus signed real advantages from recent life;
   - the value: dreams and real replay;
   - the look: relevance with a pre-look baseline, plus belief change from the acquired crop on real glimpses
     (predictive gain once memory keeps glimpses). The chooser and contingency retire.
4. **The life cycle** (Decision 4, agreed 11-23; numbers provisional until measured):
   - one learner; the awake agent adopts its weights every S real steps, with the episode's state rebuilt; dreams
     always on the current weights;
   - pretraining (world and eye), watching (adds the act tokens), then segments of living, each followed by the
     offline updates it buys (one knob, the replay ratio);
   - replay keeps everything, uniform plus an online queue; the corpus's share falls by itself;
   - a constant learning rate, one optimizer for life, plasticity tools on a measured trigger;
   - checks every 10% of the life; the budget arithmetic, with the step costs as the first build measurement.

**Designed in from the start, built on a measured trigger:**

5. **Knowing what it doesn't know** (Decision 5, opened 11-23; proposal above, undecided): readings from the start
   (surprise on arrival, learning progress, the world's own flow instability and inter-seed variance), validation
   before use, uses on triggers (dream trust, curiosity, change); ensembles only as an evaluation tool.
6. **Memory beyond the frame and the episode** (Decision 6, opened 11-23; proposal above, undecided): a
   location-keyed place store (trigger met: P1.5 recalls nothing beyond the frame), dreams on a branch copy, cleared at
   Classic's resets (new world each episode); location by frame alignment or learned path integration (open).
7. **The fast channel:** how adapting in context is learnt (variation in training, scoped above).

**Framing and judging:**

8. **Task specification and retasking:** what happened versus how a goal scores it; how new tasks enter
   (demonstrations, corrections, rewards).
9. **How goals shape learning** (on hold).
10. **The lifetime judge,** and what counts as a new situation.
11. **Acuity.**

**A separate track:** a good enough world. The day coin and the self-feeding fade are world quality, not agent design.
