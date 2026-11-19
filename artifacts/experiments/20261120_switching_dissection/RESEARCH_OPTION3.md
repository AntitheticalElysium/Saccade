# Option 3, attention-built records: research pass (2026-11-20; user: neuroscience and existing implementations first)

This pass draws on abstracts, publisher pages and reviews. Full texts were not read except where stated, and effect
sizes are not checked.

## Neuroscience

- **Object files** ([Kahneman, Treisman & Gibbs 1992](https://doi.org/10.3758%2FBF03193644)).
  - An object file is a temporary episodic representation in which an object's successive states are linked.
  - The preview benefit is object-specific and does not shrink with a 1 s preview or a 590 ms gap: a short absence
    does not erase the file.
  - A new file opens when a new object appears.
  - A replication (Moreno-Ríos & Tudela 2001) did not reproduce every experiment.
- **Coherence theory** ([Rensink 2000](https://www.cs.ubc.ca/~rensink/flicker/abs.00.1a.html)).
  - Proto-objects form in parallel across the field but are volatile: they are replaced by any new stimulus at their
    location.
  - Focused attention gives one object at a time spatiotemporal coherence. When attention is released, the object
    "dissolves back into its constituent proto-objects".
  - Coordinated attention creates a "virtual representation", and change blindness follows.
  - So the brain keeps no detailed record of unattended objects while the world is there to look at.
- **Transsaccadic object files** (Irwin's theory; summarised in
  [the review](https://pmc.ncbi.nlm.nih.gov/articles/PMC2784885)).
  - Only 3-4 object files, spatially indexed property lists, survive an eye movement.
  - Memory is best for the 3-4 most recently fixated objects, and near the saccade target.
  - Hollingworth: transsaccadic memory and visual short-term memory both hold 3-4 objects, limited by the number of
    objects, not of features ([Hollingworth & Henderson 2002](https://iro.uiowa.edu/esploro/outputs/journalArticle/Accurate-Visual-Memory-for-Previously-Attended/9984213273802771)).
- **Refreshing** ([Souza, Rerko & Oberauer 2015](https://www.psychologie.uzh.ch/dam/jcr:ffffffff-bd53-4653-0000-0000157c904b/Souza_etal_2015_ANYAS.pdf)).
  - Attending to an item held in visual working memory improves its later recall, more so the more often it is
    refreshed.
  - A 2025 study: refreshing does not protect against interference.
- **Attention pointers remapped across eye movements** ([Cavanagh, Hunt, Afraz & Rolfs 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2847621)).
  - The locations of attended targets are updated by predictive shifts of "pointers", a corollary discharge of the
    movement.
  - Identity has to be kept by an additional mechanism.
- **Gaze in imagination.**
  - Imagery re-enacts the gaze of perception, landing where things were ([Laeng & Teodorescu 2002](https://www.jimdavies.org/summaries/LaengTeodorescu2002.html)).
  - More faithful enactment goes with more accurate memory, and restricting gaze costs memory ([Laeng et al. 2014](https://cris.unibo.it/handle/11585/580275)).
  - REM saccades are gaze shifts in the dream world ([Senzai & Scanziani 2022](https://www.ucsf.edu/news/2022/08/423531/eye-movements-rem-sleep-mimic-gazes-dream-world)),
    each followed by visual-like processing ([Andrillon et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4866865)).

**What this implies for a design:**
- Records are few (3-4).
- Attention opens them.
- They are indexed by location.
- They are remapped by self-motion and refreshed by re-attention.
- They survive short absences.
- Unattended content stays coarse and volatile, which is what our periphery and per-cell record already are. In
  imagination, gaze returns to stored items and regenerates them.

## Implementations and their findings

- **Attention-gated sparse updates.**
  - [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893v2): modules interact through an attention
    bottleneck and update only when relevant. Generalisation improves under shifts.
  - [SCOFF, Object Files and Schemata](https://arxiv.org/abs/2006.16225v4): per-object files plus shared dynamics
    (schemata); attention decides which files update. A drop-in RNN replacement with better generalisation and
    next-state prediction on intuitive physics.
- **Location-indexed memory with local writes.**
  - [Neural Map](https://arxiv.org/pdf/1702.08360.pdf): a 2-D memory written only at the agent's position (the sparse
    write prevents overwriting). It beats LSTM and memory networks on 2-D and 3-D mazes and generalises to unseen
    mazes.
  - [WorldMem](https://arxiv.org/html/2504.12369v1): a video world model (diffusion forcing) with a bank of past frames
    plus pose and time, retrieved by field-of-view overlap through memory attention. Consistency on revisits improves.
    Numbers not checked.
- **Attention that writes.** [DRAW](https://arxiv.org/abs/1502.04623): differentiable read and write attention; the
  decoder modifies only the attended region of a canvas.
- **Learned object slots.** [SAVi](https://pdf.arxiv.org/pdf/2111.12594): a predict-correct slot recurrence. On
  realistic data, binding needed optical-flow targets and location cues for the first frame; SAVi++ adds depth for
  real video. Binding needs strong cues.
- **Learned "where to look" in RL.**
  - [SUGARL](https://arxiv.org/abs/2306.00975v1): a sensory policy rewarded for observations that predict the agent's
    own action. Fixation and tracking emerge.
  - [Glimpse-Attend-and-Explore](https://arxiv.org/abs/2108.11717v1): glimpses chosen by self-attention, for scene
    reconstruction.
- **Permanence lives on the predictor's side.** [Tracking Is Not Permanence](https://arxiv.org/abs/2610.07355): the
  encoder holds the hidden object, the predictor drops it, and cheap predictor-only training installs it.

## Reading for Saccade

- **Two families fit our structure.**
  - Location-indexed hold memory (Neural Map / WorldMem style): a second per-cell channel written only at fixated
    cells, holding content plus age.
  - Object files (SCOFF / transsaccadic style): K = 3-4 slots, each opened by a fixation and holding content plus a
    location pointer. The pointer is remapped by the camera shift and refreshed by re-fixation; the denoiser reads the
    slots as tokens placed at their pointers.
  - Both avoid unsupervised binding (SAVi's hard part): the eye chooses what becomes a record.
- **Open design questions:**
  - Remapping on scroll: corollary discharge from the action, or learned registration.
  - What a file is worth to the reader. In real data a file and the periphery disagree when the object moved, or at
    night when the coarse periphery misses what the fovea sees. Age has to carry that calibration.
  - The cascade: today the glimpse is drawn after and from the periphery, so a file can protect a mob only if the
    periphery's own generation reads it.
- **Scope of the expected benefit:** recovery after a deletion, the night and low-acuity misses, and an eye with
  memory. Not the daytime base loss, which occurs with full information.
