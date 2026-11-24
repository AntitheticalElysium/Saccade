# S2 on P1.5 (2026-11-19; user: "build S2 and run it on P1.5", Codex reviews the build)

S2_PROPOSAL v2 (agreed in direction 11-07; its three open choices agreed 11-13: the learned chooser as trained on
model and simulator frames alike, BC built inside S2, the four primary families). P1.5 is the first world that passes
the timers (user 11-13: S2 waits for it). P2's objective is chosen from where this reading fails (user 11-07). The world
is frozen; nothing trains it. Checkpoint: `20261116_p15/run/ckpt_10.pt` (SHA256 `5b849b2a...`).

## Scope of this build

Stage 1 (this folder): L1 coherence, L2b trajectory distributions and internal consistency, N5 (the belief on its own
generated history), TEST_SPEC S3's bookkeeping blocker. L2a (depth 1 from real prefixes) is already read by P1.5's queue
(the determinant curves, check 15, check 17) and is cited, not rerun. Stage 2, after review: BC (P3's first stage) and
L3, the one-step planning test (real return of the imagination-guided policy minus BC's).

## Protocol (`s2.py`, one GPU job per stage, in order)

- **Bank:** the terminal-inclusive DEV bank (`20261114_s2_harness`: 4,000 uniformly sampled admitted 16-step starts
  from 257 episodes, 394 early-terminal, recorded deaths at every position, the whole horizon's precommitted actions).
- **readers:** S2 reads every trajectory, simulated or generated, through one set of frozen readers fitted on
  train-400 real percepts with the S0 probe protocol (`s1.fit`):
  - entities: S1's saved reader (`run/s1/results.reader.pt`), map cells, unchanged;
  - HUD: the 16 quantities (capped at 9) and asleep, from the two HUD rows' tokens and the lighting latent;
  - camera shift: 5 classes, from two consecutive frames' map tokens (positions are the start's plus the read shifts);
  - for N5: the same HUD targets from the corrected belief b_t.
  - Dev accuracy on the bank's episodes is reported beside every reading.
- **oracle:** the simulator's 16 futures of each start, with the same keys as `oracle_l1.py` (the same futures:
  A = 1-8, B = 9-16), rendered and read through the readers, beside their labels; the recorded future read the same
  way. Lighting comes from the simulator; it is a function of time, which every branch shares.
- **model:** the world's own futures, 8 per start.
  - It starts from the real prefix: belief and record from the episode start, the real context frames (up to 8).
  - Then 16 free-running steps under the start's precommitted actions.
  - Every fixation is chosen by the checkpoint's chooser between the cascade's two stages, from the generated
    periphery and its own 8-frame history (TEACHING 14's hook in `M.sample`).
  - Reward and termination are sampled by the world's heads; each branch absorbs on its own termination.
  - The record and the belief are carried and updated from the generated frames, as in every P1.5 rollout tool.
- **gates (CPU):**
  - **Instrument checks first:** the reader's error on simulator frames (labels against reads); the null (oracle A
    against oracle B, both through the readers) and the sanity check (recorded future through the readers against
    oracle B) must not reject; the observable-level corruptions of `oracle_l1.py` (fake consequences, immortal,
    sudden death), applied to oracle A through the readers, must be rejected. The seven record-level corruptions were
    shown rejected on labels (11-14) and cannot be applied after rendering; that power is not re-shown here.
  - **L1:** `oracle_l1.py`'s gates (1-4 step persistence, visible count, share terminating, mean absorbing step, the
    one-step event family; energy score as a diagnostic), model against oracle B through the same readers.
  - **TEST_SPEC S3 bookkeeping blocker (unchanged):** a kind's mean visible count off by more than 30% and 3 SE, or
    4-step persistence of zombies, cows or skeletons more than 0.10 below the reference.
  - **L2b (i):** for each start, the share of futures in which an event has happened by step k = 1..16, model
    against oracle B: damage, death, food / drink / energy decrements, waking (starts asleep), reward-bearing unlocks;
    the clocks stratified by steps since the last decrement in the real prefix, damage by a zombie adjacent at the
    start. Return per future.
  - **L2b (ii), internal consistency:** health read 0 without termination; termination with health read above 0;
    reward inconsistent with the read health change; inventory increments and decrements per item per step; each
    against the same statistic on oracle B.
  - **N5:** after k generated steps, the HUD read from the generated belief against the trajectory's own read HUD,
    beside the same reading on the real recorded future (its encoded belief against its read frames).
- **Reading rule:** the existing one (|difference| > 3 episode-bootstrap SE rejects; unmeasured is never a pass) and
  TEST_SPEC's blocker bounds. No new threshold is added; L2 families are reported as differences with intervals,
  ordered by the agreed primary families (damage and death given danger, energy and sleep timing, food and drink, the
  first achievements of the tech tree).

## Known limits

- A reader reads generated frames off its training domain; the null and sanity checks bound its error on real frames
  only (TEST_SPEC: its real-data error does not bound its error on generated ones).
- The second drop of a clock after a drawn first one needs more than 16 steps (cycles 21-31); not read here (Codex).
- One world seed.

## Correction during the run (11-19)

The model stage's prefix check (the rebuilt prefix belief against `encode`'s, an assert at RMS 0.03 borrowed from
`recursion_steps.py`, a Mamba-2 tool) fired at start 3,902 of 4,000 (t0 = 52, RMS 0.036). `prefix_check.py` measured
all 4,000 starts: median 0.006, 99.9th percentile 0.021, one start above 0.03, no growth with prefix length; on the
five worst starts an FP32 rebuild sits 0.007-0.030 from both bf16 computations, so neither is misaligned: it is
Mamba-3's bf16 floor. The assert is now a gross-misalignment guard at 0.1 and every start's error is saved; the model
stage and the gates were rerun from the start (same seeds). Nothing was read before the rerun.

## Stage 1b (11-19, user: finish S2 before switching; `s2b.py`)

The readings stage 1 lacked, on P1.5 (frozen), with stage 1's readers and reading rule:
- **Competent starts** (`bank_competent.json`): DEV episodes of collector epsilon 0.1 / 0.25 / 0.5, start step >= 7,
  admitted by position or the start state only. 1,000 uniform starts with fresh uniform random plans, and 764 hold pairs
  (Codex): the player awake facing a block it can collect (tree, stone, coal, iron, diamond, water: up to 150 each, the
  material below 9, the tool held), the same start run with DO or NOOP first and then the same 15 random actions.
  Median 15 achievements at the start (stage 1: 1). The reference is the simulator under the same plan, rendered and read.
- **Readings:**
  - On the random-plan starts: L1 (persistence, count, the bookkeeping blocker), L2b and consistency.
  - On the holds: the action's effect (DO minus NOOP on the collected quantity by steps 1 / 4 / 16), the world against
    the simulator through the readers, beside the simulator's label effect.
  - On both banks: spread (randomised PIT of each simulator future among the world's 8; the null of oracle B among A
    beside it) and revisits (TEST_SPEC S2.2 over the real 64-frame prefix and the rollout, by gap).
- **Route localisation** (for the switching plan's step 0): stage 1's bank, 1,000 starts x 4 futures, the recorded frame
  fed to the context, to Mamba and the record, to both, or to neither.
- **Reader caveat** found in the smoke: on competent simulator frames the HUD reader is less exact (0.92-0.97 per
  quantity in 24 starts) than on random play (about 1.0); the label effect is reported beside the read one for that reason.
- **Codex's review (11-19), adopted before any stage 1b reading:**
  - The scorer passed the full bank's length to the bank selector, which reads any length as a smoke slice (264 rows
    duplicated, 264 omitted, 1,084 material labels wrong); full mode now loads the full bank and asserts row identity.
    The rollouts were unaffected.
  - The route arms now share the recording's fixation schedule (a recorded glimpse and a generated one at the same
    crop, no chooser), feed the belief the world's own sampled reward and termination in every arm (frames differ only),
    and report count changes paired against free running, with survival beside them. The route stage had not started
    when this was changed.
- **Mamba against the record (11-19, user: run before the switching review).** The route test's memory arm fed recorded
  frames to Mamba and the record together. Two more arms on the same 1,000 starts, seeds and fixation schedule, paired
  against its free arm (the first generated frame must agree exactly):
  - record_real: the record alone reads the recorded frames (the denoiser's per-cell read of it, and Mamba's read of it);
  - mamba_real: Mamba's percept input alone reads them.
  Read: mobs per frame at steps 1 / 4 / 8 / 16 per arm and kind, paired changes against free running, survival. No
  threshold: it chooses between the denoiser-only fine-tune (the record's read is denoiser parameters) and training
  Mamba too (switching plan, variant 5).
- **The two readers of the record (11-19, after the switching arm; user: run before its rerun).** record_real fed the
  recorded frames to both readers of the record at once: the denoiser's per-cell read (`denoiser.record_read`) and
  Mamba's (`belief.record_in`); the record itself has no parameters. My switching plan took its 94% for the denoiser's
  read alone, which the arm never separated. Two more arms on the same 1,000 starts, seeds and fixation schedule, each
  reader keeping its own record state, paired against free running (first generated frame identical):
  - dread_real: only the denoiser's read takes the record of recorded frames (Mamba reads the record of generated ones);
  - mread_real: only Mamba's read takes it (and the outcome heads, which read through Mamba's path, as in record_real).
  Read, fixed now: a reader carries the loss if its arm recovers at least two thirds of record_real's step-16 gain in
  all mobs (+0.43); then the rerun trains that reader. Neither alone (or both): the rerun trains both readers.
  Identity check before the run, in one process: the committed code and this one give bitwise-equal free and
  record_real arms on a 16-start smoke.
  Run 1 (22:26) stopped at the scorer's first-frame assert: 5 of 4,000 first generated frames differ from routes.npz's
  free arm (made in another process at 15:05), identically in both reader arms, so a cross-process drift, not an arm
  effect. The readers file now carries its own free and record_real arms from the same process and is scored against
  them only.
