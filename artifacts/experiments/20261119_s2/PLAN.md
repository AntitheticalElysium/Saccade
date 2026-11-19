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
