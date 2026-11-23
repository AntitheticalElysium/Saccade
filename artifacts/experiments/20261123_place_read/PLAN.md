# Does the world use a place memory? A per-cell read of the last-seen coarse token (2026-11-23, decision 6)

**Why.** Decision 6's main risk is whether the denoiser learns to use a place memory (the user: "would the denoiser
even learn to trust it or use it properly"). This is a mechanism test of the read, as test 3 was for the record
(`20261116_temporal_memory`), and the build gate Codex proposed. It is not the agent build.

**The memory (decision 6's agreed form, its coarse half).**
- Per world cell, the last coarse token observed there and the episode frame it was observed.
- Written after every real frame for the 63 map cells in view; cleared at episode starts.
- Read before each frame: for each map cell of that frame, the entry at its world address.
  - The input is 21 numbers: the 16-number token, a presence flag, log age and three decays (8, 32, 128 frames).
  - It reaches the cell's denoiser token through a per-cell MLP (Linear 21 → width, GELU, Linear width → width), last
    layer zero-initialised, as the record's read entered in test 3.
- **Addresses:** from simulator positions, a declared substitution that isolates the read from localisation. The
  deployable estimate, action-constrained visual odometry, gets 97.2% of revisit addresses right (Codex's audit,
  reproduced). Inferred addresses come second, if this passes.

**Fine-tune (test 3's recipe):**
- P1.5 checkpoint 10; the denoiser and the new read trained, everything else frozen; G1 only.
- AdamW 1e-4, 100 warm-up updates, clip 1, 64-step chunks of P1.5's own streams, carry, cache and record.
- 4,000 updates; checkpoints at 2,000 and 4,000.
- Fixations are the stream's uniform draws. P1.5's chooser is not restored: it needs the whole gaze trace, and the
  read concerns coarse tokens only.
- Stored tokens are detached. The store carries across chunks, so revisits of any gap are read, while the encoder
  receives no gradient through the store.

**Claim.** The fine-tuned world draws re-entering cells it has seen better than cells it hasn't, where the visible
neighbour doesn't give them away. Today the class-matched revisit-minus-first-sighting difference is +0.000 [-0.017,
+0.019] at gaps 2-16 (`20261123_place_memory`).

**Competing explanation.** Any change comes from the extra training, not from the memory.

**Separating readings.**
- **First, the within-model revisit-minus-first-sighting difference.** First sightings have no entry, so more
  training alone would lift both.
- **Second, the causal swap** (Codex), on the same starts with view, action and history fixed. The entering cells'
  entries are given three ways: correct; empty; and the entry at a wrong address (3 cells away along rows).
- A matched continuation without the read is run only if these pass (controls after success).

**Readings, fixed before results** (the place-memory protocol: 120 DEV episodes, 40 movement starts each, one-step
dreams from real history, 4 draws, S2's terrain reader on the drawn and real frames; episode-bootstrap 95%
intervals):
1. **Agreement by group and context** at 2,000 and 4,000 updates; the class-matched revisit-minus-first difference
   in the surprising context.
   - **Bars at gap 2-16:** used, +0.20 or more (and +0.10 at 17-64); partial, +0.05 to +0.20; not used, below +0.05.
2. **The swap**, on surprising revisits: agreement with the correct, empty and wrong-address entry.
   - "Used" needs correct above empty, and wrong below empty or drawing the wrong entry's class above its base rate.
3. **Guards:**
   - first sightings in both contexts within 0.03 of P1.5;
   - the fine-tune's G1 training loss does not rise.

**Limits.**
- Coarse half only: no fine entries, no act-token read, no dream overlay.
- Oracle addresses; one seed; a fine-tune, not a world trained with the memory from scratch.

**Run:** `MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu .venv/bin/python -m artifacts.experiments.20261123_place_read.place_read train`,
then `... place_read eval 2000` and `... eval 4000`; `smoke` for a short check.
