#!/usr/bin/env bash
# The switching arm and its readings (PLAN.md): the fine-tune (record route, p = 0.5, UPDATES updates), then S2 on its
# checkpoints with P1.5's simulator futures and frozen readers: stage 1 at UPDATES / 2 and UPDATES, stage 1b (competent
# starts, holds, spread, revisits) and the exact clock checks (P1.5's checks 17 and 15) at UPDATES; a resumable state at
# UPDATES (switch_ft --resume) extends it. Usage: bash run_arm.sh NAME UPDATES [extra switch_ft arguments, e.g. --substitute 0]
set -euo pipefail
cd "$(dirname "$0")/../../.."
NAME=$1; UPDATES=$2; shift 2
OUT=artifacts/experiments/20261119_switching/run/$NAME
LOG=artifacts/experiments/20261119_switching/$NAME
export MAMBA_DETERMINISTIC=1
HALF=$((UPDATES / 2))
.venv/bin/python -u -m artifacts.experiments.20261119_switching.switch_ft "$OUT" --routes rec --updates "$UPDATES" \
  --reads "$HALF,$UPDATES" "$@" > "$LOG.train.log" 2>&1
for U in $HALF $UPDATES; do                                              # S2 stage 1 at both reads
  export S2_CKPT=$OUT/ckpt_sw$(printf %04d "$U").pt S2_TAG=_${NAME}_$U
  .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2 model > "$LOG.s2_model_$U.log" 2>&1
  JAX_PLATFORMS=cpu .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2 gates > "$LOG.s2_gates_$U.log" 2>&1
done
.venv/bin/python -u -m artifacts.experiments.20261119_s2.s2b model > "$LOG.s2b_model.log" 2>&1          # stage 1b at the last
JAX_PLATFORMS=cpu .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2b gates > "$LOG.s2b_gates.log" 2>&1
export DISSECT_CKPT=$S2_CKPT DISSECT_TAG=$S2_TAG                         # the exact clock checks (Codex): 17, then 15
.venv/bin/python -u -m artifacts.experiments.20261116_p15.dissect schedule > "$LOG.check17.log" 2>&1
.venv/bin/python -u -m artifacts.experiments.20261116_p15.dissect catalogue > "$LOG.check15.log" 2>&1
echo done > "$LOG.done"
