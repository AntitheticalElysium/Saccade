#!/usr/bin/env bash
# The switching arm and its readings (PLAN.md): the fine-tune (mem route, p = 0.5, 4,000 updates), then S2 on its
# checkpoints with P1.5's simulator futures and frozen readers: stage 1 at 2,000 and 4,000 updates, stage 1b (competent
# starts, holds, spread, revisits) at 4,000. Usage: bash run_arm.sh NAME [extra switch_ft arguments, e.g. --substitute 0]
set -euo pipefail
cd "$(dirname "$0")/../../.."
NAME=$1; shift
OUT=artifacts/experiments/20261119_switching/run/$NAME
LOG=artifacts/experiments/20261119_switching/$NAME
export MAMBA_DETERMINISTIC=1
.venv/bin/python -u -m artifacts.experiments.20261119_switching.switch_ft "$OUT" --routes mem "$@" > "$LOG.train.log" 2>&1
for U in 2000 4000; do
  export S2_CKPT=$OUT/ckpt_sw$U.pt S2_TAG=_${NAME}_$U
  .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2 model > "$LOG.s2_model_$U.log" 2>&1
  JAX_PLATFORMS=cpu .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2 gates > "$LOG.s2_gates_$U.log" 2>&1
done
.venv/bin/python -u -m artifacts.experiments.20261119_s2.s2b model > "$LOG.s2b_model_4000.log" 2>&1
JAX_PLATFORMS=cpu .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2b gates > "$LOG.s2b_gates_4000.log" 2>&1
echo done > "$LOG.done"
