#!/usr/bin/env bash
# The switching arm and its readings (PLAN.md): the fine-tune (record route, p = 0.5, UPDATES updates), then S2 on its
# last checkpoint with P1.5's simulator futures and frozen readers: stage 1 and stage 1b (competent starts, holds,
# spread, revisits). Usage: bash run_arm.sh NAME UPDATES [extra switch_ft arguments, e.g. --substitute 0]
set -euo pipefail
cd "$(dirname "$0")/../../.."
NAME=$1; UPDATES=$2; shift 2
OUT=artifacts/experiments/20261119_switching/run/$NAME
LOG=artifacts/experiments/20261119_switching/$NAME
export MAMBA_DETERMINISTIC=1
.venv/bin/python -u -m artifacts.experiments.20261119_switching.switch_ft "$OUT" --routes rec --updates "$UPDATES" \
  --reads "$UPDATES" "$@" > "$LOG.train.log" 2>&1
export S2_CKPT=$OUT/ckpt_sw$(printf %04d "$UPDATES").pt S2_TAG=_${NAME}_$UPDATES
.venv/bin/python -u -m artifacts.experiments.20261119_s2.s2 model > "$LOG.s2_model.log" 2>&1
JAX_PLATFORMS=cpu .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2 gates > "$LOG.s2_gates.log" 2>&1
.venv/bin/python -u -m artifacts.experiments.20261119_s2.s2b model > "$LOG.s2b_model.log" 2>&1
JAX_PLATFORMS=cpu .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2b gates > "$LOG.s2b_gates.log" 2>&1
echo done > "$LOG.done"
