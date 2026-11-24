#!/usr/bin/env bash
# C2's location part, then C3 (PLAN.md, the eye's checks), one GPU job after the other.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MAMBA_DETERMINISTIC=1
D=artifacts/experiments/20261120_switching_dissection
.venv/bin/python -u -m artifacts.experiments.20261120_switching_dissection.c2_eye > $D/c2_eye.log 2>&1
.venv/bin/python -u -m artifacts.experiments.20261120_switching_dissection.c3 > $D/c3.log 2>&1
echo done > $D/eye_checks.done
