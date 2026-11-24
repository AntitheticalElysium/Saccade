#!/usr/bin/env bash
# Stage B (PLAN.md): the K sweep (S2's world futures on the first 1,000 starts at sampler steps K = 1, 2, 4, 8; files
# model_K*_smoke.npz, "smoke" being S2's name for a partial bank), then B2, then the CPU readings of the sweep.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MAMBA_DETERMINISTIC=1
D=artifacts/experiments/20261120_switching_dissection
for K in 1 2 4 8; do
  S2_STEPS=$K S2_TAG=_K$K .venv/bin/python -u -m artifacts.experiments.20261119_s2.s2 model 1000 > $D/ksweep_K$K.log 2>&1
done
.venv/bin/python -u -m artifacts.experiments.20261120_switching_dissection.b2 > $D/b2.log 2>&1
EV_WORLDS=simulator,K1_smoke,K2_smoke,K4_smoke,K8_smoke EV_REF=K4_smoke EV_N=1000 EV_TAG=ksweep \
  .venv/bin/python -u -m artifacts.experiments.20261120_switching_dissection.events > $D/ksweep_events.log 2>&1
.venv/bin/python -u -m artifacts.experiments.20261120_switching_dissection.deletions K1_smoke K2_smoke K4_smoke K8_smoke > $D/ksweep_deletions.log 2>&1
echo done > $D/stage_b.done
