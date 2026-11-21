#!/bin/sh
# P1.6's remaining evaluation (11-21), after C15 frees the GPU: the added test (blank check, then native pairs and
# consequences with the record writing no glimpse), the paused queue, reading 5 (zombies), reading 7 (trace, CPU).
# Each step is its own capped unit; the chain stops at the first failure. Run from the repository root.
set -e
H=artifacts/experiments/20261120_p16
CK=$PWD/$H/run/ckpt_10.pt
B=$H/readings_blank_fovea
mkdir -p $B $H/readings/logs
ENV="-E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu -E PYTHONPATH=$PWD -E OMP_NUM_THREADS=2 -E MKL_NUM_THREADS=2 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
step() {  # unit, memory cap, command
  echo "$(date +%T) $1 start"
  systemd-run --user --unit=$1 --collect --wait -q -p MemoryMax=$2 -p MemorySwapMax=0 $ENV --working-directory=$PWD /bin/sh -c "$3"
  echo "$(date +%T) $1 done"
}
step p16-blank-check 8G ".venv/bin/python -u -m artifacts.experiments.20261120_p16.blank_check $CK > $B/blank_check.log 2>&1"
step p16-blank-native 12G "P16_BLANK_FOVEA=1 .venv/bin/python -u -m artifacts.experiments.20261120_p16.evaluation --tool artifacts.experiments.20261111_p1f5.native_pairs $CK $B/native_pairs.json > $B/native_pairs.log 2>&1"
step p16-blank-consequences 14G "P16_BLANK_FOVEA=1 .venv/bin/python -u -m artifacts.experiments.20261120_p16.evaluation --tool artifacts.experiments.20261109_p1_hud_change.consequences $CK $B/consequences > $B/consequences.log 2>&1"
systemctl --user is-active -q saccade-p16-memwatch || systemd-run --user --unit=saccade-p16-memwatch --collect -p MemoryMax=256M --working-directory=$PWD /bin/sh $H/memwatch.sh
echo "== $(date -Iseconds) queue resumed by run_rest.sh" >> $H/evaluation.log
step saccade-p16-eval 18G ".venv/bin/python -u -m artifacts.experiments.20261120_p16.evaluation $CK >> $H/evaluation.log 2>&1"
step p16-zombies 16G ".venv/bin/python -u -m artifacts.experiments.20261120_p16.readings zombies $CK > $H/readings/logs/zombies.log 2>&1"
step p16-trace 12G ".venv/bin/python -u -m artifacts.experiments.20261120_p16.readings trace > $H/readings/logs/trace.log 2>&1"
echo "$(date +%T) all done"
