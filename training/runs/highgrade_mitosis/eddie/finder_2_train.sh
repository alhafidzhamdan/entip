#!/usr/bin/env bash
# GPU: train the mitosis finder on public data, freeze the best model, and test it once on unseen tumour types.
# If the job runs out of time, submit it again: training resumes from the last finished epoch.
#$ -N mitosis_finder_train
#$ -q gpu
#$ -l gpu=1
#$ -l h200=true
#$ -pe sharedmem 4
#$ -l h_rss=16G
#$ -l h_rt=48:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-finder1/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nvidia-smi -L || true

RUN="$BASE/runs/highgrade/mitosis-finder1"
[ -f "$RUN/index.json" ] || { echo "missing $RUN/index.json: run finder_1_data.sh first" >&2; exit 1; }
if [ ! -f "$RUN/frozen.json" ]; then
  python "$REPO/training/runs/highgrade_mitosis/finder.py" train --run "$RUN" --workers "${NSLOTS:-4}"
  python "$REPO/training/runs/highgrade_mitosis/finder.py" freeze --run "$RUN"
fi
if [ ! -f "$RUN/unseen-test.json" ]; then
  python "$REPO/training/runs/highgrade_mitosis/finder.py" test --run "$RUN"
fi
echo "done: $RUN/unseen-test.json"
