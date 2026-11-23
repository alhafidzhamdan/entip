#!/usr/bin/env bash
# GPU: train a second finder with melanoma and NET added and prophase as a negative, then freeze it.
# If the job runs out of time, submit it again: training resumes from the last finished epoch.
#$ -N mitosis_finder2_train
#$ -S /bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -l h200=true
#$ -pe sharedmem 4
#$ -l h_rss=16G
#$ -l h_rt=48:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-finder2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nvidia-smi -L || true

FIRST="$BASE/runs/highgrade/mitosis-finder1"
RUN="$BASE/runs/highgrade/mitosis-finder2"
CODE="$REPO/training/runs/highgrade_mitosis"
if [ ! -f "$RUN/index.json" ]; then
  python "$CODE/finder_data.py" derive --source "$FIRST" --out "$RUN"
fi
if [ ! -f "$RUN/frozen.json" ]; then
  python "$CODE/finder.py" train --run "$RUN" --workers "${NSLOTS:-4}"
  python "$CODE/finder.py" freeze --run "$RUN"
fi
echo "done: $RUN/frozen.json"
