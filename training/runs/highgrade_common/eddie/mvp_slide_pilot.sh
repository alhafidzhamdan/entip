#!/usr/bin/env bash
# CPU: freeze the MVP head, score the 35 reserved slides privately, and build the blinded whole-slide review pack.
#$ -N mvp_slide_pilot
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
python "$REPO/training/runs/highgrade_mvp/slide_pilot.py" build --head "$H/tile-heads-similar1/mvp/head.joblib" \
  --reserved "$H/mvp-final-test/patients.csv" --trained "$H/tile-heads-similar1/mvp/oof.csv" \
  --bags "$S/bags-h-optimus-0" --slides "$S/slides" --workers "${NSLOTS:-1}" --out "$H/mvp-pilot1"
chmod -R go-rwx "$H/mvp-pilot1/private"
echo "done: $H/mvp-pilot1/review-pack.tar"
