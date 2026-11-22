#!/usr/bin/env bash
# CPU: draw 24 patients at random, with no model, and build the blinded whole-slide necrosis review pack.
#$ -N necrosis_fresh_review
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/necrosis-fresh2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
OUT="$H/necrosis-fresh2"
# Necrosis is a separate classifier, so MVP patients may be shared; only the project's held-out patients stay out
EXCLUDE=()
if [ -f "$H/held_out.csv" ]; then EXCLUDE+=(--exclude "$H/held_out.csv"); fi

mkdir -p "$OUT/private"
chmod -R go-rwx "$OUT/private"
if [ ! -f "$OUT/private/patients.csv" ]; then
  python "$REPO/training/runs/highgrade_necrosis/slide_review.py" draw --manifest "$S/trial.csv" \
    --bags "$S/bags-h-optimus-0" ${EXCLUDE[@]+"${EXCLUDE[@]}"} --out "$OUT/private"
fi
python "$REPO/training/runs/highgrade_necrosis/slide_review.py" build --patients "$OUT/private/patients.csv" \
  ${EXCLUDE[@]+"${EXCLUDE[@]}"} --bags "$S/bags-h-optimus-0" --slides "$S/slides" --expect 24 \
  --workers "${NSLOTS:-1}" --out "$OUT"
chmod -R go-rwx "$OUT/private"
echo "done: $OUT/review-pack.tar"
