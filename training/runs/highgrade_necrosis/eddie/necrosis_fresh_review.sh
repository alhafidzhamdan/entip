#!/usr/bin/env bash
# CPU: draw 24 fresh patients at random, with no model, and build the blinded whole-slide necrosis review pack.
#$ -N necrosis_fresh_review
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/necrosis-fresh1/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
OUT="$H/necrosis-fresh1"
EXCLUDE=(--exclude "$H/review-round2/tile-review-key-round2.csv" --exclude "$H/mvp-new1/key.csv"
         --exclude "$H/mvp-retrieve1/key.csv" --exclude "$H/mvp-final-test/patients.csv")
if [ -f "$H/held_out.csv" ]; then EXCLUDE+=(--exclude "$H/held_out.csv"); fi

mkdir -p "$OUT/private"
chmod -R go-rwx "$OUT/private"
if [ ! -f "$OUT/private/patients.csv" ]; then
  python "$REPO/training/runs/highgrade_mvp/reserve_test.py" --manifest "$S/trial.csv" --bags "$S/bags-h-optimus-0" \
    "${EXCLUDE[@]}" --per-category gbm=8 astro4=6 oligo3=4 astro3=3 grade2=3 --seed 20261123 --out "$OUT/private"
fi
python "$REPO/training/runs/highgrade_necrosis/slide_review.py" build --patients "$OUT/private/patients.csv" \
  "${EXCLUDE[@]}" --bags "$S/bags-h-optimus-0" --slides "$S/slides" --expect 24 --workers "${NSLOTS:-1}" --out "$OUT"
chmod -R go-rwx "$OUT/private"
echo "done: $OUT/review-pack.tar"
