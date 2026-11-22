#!/usr/bin/env bash
# CPU: draw 20 fresh 40x patients with no model, pick 80 random 0.5 mm2 fields, and build the blinded counting packs.
#$ -N mitosis_reference
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-ref1/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
OUT="$H/mitosis-ref1"

mkdir -p "$OUT/private"
chmod -R go-rwx "$OUT/private"
if [ ! -f "$OUT/private/fields.csv" ]; then
  python "$REPO/training/runs/highgrade_mitosis/reference_set.py" draw --manifest "$S/trial.csv" \
    --bags "$S/bags-h-optimus-0" --slides "$S/slides" --out "$OUT"
fi
python "$REPO/training/runs/highgrade_mitosis/reference_set.py" build --bags "$S/bags-h-optimus-0" --slides "$S/slides" \
  --workers "${NSLOTS:-1}" --out "$OUT"
chmod -R go-rwx "$OUT/private"
echo "done: $OUT/review-pack.tar and $OUT/second-reader-pack.tar"
