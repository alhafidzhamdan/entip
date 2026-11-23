#!/usr/bin/env bash
# GPU: seal the glioma reference patients, draw 12 other TCGA patients, run the finder on 9 and build the review packs.
#$ -N mitosis_glioma_round1
#$ -S /bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -l h200=true
#$ -pe sharedmem 4
#$ -l h_rss=16G
#$ -l h_rt=12:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-glioma-round1/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nvidia-smi -L || true

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
REF="$H/mitosis-ref1"
OUT="$H/mitosis-glioma-round1"
CODE="$REPO/training/runs/highgrade_mitosis"
FINDER="$H/mitosis-finder2"
[ -f "$FINDER/frozen.json" ] || { echo "missing $FINDER/frozen.json: finder_3_retrain.sh must finish first" >&2; exit 1; }

mkdir -p "$REF/private" "$OUT/private"
chmod -R go-rwx "$REF/private" "$OUT/private"
if [ ! -f "$REF/private/fields.csv" ]; then
  python "$CODE/reference_set.py" draw --manifest "$S/trial.csv" --bags "$S/bags-h-optimus-0" --slides "$S/slides" --out "$REF" > "$REF/private/draw.json"
fi
if [ ! -f "$OUT/private/fields.csv" ]; then
  python "$CODE/glioma_review.py" draw --manifest "$S/trial.csv" --bags "$S/bags-h-optimus-0" --slides "$S/slides" \
    --reference "$REF/private/patients.csv" --out "$OUT"
fi
if [ ! -f "$OUT/private/candidates.csv" ]; then
  python "$CODE/glioma_review.py" calls --finder "$FINDER" --slides "$S/slides" \
    --reference "$REF/private/patients.csv" --out "$OUT"
fi
if [ ! -f "$OUT/private/calls-key.csv" ]; then
  python "$CODE/glioma_review.py" build --slides "$S/slides" --reference "$REF/private/patients.csv" \
    --workers "${NSLOTS:-1}" --out "$OUT"
fi
chmod -R go-rwx "$REF/private" "$OUT/private"
echo "done: $OUT/full-fields-pack.tar and $OUT/calls-pack.tar"
