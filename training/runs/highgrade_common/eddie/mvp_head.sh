#!/usr/bin/env bash
# CPU: find the rated review tiles on their slides, then train and score the MVP head. Skips the first step if done.
#$ -N mvp_head
#$ -S /bin/bash
#$ -pe sharedmem 4
#$ -l h_rss=8G
#$ -l h_rt=04:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
R="$H/review-round2"
OUT="$H/tile-heads-r2"
TOOL="$REPO/training/runs/highgrade_common/tile_heads.py"
KEY="$R/tile-review-key-round2.csv"
RATINGS="$R/ratings-AH-round2-standard.csv"
for f in "$KEY" "$RATINGS"; do
  [ -f "$f" ] || { echo "missing $f: copy the round 2 files to Eddie first" >&2; exit 1; }
done

if [ -f "$OUT/tile-locations.csv" ]; then
  echo "tile locations already found: skipping locate"
else
  # Each worker reads one slide, so its maths libraries use one core.
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python "$TOOL" locate --workers "${NSLOTS:-1}" --key "$KEY" --tiles "$H/grade-mil/tiles" \
    --bags "$S/bags-h-optimus-0" --slides "$S/slides" --out "$OUT" --leave-out T106 T313  # both rated artefact
fi

python "$TOOL" train --feature mvp --key "$KEY" --ratings "$RATINGS" --rater AH \
  --locations "$OUT/tile-locations.csv" --bags "$S/bags-h-optimus-0" --out "$OUT"
echo "done: $OUT/mvp/metrics.json"
