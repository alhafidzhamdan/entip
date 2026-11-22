#!/usr/bin/env bash
# CPU: retrain the MVP head with the high-power recheck labels, in its own folder.
#$ -N mvp_head_recheck
#$ -S /bin/bash
#$ -pe sharedmem 4
#$ -l h_rss=8G
#$ -l h_rt=02:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
R="$H/review-round2"
LABELS="$H/mvp-recheck1/labels-AH-recheck1.csv"
[ -f "$LABELS" ] || { echo "missing $LABELS: copy the recheck labels to Eddie first" >&2; exit 1; }
python "$REPO/training/runs/highgrade_common/tile_heads.py" train --feature mvp \
  --key "$R/tile-review-key-round2.csv" --ratings "$R/ratings-AH-round2-standard.csv" --rater AH \
  --locations "$H/tile-heads-r2/tile-locations.csv" --bags "$BASE/runs/triage-diverse/bags-h-optimus-0" \
  --recheck-key "$H/mvp-recheck1/key.csv" --recheck-labels "$LABELS" --out "$H/tile-heads-r2-recheck1"
echo "done: $H/tile-heads-r2-recheck1/mvp/metrics.json"
