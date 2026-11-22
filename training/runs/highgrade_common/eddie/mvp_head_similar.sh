#!/usr/bin/env bash
# CPU: retrain the MVP head on every labelled tile, including the reviewed similar tiles, in its own folder.
#$ -N mvp_head_similar
#$ -S /bin/bash
#$ -pe sharedmem 4
#$ -l h_rss=16G
#$ -l h_rt=08:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
R="$H/review-round2"
LABELS="$H/mvp-retrieve1/labels-AH-retrieve1.csv"
RESERVED="$H/mvp-final-test/patients.csv"
[ -f "$LABELS" ] || { echo "missing $LABELS: copy the similar-tile labels to Eddie first" >&2; exit 1; }
[ -f "$RESERVED" ] || { echo "missing $RESERVED: the final-test patients must be reserved first" >&2; exit 1; }
OLD_HOLD=""
if [ -f "$H/held_out.csv" ]; then OLD_HOLD="--holdout $H/held_out.csv"; fi
# shellcheck disable=SC2086  # empty or two words; these paths have no spaces
python "$REPO/training/runs/highgrade_common/tile_heads.py" train --feature mvp \
  --key "$R/tile-review-key-round2.csv" --ratings "$R/ratings-AH-round2-standard.csv" --rater AH \
  --locations "$H/tile-heads-r2/tile-locations.csv" --bags "$BASE/runs/triage-diverse/bags-h-optimus-0" \
  --recheck-key "$H/mvp-recheck1/key.csv" --recheck-labels "$H/mvp-recheck1/labels-AH-recheck1.csv" \
  --extra-key "$H/mvp-new1/key.csv" --extra-labels "$H/mvp-new1/labels-AH-new1.csv" \
  --similar "$H/mvp-retrieve1/key.csv" "$LABELS" \
  --holdout "$RESERVED" $OLD_HOLD --out "$H/tile-heads-similar1"
echo "done: $H/tile-heads-similar1/mvp/metrics.json"
