#!/usr/bin/env bash
# CPU: label tiles from the locked slide-pilot outlines and train a new MVP head. The frozen pilot head is not touched.
#$ -N mvp_head_outlines
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
S="$BASE/runs/triage-diverse"
R="$H/review-round2"
P="$H/mvp-pilot1"
O="$H/mvp-outlines1"
OUT="$H/tile-heads-outlines1"
[ -f "$P/reveal/location-results.json" ] || { echo "missing the pilot's location results: finish the pilot first" >&2; exit 1; }
[ ! -e "$OUT/mvp/head.joblib" ] || { echo "$OUT/mvp/head.joblib already exists" >&2; exit 1; }
[ -f "$O/key.csv" ] || python "$REPO/training/runs/highgrade_mvp/outline_labels.py" --pilot "$P" \
  --calls "$P/slide-calls-AH.csv" --bags "$S/bags-h-optimus-0" --out "$O"
OLD_HOLD=""
if [ -f "$H/held_out.csv" ]; then OLD_HOLD="--holdout $H/held_out.csv"; fi
# The pilot's reserved patients are deliberately not held out here: this head trains on them.
# shellcheck disable=SC2086  # empty or two words; these paths have no spaces
python "$REPO/training/runs/highgrade_common/tile_heads.py" train --feature mvp \
  --key "$R/tile-review-key-round2.csv" --ratings "$R/ratings-AH-round2-standard.csv" --rater AH \
  --locations "$H/tile-heads-r2/tile-locations.csv" --bags "$S/bags-h-optimus-0" \
  --recheck-key "$H/mvp-recheck1/key.csv" --recheck-labels "$H/mvp-recheck1/labels-AH-recheck1.csv" \
  --extra-key "$H/mvp-new1/key.csv" --extra-labels "$H/mvp-new1/labels-AH-new1.csv" \
  --similar "$H/mvp-retrieve1/key.csv" "$H/mvp-retrieve1/labels-AH-retrieve1.csv" \
  --similar "$O/key.csv" "$O/labels.csv" \
  $OLD_HOLD --out "$OUT"
echo "done: $OUT/mvp/metrics.json"
