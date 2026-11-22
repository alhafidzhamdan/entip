#!/usr/bin/env bash
# CPU: reserve fresh patients for the final test, then retrain the MVP head on all labelled patients.
#$ -N mvp_head_pooled
#$ -S /bin/bash
#$ -pe sharedmem 4
#$ -l h_rss=12G
#$ -l h_rt=06:00:00
#$ -j y
#$ -o /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/tile-heads-r2/logs
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
R="$H/review-round2"
LABELS="$H/mvp-new1/labels-AH-new1.csv"
RESERVED="$H/mvp-final-test/patients.csv"
[ -f "$LABELS" ] || { echo "missing $LABELS: copy the new-patient labels to Eddie first" >&2; exit 1; }

OLD_HOLD=""; OLD_EXCL=""
if [ -f "$H/held_out.csv" ]; then OLD_HOLD="--holdout $H/held_out.csv"; OLD_EXCL="--exclude $H/held_out.csv"; fi
if [ -f "$RESERVED" ]; then
  echo "final-test patients already reserved: keeping them"
else
  # shellcheck disable=SC2086  # empty or two words; these paths have no spaces
  python "$REPO/training/runs/highgrade_mvp/reserve_test.py" --manifest "$S/trial.csv" --bags "$S/bags-h-optimus-0" \
    --exclude "$R/tile-review-key-round2.csv" --exclude "$H/mvp-new1/key.csv" $OLD_EXCL --out "$H/mvp-final-test"
fi

# shellcheck disable=SC2086
python "$REPO/training/runs/highgrade_common/tile_heads.py" train --feature mvp \
  --key "$R/tile-review-key-round2.csv" --ratings "$R/ratings-AH-round2-standard.csv" --rater AH \
  --locations "$H/tile-heads-r2/tile-locations.csv" --bags "$S/bags-h-optimus-0" \
  --recheck-key "$H/mvp-recheck1/key.csv" --recheck-labels "$H/mvp-recheck1/labels-AH-recheck1.csv" \
  --extra-key "$H/mvp-new1/key.csv" --extra-labels "$LABELS" \
  --holdout "$RESERVED" $OLD_HOLD --out "$H/tile-heads-pooled1"
echo "done: $H/tile-heads-pooled1/mvp/metrics.json"
