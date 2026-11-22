#!/usr/bin/env bash
# Login node, read-only: count the patients still available to calibrate and test the frozen MVP head.
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
python "$REPO/training/runs/highgrade_mvp/validation_census.py" \
  --internal diverse "$BASE/runs/triage-diverse/trial.csv" "$BASE/runs/triage-diverse/bags-h-optimus-0" \
  --internal stage-a "$BASE/runs/triage-stage-a/trial.csv" "$BASE/runs/triage-stage-a/bags-h-optimus-0" \
  --external ipd "$BASE/runs/ipd/manifest.csv" "$BASE/runs/ipd/bags-h-optimus-0" \
  --external dbta "$BASE/runs/dbta/manifest.csv" "$BASE/runs/dbta/bags-h-optimus-0" \
  --used "$H/review-round2/tile-review-key-round2.csv" --used "$H/mvp-new1/key.csv" \
  --used "$H/mvp-retrieve1/key.csv" --used "$H/tile-heads-similar1/mvp/oof.csv" \
  --used "$H/mvp-final-test/patients.csv" --used "$H/mvp-outlines1/key.csv" \
  --held-out "$H/held_out.csv"
