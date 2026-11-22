#!/usr/bin/env bash
# Login node, read-only: size a glioma-only download of unused patients for validating the frozen MVP head.
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env

H="$BASE/runs/highgrade"
W="$BASE/runs/triage-diverse"
python "$REPO/training/runs/highgrade_mvp/validation_size.py" \
  --manifest "$W/manifest.csv" --idh "$W/idh.tsv" --bags "$W/bags-h-optimus-0" \
  --used "$H/review-round2/tile-review-key-round2.csv" --used "$H/mvp-new1/key.csv" \
  --used "$H/mvp-retrieve1/key.csv" --used "$H/tile-heads-similar1/mvp/oof.csv" \
  --used "$H/mvp-final-test/patients.csv" --used "$H/mvp-outlines1/key.csv"
