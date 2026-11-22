#!/usr/bin/env bash
# CPU: OCR and rule-label the reports, then measure coverage and draw the blinded slide-review sample.
#$ -N mvp_reports_extract
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=4G
#$ -l h_rt=12:00:00
#$ -j y
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
activate_env
H="$BASE/runs/highgrade"
S="$BASE/runs/triage-diverse"
OCR_ENV="$BASE/envs/entip-ocr"
TOOL="$REPO/training/runs/highgrade_mvp/report_labels.py"
PATH="$OCR_ENV/bin:$PATH" TESSDATA_PREFIX="$OCR_ENV/share/tessdata" OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1 \
  "$OCR_ENV/bin/python" "$TOOL" extract --out "$H/mvp-reports" --workers "${NSLOTS:-1}"
python "$TOOL" check --out "$H/mvp-reports" --manifest "$S/trial.csv" --bags "$S/bags-h-optimus-0" \
  --labelled "$H/mvp-recheck1/key.csv" "$H/mvp-recheck1/labels-AH-recheck1.csv" \
  --labelled "$H/mvp-new1/key.csv" "$H/mvp-new1/labels-AH-new1.csv" \
  --labelled "$H/mvp-retrieve1/key.csv" "$H/mvp-retrieve1/labels-AH-retrieve1.csv"
chmod -R go-rwx "$H/mvp-reports/private"
echo "done: $H/mvp-reports/report-check.json"
