#!/usr/bin/env bash
# CPU: read this run's reports, using OCR for scanned pages. Only the text is kept for necrosis labelling.
#$ -N necrosis_report_extract
#$ -S /bin/bash
#$ -pe sharedmem 8
#$ -l h_rss=4G
#$ -l h_rt=12:00:00
#$ -j y
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
OUT="$BASE/runs/highgrade/necrosis-report1/reports"
OCR_ENV="$BASE/envs/entip-ocr"
PATH="$OCR_ENV/bin:$PATH" TESSDATA_PREFIX="$OCR_ENV/share/tessdata" OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1 \
  "$OCR_ENV/bin/python" "$REPO/training/runs/highgrade_mvp/report_labels.py" extract --out "$OUT" \
  --workers "${NSLOTS:-1}"
# the shared reader also writes MVP labels: this run does not keep them
rm -f "$OUT/private/labels.csv" "$OUT/private/audit.csv"
chmod -R go-rwx "$OUT/private"
touch "$OUT/private/text-ready"
echo "done: $OUT/private/text"
