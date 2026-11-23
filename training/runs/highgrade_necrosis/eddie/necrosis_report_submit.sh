#!/usr/bin/env bash
# Run on an Eddie login node. Queues this run's report download and reading, then necrosis labelling and training.
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
OUT="$BASE/runs/highgrade/necrosis-report1"
JOBS="$REPO/training/runs/highgrade_necrosis/eddie"
OCR_ENV="$BASE/envs/entip-ocr"
[ ! -f "$OUT/summary.json" ] || { echo "$OUT already has a result: it is never overwritten" >&2; exit 1; }
if [ ! -x "$OCR_ENV/bin/tesseract" ]; then
  set +u
  . /etc/profile.d/modules.sh
  module load anaconda
  conda create -y -p "$OCR_ENV" -c conda-forge --override-channels python=3.11 tesseract
  set -u
  "$OCR_ENV/bin/python" -m pip install pypdfium2 pytesseract pillow
fi
mkdir -p "$OUT/logs" "$OUT/private" "$OUT/reports/private"
chmod 700 "$OUT/private" "$OUT/reports" "$OUT/reports/private"
HOLD=()
if [ ! -f "$OUT/reports/private/text-ready" ]; then
  FETCH=$(qsub -terse -v REPO="$REPO" -o "$OUT/logs" "$JOBS/necrosis_report_fetch.sh")
  READ=$(qsub -terse -hold_jid "$FETCH" -v REPO="$REPO" -o "$OUT/logs" "$JOBS/necrosis_report_extract.sh")
  HOLD=(-hold_jid "$READ")
  echo "queued report download $FETCH and reading $READ"
fi
JOB=$(qsub -terse ${HOLD[@]+"${HOLD[@]}"} -v REPO="$REPO" -o "$OUT/logs" "$JOBS/necrosis_report_mil.sh")
echo "queued necrosis labelling and training job $JOB"
