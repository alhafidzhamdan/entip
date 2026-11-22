#!/usr/bin/env bash
# Run on an Eddie login node. Makes the small OCR environment once, then queues the report download and the labelling.
set -euo pipefail
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
. "$REPO/training/runs/triage/eddie/config.sh"
OCR_ENV="$BASE/envs/entip-ocr"
OUT="$BASE/runs/highgrade/mvp-reports"
if [ ! -x "$OCR_ENV/bin/tesseract" ]; then
  set +u
  . /etc/profile.d/modules.sh
  module load anaconda
  conda create -y -p "$OCR_ENV" -c conda-forge --override-channels python=3.11 tesseract
  set -u
  "$OCR_ENV/bin/python" -m pip install pypdfium2 pytesseract pillow
fi
PATH="$OCR_ENV/bin:$PATH" TESSDATA_PREFIX="$OCR_ENV/share/tessdata" "$OCR_ENV/bin/python" -c "
from PIL import Image, ImageDraw, ImageFont
import pypdfium2, pytesseract
im = Image.new('L', (1400, 180), 255)
ImageDraw.Draw(im).text((30, 50), 'microvascular proliferation', fill=0, font=ImageFont.load_default(size=72))
text = pytesseract.image_to_string(im).lower()
assert 'microvascular' in text, text
print('OCR check passed')"
mkdir -p "$OUT/logs" "$OUT/private"
chmod 700 "$OUT/private"
FETCH=$(qsub -terse -v REPO="$REPO" -o "$OUT/logs" "$REPO/training/runs/highgrade_common/eddie/mvp_reports_fetch.sh")
READ=$(qsub -terse -hold_jid "$FETCH" -v REPO="$REPO" -o "$OUT/logs" "$REPO/training/runs/highgrade_common/eddie/mvp_reports_extract.sh")
echo "queued download job $FETCH and labelling job $READ"
