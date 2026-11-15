#!/usr/bin/env bash
# Test training run of the AI glioma workflow on 10 public IDC slides: manifest -> pick 10 -> step 1 embed -> step 2 triage.
# Needs internet and: pip install -e platform/live-training -e "training/glioma[idc,slides]"  (add [foundation] for H-optimus-0)
# Usage (from anywhere):  training/runs/triage-trial-10/run_local.sh            # handcrafted test encoder, CPU
#                         ENCODER=h-optimus-0 training/runs/triage-trial-10/run_local.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/out"
ENCODER="${ENCODER:-handcrafted}"
mkdir -p "$OUT"
cd "$HERE"

# 1. labelled slide list from IDC (FFPE gliomas + GTEx normal brain, which is PAXgene-fixed). Cached.
[ -s "$OUT/manifest.csv" ] || entip-train triage manifest --prep ffpe paxgene_paraffin --out "$OUT/manifest.csv" \
  > "$OUT/manifest.json"

# 2. the 10 smallest slides: 5 glioma + 5 normal brain, one per patient (training needs >= 5 patients per class)
python pick_slides.py "$OUT/manifest.csv" "$OUT/trial.csv" --per-label glioma=5 normal_brain=5 | tee "$OUT/trial.txt"

# 3. step 1, pre-training: download each slide, embed it, delete it. Skips slides already done.
entip-train embed build-bags --manifest "$OUT/trial.csv" --out "$OUT/bags-$ENCODER" --encoder "$ENCODER" \
  | tee "$OUT/build-bags-$ENCODER.jsonl"

# 4. step 2, triage: cross-validated attention-MIL, calibration, accept threshold. With 10 patients the gate
#    accepts nothing (it needs >= 20 accepted patients); this run checks the pipeline, not the model.
entip-train triage train --manifest "$OUT/trial.csv" --bags "$OUT/bags-$ENCODER" --out "$OUT/model-$ENCODER" \
  --no-loso > "$OUT/report-$ENCODER.json"
python - "$OUT/report-$ENCODER.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
cv = r["cv"]["calibrated"]
print(json.dumps({"classes_trained": r["classes_trained"], "slides_missing_bags": r["slides_missing_bags"],
                  "cv_balanced_accuracy": cv["balanced_accuracy"], "cv_ece": cv["ece"], "gate": r["cv"]["gate"],
                  "source_check": r["source_check"]}, indent=2, default=float))
EOF
echo "outputs: $OUT"
