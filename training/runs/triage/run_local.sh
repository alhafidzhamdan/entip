#!/usr/bin/env bash
# A triage training run on public IDC slides, on this machine: manifest -> IDH status -> pick the slides in the run
# plan -> step 1 embed -> step 2 triage. Needs internet and:
#   pip install -e platform/live-training -e "training/glioma[idc,slides]"   (add [foundation] for H-optimus-0)
# Usage (from anywhere):
#   training/runs/triage/run_local.sh                                  # plans/test-run.json, handcrafted encoder (CPU)
#   RUN=diverse ENCODER=h-optimus-0 training/runs/triage/run_local.sh   # needs a GPU or Apple MPS, ~190 GB of downloads
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${RUN:-test-run}"
PLAN="${PLAN:-$HERE/plans/$RUN.json}"
OUT="$HERE/out/$RUN"
ENCODER="${ENCODER:-handcrafted}"
mkdir -p "$OUT"
cd "$HERE"
plan_list() { python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get(sys.argv[2], [])))" "$PLAN" "$1"; }

# 1. labelled slide list from IDC, with the plan's options (cached), and TCGA IDH status (cached)
# shellcheck disable=SC2046
[ -s "$OUT/manifest.csv" ] || entip-train triage manifest $(plan_list manifest_args) --out "$OUT/manifest.csv" \
  > "$OUT/manifest.json"
[ -s "$OUT/idh.tsv" ] || python fetch_idh.py "$OUT/idh.tsv"

# 2. the slides in the plan
python pick_slides.py "$OUT/manifest.csv" "$OUT/trial.csv" --plan "$PLAN" --idh "$OUT/idh.tsv" | tee "$OUT/trial.txt"

# 3. step 1, pre-training: download each slide, embed it, delete it. Skips slides already done.
entip-train embed build-bags --manifest "$OUT/trial.csv" --out "$OUT/bags-$ENCODER" --encoder "$ENCODER" \
  | tee "$OUT/build-bags-$ENCODER.jsonl"

# 4. step 2, triage: cross-validated attention-MIL, calibration, accept threshold (options from the plan)
# shellcheck disable=SC2046
entip-train triage train --manifest "$OUT/trial.csv" --bags "$OUT/bags-$ENCODER" --out "$OUT/model-$ENCODER" \
  $(plan_list train_args) > "$OUT/report-$ENCODER.json"
python - "$OUT/report-$ENCODER.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
cv = r["cv"]["calibrated"]
print(json.dumps({"classes_trained": r["classes_trained"], "slides_missing_bags": r["slides_missing_bags"],
                  "slides_empty_bags": r.get("slides_empty_bags"), "cv_balanced_accuracy": cv["balanced_accuracy"],
                  "cv_ece": cv["ece"], "gate": r["cv"]["gate"], "source_check": r["source_check"],
                  "leave_one_source_out": r.get("leave_one_source_out")}, indent=2, default=float))
EOF
echo "outputs: $OUT"
