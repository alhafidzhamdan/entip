#!/usr/bin/env bash
# Job 3 of 3, CPU: step 2 (triage) - cross-validated attention-MIL, calibration and the accept threshold.
# Training options (e.g. --no-loso) come from the run plan.
#$ -N triage_train
#$ -pe sharedmem 4
#$ -l h_rt=12:00:00
#$ -l h_rss=8G
#$ -j y
set -euo pipefail
. "$TRIAL/eddie/config.sh"
activate_env
cd "$WORK"

# shellcheck disable=SC2046
entip-train triage train --manifest trial.csv --bags "bags-$ENCODER" --out "model-$ENCODER" \
  $(python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('train_args', [])))" "$PLAN") \
  > "report-$ENCODER.json"
python - "report-$ENCODER.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
cv = r["cv"]["calibrated"]
print(json.dumps({"classes_trained": r["classes_trained"], "slides_missing_bags": r["slides_missing_bags"], "slides_empty_bags": r.get("slides_empty_bags"),
                  "cv_balanced_accuracy": cv["balanced_accuracy"], "cv_ece": cv["ece"], "gate": r["cv"]["gate"],
                  "source_check": r["source_check"], "leave_one_source_out": r.get("leave_one_source_out")},
                 indent=2, default=float))
EOF
python "$TRIAL/mistakes.py" "report-$ENCODER.json"
