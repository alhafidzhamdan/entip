#!/usr/bin/env bash
# Job 3 of 3, CPU: step 2 (triage) - cross-validated attention-MIL, calibration and the accept threshold.
# With 10 patients the gate accepts nothing (it needs >= 20 accepted patients): this checks the pipeline, not the model.
#$ -N triage10_train
#$ -pe sharedmem 4
#$ -l h_rt=02:00:00
#$ -l h_vmem=8G
#$ -j y
set -euo pipefail
. "$TRIAL/eddie/config.sh"
activate_env
cd "$WORK"

entip-train triage train --manifest trial.csv --bags "bags-$ENCODER" --out "model-$ENCODER" --no-loso \
  > "report-$ENCODER.json"
python - "report-$ENCODER.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
cv = r["cv"]["calibrated"]
print(json.dumps({"classes_trained": r["classes_trained"], "slides_missing_bags": r["slides_missing_bags"],
                  "cv_balanced_accuracy": cv["balanced_accuracy"], "cv_ece": cv["ece"], "gate": r["cv"]["gate"],
                  "source_check": r["source_check"]}, indent=2, default=float))
EOF
