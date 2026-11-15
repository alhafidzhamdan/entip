#!/usr/bin/env bash
# Job 3 of 3, GPU: does the student do the teacher's job?
#  a) embedding agreement with the teacher on the held-out patients
#  b) re-embed every slide of the triage run with the student, retrain triage on those bags with the same plan,
#     and print the student's triage report next to the teacher's
#$ -N distill_check
#$ -l h_rt=24:00:00
#$ -j y
set -euo pipefail
. "$DTRIAL/eddie/config.sh"
activate_env
export HF_HUB_OFFLINE=1
cd "$DWORK"
S="$DWORK/student/student.pt"
entip-glioma distill evaluate --tiles tiles --student "$S" --out evaluation.json

todo=()
for d in "$SRC"/slides/*/; do
  uid="$(basename "$d")"
  [ -s "bags-student/$uid.npz" ] || todo+=("$d")
done
if [ "${#todo[@]}" -gt 0 ]; then
  entip-glioma embed extract "${todo[@]}" --out bags-student --encoder "student:$S" > embed-student.jsonl
fi
# shellcheck disable=SC2046
entip-glioma triage train --manifest "$SRC/trial.csv" --bags bags-student --out model-student \
  $(python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('train_args', [])))" "$PLAN") \
  > report-student.json
python "$DTRIAL/compare.py" "$SRC/report-h-optimus-0.json" report-student.json evaluation.json | tee comparison.txt
