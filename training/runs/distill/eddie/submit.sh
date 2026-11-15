#!/usr/bin/env bash
# Distil the teacher used in a finished triage run into a small student, on Eddie:
#   RUN=diverse bash training/runs/distill/eddie/submit.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
DTRIAL="$HERE/.."
. "$HERE/config.sh"
[ -s "$SRC/report-h-optimus-0.json" ] || { echo "no finished triage run in $SRC (RUN=$RUN); run that first" >&2; exit 1; }
mkdir -p "$DWORK/logs"
# Compute nodes have no internet: cache the student's ImageNet start weights now (login nodes have internet).
"$ENV_DIR/bin/python" -c "import timm; timm.create_model('$STUDENT_ARCH', pretrained=True)" >/dev/null
pass="BASE=$BASE,REPO=$REPO,RUN=$RUN,PLAN=$PLAN,WORK=$WORK,ENV_DIR=$ENV_DIR,TRIAL=$TRIAL,HF_HOME=$HF_HOME,DTRIAL=$(cd "$DTRIAL" && pwd),DWORK=$DWORK,STUDENT_ARCH=$STUDENT_ARCH,EPOCHS=$EPOCHS,MAX_TILES=$MAX_TILES"
q() { qsub -terse -v "$pass" -o "$DWORK/logs" "$@"; }
a=$(q -h -N "distill_${RUN}_tiles" "$HERE/1_tiles.sh")
cleanup() { qdel "$@" >/dev/null 2>&1 || true; echo "submission failed; nothing left queued" >&2; exit 1; }
# shellcheck disable=SC2086
b=$(q -hold_jid "$a" -N "distill_${RUN}_train" $GPU_REQUEST "$HERE/2_train.sh") || cleanup "$a"
# shellcheck disable=SC2086
c=$(q -hold_jid "$b" -N "distill_${RUN}_check" $GPU_REQUEST "$HERE/3_check.sh") || cleanup "$a" "$b"
qrls "$a" >/dev/null
echo "submitted distillation of run $RUN: tiles $a -> train $b -> check $c"
echo "watch: qstat;  logs: $DWORK/logs;  result: $DWORK/comparison.txt"
