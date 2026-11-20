#!/usr/bin/env bash
# Submit the three jobs on Eddie, each waiting for the one before:
#   bash training/runs/triage/eddie/submit.sh                 # test run (plans/test-run.json)
#   RUN=diverse bash training/runs/triage/eddie/submit.sh     # plans/diverse.json
#   SKIP_TRAIN=1 RUN=astro-grade bash ...                    # stage and embed only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
. "$HERE/config.sh"
mkdir -p "$WORK/logs"
pass="BASE=$BASE,REPO=$REPO,RUN=$RUN,PLAN=$PLAN,WORK=$WORK,ENV_DIR=$ENV_DIR,ENCODER=$ENCODER,TRIAL=$TRIAL,HF_HOME=$HF_HOME"
q() { qsub -terse -v "$pass" -o "$WORK/logs" "$@"; }

# All or nothing: the stage job is held until the other two are accepted, so a rejected request leaves no orphans.
s=$(q -h -N "triage_${RUN}_stage" "$HERE/1_stage.sh")
cleanup() { qdel "$@" >/dev/null 2>&1 || true; echo "submission failed; nothing left queued" >&2; exit 1; }
# shellcheck disable=SC2086
shards=""; [ "$EMBED_SHARDS" -gt 1 ] && shards="-t 1-$EMBED_SHARDS"
# shellcheck disable=SC2086
e=$(q -hold_jid "$s" -N "triage_${RUN}_embed" $shards $GPU_REQUEST "$HERE/2_embed.sh") || cleanup "$s"
e="${e%%.*}"                                         # an array job id comes back as 12345.1-8:1
# shellcheck disable=SC2086
if [ "${SKIP_TRAIN:-0}" = 1 ]; then                    # stage and embed only (bags for another step)
  t="skipped"
else
  t=$(q -hold_jid "$e" -N "triage_${RUN}_train" $TRAIN_REQUEST "$HERE/3_train.sh") || cleanup "$s" "$e"
fi
qrls "$s" >/dev/null
echo "submitted run $RUN ($PLAN): stage $s -> embed $e (${EMBED_SHARDS} task(s), $GPU_REQUEST) -> train $t"
echo "watch: qstat;  logs: $WORK/logs;  results: $WORK/report-$ENCODER.json"
