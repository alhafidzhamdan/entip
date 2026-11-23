#!/usr/bin/env bash
# Run on an Eddie login node.
#   bash necrosis_cohort_submit.sh pick   # choose the slides and report the download size; downloads no slides
#   bash necrosis_cohort_submit.sh run    # after checking cohort.json: download, embed on GPUs, then train
set -euo pipefail
MODE="${1:-}"
EMBED_SHARDS="${EMBED_SHARDS:-4}"
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"
RUN=necrosis-report2
WORK="${BASE:-/exports/igmm/eddie/Glioblastoma-WGS/entip}/runs/highgrade/$RUN"
. "$REPO/training/runs/triage/eddie/config.sh"
JOBS="$REPO/training/runs/highgrade_necrosis/eddie"
mkdir -p "$WORK/logs"
case "$MODE" in
  pick)
    [ ! -f "$WORK/trial.csv" ] || { echo "$WORK/trial.csv already exists: the selection is never redone" >&2; exit 1; }
    [ -f "$BASE/runs/highgrade/necrosis-report1/private/report-labels.csv" ] || {
      echo "necrosis-report1 has no report labels" >&2; exit 1; }
    JOB=$(qsub -terse -v REPO="$REPO" -o "$WORK/logs" "$JOBS/necrosis_cohort_pick.sh")
    echo "queued slide selection job $JOB; it downloads no slides"
    ;;
  run)
    [ -s "$WORK/trial.csv" ] && [ -s "$WORK/cohort.json" ] || { echo "run the pick step first" >&2; exit 1; }
    [ -s "$WORK/private/report-review.csv" ] || { echo "record the report review first" >&2; exit 1; }
    [ ! -f "$WORK/model/summary.json" ] || { echo "$WORK/model already has a result" >&2; exit 1; }
    pass="BASE=$BASE,REPO=$REPO,RUN=$RUN,WORK=$WORK,ENV_DIR=$ENV_DIR,ENCODER=$ENCODER,TRIAL=$TRIAL,HF_HOME=$HF_HOME"
    q() { qsub -terse -v "$pass" -o "$WORK/logs" "$@"; }
    cleanup() { qdel "$@" >/dev/null 2>&1 || true; echo "submission failed; nothing left queued" >&2; exit 1; }
    s=$(q -h "$JOBS/necrosis_cohort_stage.sh")
    # shellcheck disable=SC2086
    e=$(q -hold_jid "$s" -N necrosis_cohort_embed -t "1-$EMBED_SHARDS" $GPU_REQUEST "$TRIAL/eddie/2_embed.sh") || cleanup "$s"
    e="${e%%.*}"
    t=$(q -hold_jid "$e" "$JOBS/necrosis_cohort_train.sh") || cleanup "$s" "$e"
    qrls "$s" >/dev/null
    echo "queued download $s, embedding $e ($EMBED_SHARDS GPU tasks) and training $t"
    ;;
  *)
    echo "usage: necrosis_cohort_submit.sh pick|run" >&2
    exit 2
    ;;
esac
