#!/usr/bin/env bash
# External test of a trained triage model on the Digital Brain Tumour Atlas:
#   MODEL_RUN=stage-a EMBED_SHARDS=8 bash training/runs/dbta/eddie/submit.sh
# Builds the manifest here (login node, seconds), then submits the GPU embedding and the evaluation after it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
DTRIAL="$HERE/.."
. "$HERE/config.sh"
[ -s "$DBTA_META" ] || { echo "no DBTA metadata at $DBTA_META: download the dataset (EBRAINS, after the DUA) and set DBTA_META" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "no trained model at $MODEL: finish the triage run first, or set MODEL" >&2; exit 1; }
mkdir -p "$TWORK/logs"
"$ENV_DIR/bin/python" "$DTRIAL/make_manifest.py" --metadata "$DBTA_META" --slides "$DBTA_DIR" --out "$TWORK"
pass="BASE=$BASE,REPO=$REPO,RUN=$RUN,ENV_DIR=$ENV_DIR,ENCODER=$ENCODER,TRIAL=$TRIAL,HF_HOME=$HF_HOME,DTRIAL=$(cd "$DTRIAL" && pwd),DBTA_DIR=$DBTA_DIR,DBTA_META=$DBTA_META,TWORK=$TWORK,MODEL_RUN=$MODEL_RUN,MODEL=$MODEL"
q() { qsub -terse -v "$pass" -o "$TWORK/logs" "$@"; }
shards=""; [ "$EMBED_SHARDS" -gt 1 ] && shards="-t 1-$EMBED_SHARDS"
# shellcheck disable=SC2086
e=$(q -N "dbta_embed" $shards $GPU_REQUEST "$HERE/2_embed.sh")
e="${e%%.*}"
v=$(q -hold_jid "$e" -N "dbta_evaluate" "$HERE/3_evaluate.sh") || { qdel "$e" >/dev/null 2>&1 || true; echo "submission failed; nothing left queued" >&2; exit 1; }
echo "submitted DBTA test of $MODEL: embed $e (${EMBED_SHARDS} task(s)) -> evaluate $v"
echo "results: $TWORK/evaluation-$MODEL_RUN.json"
