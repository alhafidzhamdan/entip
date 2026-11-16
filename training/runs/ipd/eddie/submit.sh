#!/usr/bin/env bash
# External test of a trained triage model on IPD-Brain (gliomas from NIMS Hyderabad, India):
#   MODEL_RUN=stage-a EMBED_SHARDS=4 bash training/runs/ipd/eddie/submit.sh
# Builds the manifest here (login node, seconds), checks the slides state their pixel size, then submits the GPU
# embedding and the evaluation after it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
ITRIAL="$HERE/.."
. "$HERE/config.sh"
[ -s "$IPD_META" ] || { echo "no IPD-Brain table at $IPD_META: download the dataset (India Data, after registration) and set IPD_META" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "no trained model at $MODEL: finish the triage run first, or set MODEL" >&2; exit 1; }
mkdir -p "$TWORK/logs"
"$ENV_DIR/bin/python" "$ITRIAL/make_manifest.py" --metadata "$IPD_META" --slides "$IPD_DIR" --out "$TWORK"
if [ -z "$IPD_MPP" ]; then                        # the TIFFs may not state their pixel size: check one
  first="$(ls "$TWORK"/slides/* | head -n 1)"
  stated="$("$ENV_DIR/bin/python" -c 'import sys, openslide; print(openslide.OpenSlide(sys.argv[1]).properties.get("openslide.mpp-x") or "")' "$first")"
  if [ -z "$stated" ]; then
    echo "$(basename "$first") does not state its pixel size: set IPD_MPP (the paper gives 0.22 µm/px at 40x) and resubmit" >&2
    exit 1
  fi
  echo "pixel size from the files: $stated µm/px"
fi
pass="BASE=$BASE,REPO=$REPO,RUN=$RUN,ENV_DIR=$ENV_DIR,ENCODER=$ENCODER,TRIAL=$TRIAL,HF_HOME=$HF_HOME,ITRIAL=$(cd "$ITRIAL" && pwd),IPD_DIR=$IPD_DIR,IPD_META=$IPD_META,IPD_MPP=$IPD_MPP,TWORK=$TWORK,MODEL_RUN=$MODEL_RUN,MODEL=$MODEL"
q() { qsub -terse -v "$pass" -o "$TWORK/logs" "$@"; }
shards=""; [ "$EMBED_SHARDS" -gt 1 ] && shards="-t 1-$EMBED_SHARDS"
# shellcheck disable=SC2086
e=$(q -N "ipd_embed" $shards $GPU_REQUEST "$HERE/2_embed.sh")
e="${e%%.*}"
v=$(q -hold_jid "$e" -N "ipd_evaluate" "$HERE/3_evaluate.sh") || { qdel "$e" >/dev/null 2>&1 || true; echo "submission failed; nothing left queued" >&2; exit 1; }
echo "submitted IPD-Brain test of $MODEL: embed $e (${EMBED_SHARDS} task(s)) -> evaluate $v"
echo "results: $TWORK/evaluation-$MODEL_RUN.json"
