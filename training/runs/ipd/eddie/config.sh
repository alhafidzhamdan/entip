# Settings for the IPD-Brain external test. Reuses the triage settings (group-space HOME, caches, env, GPU request).
. "$REPO/training/runs/triage/eddie/config.sh"          # BASE, ENV_DIR, ENCODER, GPU_REQUEST, EMBED_SHARDS, ...
ITRIAL="$REPO/training/runs/ipd"
IPD_DIR="${IPD_DIR:-$BASE/data/ipd}"                     # where you downloaded the 40x TIFFs (India Data, after registration)
IPD_META="${IPD_META:-$IPD_DIR/IPD_Brain.csv}"           # the clinical table (Case Number, Subtype, IDH1R132H, ...)
IPD_MPP="${IPD_MPP:-}"                                   # pixel size if the TIFFs do not state it (paper: 0.22 µm/px)
TWORK="${TWORK:-$BASE/runs/ipd}"                         # manifest, slide links, bags, evaluation, logs
MODEL_RUN="${MODEL_RUN:-stage-a}"                        # which triage run's model to test
MODEL="${MODEL:-$BASE/runs/triage-$MODEL_RUN/model-$ENCODER}"
