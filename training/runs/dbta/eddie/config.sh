# Settings for the DBTA external test. Reuses the triage settings (group-space HOME, caches, env, GPU request).
. "$REPO/training/runs/triage/eddie/config.sh"          # BASE, ENV_DIR, ENCODER, GPU_REQUEST, EMBED_SHARDS, ...
DTRIAL="$REPO/training/runs/dbta"
DBTA_DIR="${DBTA_DIR:-$BASE/data/dbta}"                  # where you downloaded the .ndpi files (EBRAINS, after the DUA)
DBTA_META="${DBTA_META:-$DBTA_DIR/metadata.csv}"         # the DBTA sample table (uuid, pat_id, diagnosis, ...)
TWORK="${TWORK:-$BASE/runs/dbta}"                        # manifest, slide links, bags, evaluation, logs
MODEL_RUN="${MODEL_RUN:-stage-a}"                        # which triage run's model to test
MODEL="${MODEL:-$BASE/runs/triage-$MODEL_RUN/model-$ENCODER}"
