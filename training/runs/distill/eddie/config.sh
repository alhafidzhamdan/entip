# Settings for the distillation jobs. They reuse the triage run settings (group-space HOME, caches, env, GPU
# request) and read one triage run's slides and teacher bags.
. "$REPO/training/runs/triage/eddie/config.sh"       # sets BASE, RUN, WORK (the triage run), ENV_DIR, GPU_REQUEST, ...
SRC="$WORK"                                           # the triage run to distil from: slides/, bags-h-optimus-0/, trial.csv
DWORK="${DWORK:-$BASE/runs/distill-$RUN}"             # tile cache, student, evaluation, student bags, logs
DTRIAL="$REPO/training/runs/distill"
STUDENT_ARCH="${STUDENT_ARCH:-vit_small_patch16_224}"
EPOCHS="${EPOCHS:-20}"
MAX_TILES="${MAX_TILES:-1000}"                        # tiles per slide in the cache (150 KB each at 224 px)
