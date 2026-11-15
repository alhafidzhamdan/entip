# Shared settings for the Eddie jobs. Override any of them in the environment before running submit.sh.
REPO="${REPO:-$HOME/entip}"                                          # clone of this repository
WORK="${WORK:-/exports/eddie/scratch/$USER/entip-triage-test-run}"  # slides, bags, model, logs (scratch is purged after a month)
ENV_DIR="${ENV_DIR:-/exports/eddie/scratch/$USER/envs/entip}"       # conda env made by setup.sh (home quota is too small for torch)
ENCODER="${ENCODER:-h-optimus-0}"                                    # handcrafted for a CPU-only dry run
TRIAL="$REPO/training/runs/triage-test-run"

export HF_HOME="$WORK/hf"                  # H-optimus-0 weights, fetched by the staging job
export PIP_CACHE_DIR="/exports/eddie/scratch/$USER/pip-cache"

activate_env() {
  . /etc/profile.d/modules.sh
  module load anaconda
  # shellcheck disable=SC1091
  source activate "$ENV_DIR"
}
