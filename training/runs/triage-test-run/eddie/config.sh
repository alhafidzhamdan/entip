# Shared settings for the Eddie jobs. Override any of them in the environment before running setup.sh / submit.sh.
# Everything lives in the group space, which is kept (scratch is cleared after a month; home is too small).
BASE="${BASE:-/exports/igmm/eddie/Glioblastoma-WGS/entip}"
REPO="${REPO:-$HOME/entip}"                                  # clone of this repository
WORK="${WORK:-$BASE/runs/triage-test-run}"                   # slides, bags, model, report, logs
ENV_DIR="${ENV_DIR:-$BASE/envs/entip}"                       # conda env made by setup.sh
ENCODER="${ENCODER:-h-optimus-0}"                            # handcrafted for a CPU-only dry run
TRIAL="$REPO/training/runs/triage-test-run"

export HF_HOME="${HF_HOME:-$BASE/hf}"                        # H-optimus-0 weights, fetched once by the staging job
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"                # no second copy of torch and CUDA wheels
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BASE/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$BASE/cache/conda-pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/cache/xdg}"       # other tools' caches; nothing goes to the small home quota

activate_env() {
  set +u          # module and conda activate/deactivate scripts read unset variables
  . /etc/profile.d/modules.sh
  module load anaconda
  # shellcheck disable=SC1091
  source activate "$ENV_DIR"
  set -u
}
