# Shared settings for the Eddie scripts (setup.sh, submit.sh and the three jobs all source this file).
# Override any of them in the environment before running setup.sh / submit.sh.
#
# Nothing may write to the real home folder: its quota is small, and pip, conda, idc-index, Hugging Face, torch,
# triton and CUDA all write under ~ by default. So HOME itself is moved into the group space below, and every
# tool that writes under ~ lands there. The explicit locations after it keep the layout tidy.
BASE="${BASE:-/exports/igmm/eddie/Glioblastoma-WGS/entip}"
REPO="${REPO:-/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip}"   # clone of this repository
RUN="${RUN:-test-run}"                                       # which plan: plans/<RUN>.json (test-run, diverse)
TRIAL="$REPO/training/runs/triage"
PLAN="${PLAN:-$TRIAL/plans/$RUN.json}"
WORK="${WORK:-$BASE/runs/triage-$RUN}"                       # slides, bags, model, report, logs for this run
ENV_DIR="${ENV_DIR:-$BASE/envs/entip}"                       # conda env made by setup.sh
ENCODER="${ENCODER:-h-optimus-0}"                            # handcrafted for a CPU-only dry run
# How the embed job asks Eddie for a GPU. Check what the gpu queue offers with:
#   qconf -sq gpu | grep -E "pe_list|complex_values";  qconf -spl | grep -i gpu;  qconf -sc | grep -i gpu
# Eddie GPU wiki: gpu queue + gpu resource; a100=true picks an A100 (else A100 or L40S, whichever is free);
# memory is h_rss per slot (max 32G), times the sharedmem slots. No h_vmem.
GPU_REQUEST="${GPU_REQUEST:--q gpu -l gpu=1 -l a100=true -pe sharedmem 4 -l h_rss=16G}"   # 1 A100, 4 cores, 64 GB RAM
# The CPU training job: cores (sharedmem slots) and memory per core (h_rss, at most 32G). Total RAM = both multiplied.
TRAIN_REQUEST="${TRAIN_REQUEST:--pe sharedmem 8 -l h_rss=8G}"             # 8 cores, 64 GB RAM

export HF_HOME="${HF_HOME:-$BASE/hf}"                        # H-optimus-0 weights, fetched once by the staging job

# A Hugging Face login made from the normal shell (token in the real home) still works after HOME moves.
if [ -z "${HF_TOKEN:-}" ]; then
  for t in "$HF_HOME/token" "$HOME/.cache/huggingface/token"; do
    if [ -s "$t" ]; then export HF_TOKEN="$(cat "$t")"; break; fi
  done
fi

export ENTIP_REAL_HOME="${ENTIP_REAL_HOME:-$HOME}"
export HOME="$BASE/home"                                     # every "~" write from here on goes to the group space
export XDG_CACHE_HOME="$BASE/cache/xdg"
export XDG_DATA_HOME="$BASE/data"                            # e.g. idc-index's slide index
export XDG_STATE_HOME="$BASE/state"
export XDG_CONFIG_HOME="$BASE/config"
export CONDA_PKGS_DIRS="$BASE/cache/conda-pkgs"
export CONDA_ENVS_PATH="$BASE/envs"
export PIP_NO_CACHE_DIR=1                                    # no second copy of torch and CUDA wheels
export PIP_CACHE_DIR="$BASE/cache/pip"
export TORCH_HOME="$BASE/cache/torch"
export TRITON_CACHE_DIR="$BASE/cache/triton"
export CUDA_CACHE_PATH="$BASE/cache/nv"
export MPLCONFIGDIR="$BASE/cache/matplotlib"
export TMPDIR="${ENTIP_TMPDIR:-$BASE/tmp}"                   # pip's build space, joblib, tempfile
export PYTHONNOUSERSITE=1                                    # never read or install packages in ~/.local
unset PYTHONPATH PYTHONHOME                                  # only the env's own packages, never a shared Anaconda's

mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CONFIG_HOME" "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR" "$HF_HOME"

activate_env() {   # put the env's python, pip and entip-* commands first; no conda activate needed
  export PATH="$ENV_DIR/bin:$PATH"
  # Maths libraries (NumPy's BLAS, PyTorch) use only the cores this job was given, not every core on the node.
  local n="${NSLOTS:-1}"
  export OMP_NUM_THREADS="$n" OPENBLAS_NUM_THREADS="$n" MKL_NUM_THREADS="$n" NUMEXPR_NUM_THREADS="$n"
  command -v entip-train >/dev/null || { echo "entip-train not found in $ENV_DIR; run setup.sh" >&2; exit 1; }
}
