#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup_env_and_run_mmlu.sh
# -----------------------------------------------------------------------------
# Creates a dedicated Conda environment, installs all dependencies for PackLLM,
# then launches the common-fusion experiment on the MMLU benchmark using
# NousResearch/Llama-2-7b-hf instead of Meta's model.
# -----------------------------------------------------------------------------
# Usage
#   bash setup_env_and_run_mmlu.sh [conda_env_name]
#   Default conda_env_name = packllm_env
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${1:-packllm_env}"
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
DS_REQUIREMENTS_FILE="${PROJECT_ROOT}/downstream_tasks/requirements.txt"

# 1. ---------------------------------------------------------------------------
#    Create & activate Conda env if it does not exist
# -----------------------------------------------------------------------------
if ! conda info --envs | awk '{print $1}' | grep -q "^${ENV_NAME}$" ; then
  echo "[+] Creating conda environment '${ENV_NAME}' (python=3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi

# Ensure the script can run both interactively & non-interactively
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[+] Installing Python dependencies"
python -m pip install --upgrade pip
# Top-level requirements (if any)
if [[ -f "$REQUIREMENTS_FILE" ]]; then
  python -m pip install -r "$REQUIREMENTS_FILE"
fi
# Downstream-tasks requirements
if [[ -f "$DS_REQUIREMENTS_FILE" ]]; then
  python -m pip install -r "$DS_REQUIREMENTS_FILE"
fi
# Extras that may not be listed explicitly
python -m pip install --upgrade "transformers>=4.41.0" accelerate datasets evaluate scikit-learn sentencepiece

echo "[+] Dependency installation complete"

# 2. ---------------------------------------------------------------------------
#    Run the fusion experiment for the MMLU benchmark
# -----------------------------------------------------------------------------
MODEL_PATHS=(
  "NousResearch/Llama-2-7b-hf"
  "mistralai/Mistral-7B-v0.1"
  "microsoft/phi-2"
  "Deci/DeciLM-7B"
)

join_by_comma() {
  local IFS=","; echo "$*";
}
MODEL_NAMES="$( join_by_comma "${MODEL_PATHS[@]}" )"

FUSION="opt"         # can be "sim", "ensemble", "top1" etc.
TASK="mmlu"          # common STEM/commonsense category
OUTPUT_DIR="${PROJECT_ROOT}/outputs/fusion_${FUSION}_${TASK}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/fusion_${FUSION}.txt"

echo "[+] Starting experiment → task=${TASK}, fusion=${FUSION}"
python "${PROJECT_ROOT}/downstream_tasks/main.py" \
  --task_name "$TASK" \
  --fusion "$FUSION" \
  --model_name "$MODEL_NAMES" \
  --few_shot 0 \
  --data_cache_dir "${PROJECT_ROOT}/datasets" \
  --output_dir "$OUTPUT_DIR" \
  --annotation_size 5 \
  --seed 1  | tee -a "$LOG_FILE"

echo "[✓] Experiment finished. Summary saved to $LOG_FILE"
