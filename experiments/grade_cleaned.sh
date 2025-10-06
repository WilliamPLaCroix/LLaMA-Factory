#!/usr/bin/env bash


grade="${1:?grade required, e.g., grade05}"
variation="cleaned"

# Edit these once
echo "Current conda environment: $CONDA_DEFAULT_ENV"
REPO="/nethome/wlacroix/LLaMA-Factory"
ENV_ACTIVATE="/nethome/wlacroix/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="/nethome/wlacroix/miniconda3/envs/llama_factory_v2"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"

RUN_ID="${variation}-${grade}"
LOG_DIR="${REPO}/experiments/logs/${variation}"
OUT_ADAPTER="${CACHE}/${RUN_ID}_adapter"
OUT_MERGED="${CACHE}/${RUN_ID}"
BASELINE_ADAPTER="${CACHE}/${variation}-baseline_adapter"

mkdir -p "${LOG_DIR}" # "$(dirname "${REPO}/experiments/logs/condor/dummy")" "${OUT_MERGED}"

# Guard
if [[ ! -d "${BASELINE_ADAPTER}" ]]; then
  echo "Missing baseline adapter for ${variation}: ${BASELINE_ADAPTER}" >&2
  # exit 2
fi

# Env
source "$ENV_ACTIVATE"
conda activate "$ENV_NAME"
echo "Activated conda environment: $CONDA_DEFAULT_ENV"
cd "$REPO"
echo "=== CUDA Debugging Information ==="
echo "HOST: $HOSTNAME"; nvidia-smi || true
nvcc --version
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================="
which python

set -euo pipefail

# Train grade from the chosen baseline adapter
echo "Starting ${variation} ${grade} workflow"

# llamafactory-cli train "experiments/grade_cleaned.yaml" \
#   --dataset "${variation}_${grade}_train" \
#   --eval_dataset "${variation}_${grade}_validation" \
#   --adapter_name_or_path "${BASELINE_ADAPTER}" \
#   --output_dir "${OUT_ADAPTER}" \
#   > "${LOG_DIR}/${grade}_train.log" 2>&1

# echo variables from llamafactory-cli train call

echo "Dummy call for debugging: $(date) on $(hostname)"
echo "training variables for ${variation} ${grade}:"
echo "dataset: ${variation}_${grade}_train"
echo "eval_dataset: ${variation}_${grade}_validation"
echo "adapter_name_or_path: ${BASELINE_ADAPTER}"
echo "output_dir: ${OUT_ADAPTER}"
echo "log: ${LOG_DIR}/${grade}_train.log"

# Merge
# llamafactory-cli export <(cat <<EOF
# model_name_or_path: ${BASE_MODEL}
# adapter_name_or_path: ${OUT_ADAPTER}
# template: llama3
# export_dir: ${OUT_MERGED}
# export_size: 2
# export_device: cpu
# export_legacy_format: false
# task: sft
# EOF
# ) > "${LOG_DIR}/${grade}_merge.log" 2>&1

# Inference on this grade’s test
gnum="${grade#grade}"   # "02".."12"

echo "Infer ${variation} on ${grade} test"

# python3 scripts/vllm_infer_metrics.py \
#   --model_name_or_path "${BASE_MODEL}" \
#   --adapter_name_or_path "${OUT_ADAPTER}" \
#   --save_path "${OUT_ADAPTER}" \
#   --save_name "${variation}_${grade}_test_preds.jsonl" \
#   --template llama3 \
#   --dataset "${variation}_${grade}_test" \
#   --temperature 0 \
#   --grade "${gnum}" \
#   > "${LOG_DIR}/${gnum}_infer.log" 2>&1

# echo variables from python3 scripts/vllm_infer_metrics.py call
echo "Dummy call for debugging: $(date) on $(hostname)"
echo "inference variables for ${variation} ${grade}:"
echo "model_name_or_path: ${BASE_MODEL}"
echo "adapter_name_or_path: ${OUT_ADAPTER}"
echo "save_path: ${OUT_ADAPTER}"
echo "save_name: ${variation}_${grade}_test_preds.jsonl"
echo "dataset: ${variation}_${grade}_test"
echo "temperature: 0"
echo "grade: ${gnum}"
echo "log: ${LOG_DIR}/${gnum}_infer.log"


echo "Done ${variation} ${grade}"
