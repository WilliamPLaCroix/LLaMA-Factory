#!/usr/bin/env bash

source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
variation="${1:?group required, e.g., cleaned}"
group="baseline"  # fixed for baselines

#! Edit these once
echo "Current conda environment: $CONDA_DEFAULT_ENV"
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
RUN_ID="${variation}-${group}"
LOG_DIR="${REPO}/experiments/logs/${variation}"
OUT_ADAPTER="${CACHE}/${variation}_${group}-adapter"
WBRUN_FILE="${OUT_ADAPTER}/wandb_run_id.txt"
OUT_MERGED="${CACHE}/${RUN_ID}"

# --- W&B wiring ---
export WANDB_PROJECT="Thesis_Phase"
#export WANDB_ENTITY="your_entity"              # optional
export WANDB_DIR="${LOG_DIR}"                  # keeps artifacts and offline caches with your logs
export WANDB_RUN_GROUP="${variation}-${group}" # groups training + inference
export WANDB_NAME="${RUN_ID}"                  # training run name
export WANDB_TAGS="baseline,${variation},${group}"
export WANDB_RESUME=allow
if [ -f "${WBRUN_FILE}" ]; then
  export WANDB_RUN_ID="$(cat "${WBRUN_FILE}")"
else
  # 8-char base36-ish ID; W&B accepts custom IDs
  export WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-8)"
  echo "${WANDB_RUN_ID}" > "${WBRUN_FILE}"
fi
# export WANDB_MODE=offline                    # uncomment if you need offline logging


mkdir -p "${LOG_DIR}" # "${OUT_MERGED}" "$(dirname "${REPO}/experiments/logs/condor/dummy")"

# Env

source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
echo "Activated conda environment: $CONDA_DEFAULT_ENV"
cd "$REPO"
echo "=== CUDA Debugging Information ==="
echo "HOST: $HOSTNAME"; nvidia-smi || true
nvcc --version
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================="
which python

# python3 -m pip install --upgrade --quiet wandb \
#   || python3 -m pip install --user --upgrade --quiet wandb

set -euo pipefail

echo "Starting ${variation} ${group} workflow"

head -n5 experiments/${variation}_baseline.yaml

# Train baseline
# llamafactory-cli train experiments/${variation}_baseline.yaml \
# > "${LOG_DIR}/train.log" 2>&1

echo "$WANDB_RUN_ID" > "${OUT_ADAPTER}/wandb_parent_id.txt"
echo "$WANDB_PROJECT" > "${OUT_ADAPTER}/wandb_project.txt"

echo variables from llamafactory-cli train call
echo "Dummy call for debugging: $(date) on $(hostname)"
echo "training variables for ${variation} ${group}:"
echo "dataset: ${variation}_${group}_train"
echo "eval_dataset: ${variation}_${group}_validation"
echo "output_dir: ${OUT_ADAPTER}"
echo "log: ${LOG_DIR}/${group}_train.log"

# Merge (optional, keeps parity with grades)
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
# ) > "${LOG_DIR}/merge.log" 2>&1

# Inference on each grade test with the baseline adapter
# echo variables from python3 scripts/vllm_infer_metrics.py call
echo "Dummy call for debugging: $(date) on $(hostname)"
echo "inference variables for ${variation} ${group}:"
echo "model_name_or_path: ${BASE_MODEL}"
echo "adapter_name_or_path: ${OUT_ADAPTER}"
echo "save_path: ${OUT_ADAPTER}"
echo "temperature: 0"
echo "template: llama3"

# 2 digit zero-padded sequence
for grade in {02..12}; do
  echo "Infer baseline ${variation} on grade ${grade}"
  unset WANDB_RUN_ID
  export WANDB_TAGS="baseline,${variation},${group},g${grade}"
  WANDB_RESUME=never
  WANDB_NAME="${RUN_ID}"
  WANDB_JOB_TYPE="inference" \
  python3 scripts/vllm_infer_metrics.py \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_name_or_path "${OUT_ADAPTER}" \
    --save_path "${OUT_ADAPTER}" \
    --save_name "baseline_${variation}_g${grade}.jsonl" \
    --template llama3 \
    --dataset "${variation}_grade${grade}_validation" \
    --temperature 0 \
    --grade "${grade}" \
    > "${LOG_DIR}/infer_g${grade}.log" 2>&1
  echo "save_name: baseline_${variation}_g${grade}.jsonl"
  echo "dataset: ${variation}_grade${grade}_validation"
  echo "grade: ${grade}"
done

echo "Main ${variation} ${group} Workflow Completed!"


# Optional: write a summary metrics.json by averaging SARI across grades
# python3 - <<'PY'
# import json, glob, numpy as np, os
# import os
# cache=os.environ["CACHE"]; variation=os.environ.get("variation","${variation}")
# root=f"{cache}/{variation}-baseline"
# vals=[]
# for p in glob.glob(f"{root}/metrics.json"):  # metrics.json already written by vllm_infer_metrics
#     with open(p) as f:
#         d=json.load(f); vals.append(float(d.get("sari", "nan")))
# if vals:
#     with open(f"{root}/baseline_summary.json","w") as f:
#         json.dump({"sari_mean": float(np.mean(vals))}, f)
# PY
