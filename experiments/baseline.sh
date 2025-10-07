#!/usr/bin/env bash


variation="${1:?group required, e.g., cleaned}"
group="baseline"  # fixed for baselines

#! Edit these once
echo "Current conda environment: $CONDA_DEFAULT_ENV"
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
RUN_ID="${variation}-${group}"
LOG_DIR="${REPO}/experiments/logs/${variation}"
OUT_ADAPTER="${CACHE}/${RUN_ID}_adapter"
OUT_MERGED="${CACHE}/${RUN_ID}"

# --- W&B wiring ---
export WANDB_PROJECT="Thesis_Phase"
#export WANDB_ENTITY="your_entity"              # optional
export WANDB_DIR="${LOG_DIR}"                  # keeps artifacts and offline caches with your logs
export WANDB_RUN_GROUP="${variation}-${group}" # groups training + inference
export WANDB_NAME="${RUN_ID}"                  # training run name
export WANDB_TAGS="baseline,${variation},${group}"
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

# Train baseline
llamafactory-cli train "experiments/baseline.yaml" \
  --dataset="['${variation}_grade02_train', '${variation}_grade03_train', '${variation}_grade04_train', '${variation}_grade05_train', '${variation}_grade06_train', '${variation}_grade07_train', '${variation}_grade08_train', '${variation}_grade09_train', '${variation}_grade010_train', '${variation}_grade011_train', '${variation}_grade012_train']" \
  --eval_dataset "${variation}_${group}_validate" \
  --metric_for_best_model "eval_${variation}_${group}_validate_sari" \
  --output_dir "${OUT_ADAPTER}" \
  --run_name "${variation}-${group}" \
  > "${LOG_DIR}/train.log" 2>&1

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

for n in {2..12}; do
  echo "Infer baseline ${variation} on grade ${n}"
  WANDB_NAME="${RUN_ID}-infer-g${n}" \
  WANDB_JOB_TYPE="inference" \
  python3 scripts/vllm_infer_metrics.py \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_name_or_path "${OUT_ADAPTER}" \
    --save_path "${OUT_ADAPTER}" \
    --save_name "baseline_${variation}_g${n}.jsonl" \
    --template llama3 \
    --dataset "${variation}_grade${n}_validation" \
    --temperature 0 \
    --grade "${n}" \
    > "${LOG_DIR}/infer_g${n}.log" 2>&1
  echo "save_name: baseline_${variation}_g${n}.jsonl"
  echo "dataset: ${variation}_grade${n}_validation"
  echo "grade: ${n}"
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
