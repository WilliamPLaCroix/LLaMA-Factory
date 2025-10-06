#!/usr/bin/env bash
set -euo pipefail

variation="cleaned"
group="baseline"  # fixed for baselines

#! Edit these once
echo "Current conda environment: $CONDA_DEFAULT_ENV"
REPO="/nethome/wlacroix/LLaMA-Factory"
ENV_ACTIVATE="/nethome/wlacroix/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="/nethome/wlacroix/miniconda3/envs/llama_factory_v2"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"

RUN_ID="${variation}-${group}"
LOG_DIR="${REPO}/experiments/logs/${variation}/${group}"
OUT_ADAPTER="${CACHE}/${RUN_ID}_adapter"
OUT_MERGED="${CACHE}/${RUN_ID}"

mkdir -p "${LOG_DIR}" # "${OUT_MERGED}" "$(dirname "${REPO}/experiments/logs/condor/dummy")"

# Env
source "$ENV_ACTIVATE"
conda activate "$ENV_NAME"
echo "Activated conda environment: $CONDA_DEFAULT_ENV"
cd "$REPO"
echo "HOST: $HOSTNAME"; nvidia-smi || true
echo "=== CUDA Debugging Information ==="
nvcc --version
nvidia-smi
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================="
which python

echo "Starting ${variation} ${group} workflow"

# Train baseline
# llamafactory-cli train "experiments/baseline_cleaned.yaml" \
#   --dataset "${variation}_${group}_train" \
#   --eval_dataset "${variation}_${group}_validation" \
#   --output_dir "${OUT_ADAPTER}" \
#   > "${LOG_DIR}/train.log" 2>&1

# echo variables from llamafactory-cli train call
echo "Dummy call for debugging: $(date) on $(hostname)"
echo "training variables for ${variation} ${group}:"
echo "dataset: ${variation}_${group}_train"
echo "eval_dataset: ${variation}_${group}_validation"
echo "adapter_name_or_path: ${BASELINE_ADAPTER}"
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
# for n in {2..12}; do
#   echo "Infer baseline ${variation} on grade ${n}"
#   python3 scripts/vllm_infer_metrics.py \
#     --model_name_or_path "${BASE_MODEL}" \
#     --adapter_name_or_path "${OUT_ADAPTER}" \
#     --save_path "${OUT_ADAPTER}" \
#     --save_name "baseline_${variation}_g${n}.jsonl" \
#     --template llama3 \
#     --dataset "${variation}_grade${n}_validation" \
#     --temperature 0 \
#     --grade "${n}" \
#     > "${LOG_DIR}/infer_g${n}.log" 2>&1
# done

# echo variables from python3 scripts/vllm_infer_metrics.py call
echo "Dummy call for debugging: $(date) on $(hostname)"
echo "inference variables for ${variation} ${grade}:"
echo "model_name_or_path: ${BASE_MODEL}"
echo "adapter_name_or_path: ${OUT_ADAPTER}"
echo "save_path: ${OUT_ADAPTER}"
echo "save_name: baseline_${variation}_g${n}.jsonl"
echo "template: llama3"
echo "dataset: ${variation}_grade${n}_validation"
echo "temperature: 0"
echo "grade: ${n}"

# Optional: write a summary metrics.json by averaging SARI across grades
# python3 - <<'PY'
# import json, glob, numpy as np, os
# import os
# cache=os.environ["CACHE"]; variation=os.environ.get("variation","cleaned")
# root=f"{cache}/{variation}-baseline"
# vals=[]
# for p in glob.glob(f"{root}/metrics.json"):  # metrics.json already written by vllm_infer_metrics
#     with open(p) as f:
#         d=json.load(f); vals.append(float(d.get("sari", "nan")))
# if vals:
#     with open(f"{root}/baseline_summary.json","w") as f:
#         json.dump({"sari_mean": float(np.mean(vals))}, f)
# PY
