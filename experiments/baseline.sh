#!/usr/bin/env bash

source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
MODEL_VARIATION="${1:?group required, e.g., cleaned}"
group="baseline"  # fixed for baselines
run_version="v0-2"  # e.g., v0-2; used in WANDB_PROJECT


echo "Current conda environment: $CONDA_DEFAULT_ENV"
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
RUN_ID="${MODEL_VARIATION}-${group}"
LOG_DIR="${REPO}/experiments/logs/${MODEL_VARIATION}"
CFG_DIR="${REPO}/experiments"
OUT_ADAPTER="${CACHE}/${run_version}_${MODEL_VARIATION}_${group}-adapter"
mkdir -p "${OUT_ADAPTER}"
echo "OUT_ADAPTER=${OUT_ADAPTER}"
ls -la "${OUT_ADAPTER}" || echo "(new dir)"

if compgen -G "${OUT_ADAPTER}/checkpoint-*" > /dev/null; then
  CFG="${CFG_DIR}/${MODEL_VARIATION}_${group}.resume.yaml"
  echo "[train] Resuming with ${CFG}"
else
  CFG="${CFG_DIR}/${MODEL_VARIATION}_${group}.init.yaml"
  echo "[train] Fresh start with ${CFG}"
fi
# OUT_MERGED="${CACHE}/${RUN_ID}"

# --- W&B wiring ---
ID_DIR="${HOME}/.llf_wandb_ids"
mkdir -p "${ID_DIR}"
WBRUN_FILE="${ID_DIR}/${MODEL_VARIATION}-${group}.id"

if [ -f "${WBRUN_FILE}" ]; then
  export WANDB_RUN_ID="$(cat "${WBRUN_FILE}")"   # reuse same run
else
  export WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
  echo "${WANDB_RUN_ID}" > "${WBRUN_FILE}"
fi

grep -E "Resuming|Loaded state" -n "${LOG_DIR}/train*.log" || echo "No resume detected"
export WANDB_RESUME=allow

export WANDB_PROJECT="Thesis_Phase_${run_version}"
#export WANDB_ENTITY="your_entity"              # optional
export WANDB_DIR="${LOG_DIR}"                  # keeps artifacts and offline caches with your logs
export WANDB_RUN_GROUP="${MODEL_VARIATION}-${group}" # groups training + inference
export WANDB_NAME="${RUN_ID}"                  # training run name
export WANDB_TAGS="baseline,${MODEL_VARIATION},${group}"
export WANDB_RESUME=allow

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

echo "Starting ${MODEL_VARIATION} ${group} workflow"

# Train baseline
# llamafactory-cli train "${CFG}" \
# > "${LOG_DIR}/train.log" 2>&1

echo "$WANDB_RUN_ID" > "${OUT_ADAPTER}/wandb_parent_id.txt"
echo "$WANDB_PROJECT" > "${OUT_ADAPTER}/wandb_project.txt"

echo variables from llamafactory-cli train call
echo "Dummy call for debugging: $(date) on $(hostname)"
echo "training variables for ${MODEL_VARIATION} ${group}:"
echo "dataset: ${MODEL_VARIATION}_${group}_train"
echo "eval_dataset: ${MODEL_VARIATION}_${group}_validation"
echo "output_dir: ${OUT_ADAPTER}"
echo "log: ${LOG_DIR}/${group}_train.log"


# Inference on each grade test with the baseline adapter
# echo variables from python3 scripts/vllm_infer_metrics.py call
echo "Dummy call for debugging: $(date) on $(hostname)"
echo "inference variables for ${MODEL_VARIATION} ${group}:"
echo "model_name_or_path: ${BASE_MODEL}"
echo "adapter_name_or_path: ${OUT_ADAPTER}"
echo "save_path: ${OUT_ADAPTER}"
echo "temperature: 0"
echo "template: llama3"

ds_variations=(original cleaned augmented)
echo "Dataset variations for inference: ${ds_variations[*]}"

# 2 digit zero-padded sequence
for DATASET_VARIATION in "${ds_variations[@]}"; do
    for grade in {02..12}; do
    echo "Infer baseline ${MODEL_VARIATION} on grade ${grade} variation ${DATASET_VARIATION}"
    grade_int=$((10#$grade))

    export DATASET_VARIATION="${DATASET_VARIATION}"
    export WANDB_RUN_ID
    export WANDB_NAME="${RUN_ID}@${DATASET_VARIATION}"                  # training run name
    export WANDB_RESUME=allow
    export WANDB_JOB_TYPE="inference"

    # --- persist & reuse a unique run id for each grade ---
    IDFILE="${OUT_ADAPTER}/wandb_infer_g${grade}.id"
    if [ -f "$IDFILE" ]; then
        export WANDB_RUN_ID="$(cat "$IDFILE")"   # reuse same run id -> same W&B run
    else
        export WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
        echo "$WANDB_RUN_ID" > "$IDFILE"
    fi
    
    # --- resume so re-runs overwrite points with the same step ---
    export WANDB_TAGS="baseline,${MODEL_VARIATION},${group},dsvar:${DATASET_VARIATION},group2:${MODEL_VARIATION}-${DATASET_VARIATION},grade:${grade_int}"
    export WANDB_NOTES="dsvar=${DATASET_VARIATION}; grade=${grade_int}; adapter=${MODEL_VARIATION}"

    python3 scripts/vllm_infer_metrics.py \
        --model_name_or_path "${BASE_MODEL}" \
        --adapter_name_or_path "${OUT_ADAPTER}" \
        --save_path "${OUT_ADAPTER}" \
        --save_name "baseline_${MODEL_VARIATION}_g${grade}@${DATASET_VARIATION}.jsonl" \
        --template llama3 \
        --dataset "${MODEL_VARIATION}_grade${grade}_validation" \
        --temperature 0 \
        --grade "${grade_int}" \
        > "${LOG_DIR}/infer_g${grade}@${DATASET_VARIATION}.log" 2>&1
    echo "save_name: baseline_${MODEL_VARIATION}_g${grade}@${DATASET_VARIATION}.jsonl"
    echo "dataset: ${DATASET_VARIATION}_grade${grade}_validation"
    echo "grade: ${grade}@${DATASET_VARIATION}"
    done
done

echo "Main ${MODEL_VARIATION} ${group} Workflow Completed!"


# Optional: write a summary metrics.json by averaging SARI across grades
# python3 - <<'PY'
# import json, glob, numpy as np, os
# import os
# cache=os.environ["CACHE"]; MODEL_VARIATION=os.environ.get("MODEL_VARIATION","${MODEL_VARIATION}")
# root=f"{cache}/{MODEL_VARIATION}-baseline"
# vals=[]
# for p in glob.glob(f"{root}/metrics.json"):  # metrics.json already written by vllm_infer_metrics
#     with open(p) as f:
#         d=json.load(f); vals.append(float(d.get("sari", "nan")))
# if vals:
#     with open(f"{root}/baseline_summary.json","w") as f:
#         json.dump({"sari_mean": float(np.mean(vals))}, f)
# PY
