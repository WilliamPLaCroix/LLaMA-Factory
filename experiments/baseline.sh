#!/usr/bin/env bash
# baseline.sh

# --- GPU renaming helper ---
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh

# --- Positional args ---
MODEL_VARIATION="${1:?model variation required, e.g. augmented}"
group="baseline"                     # fixed for baselines
run_version="v0-2"                   # used in WANDB_PROJECT

# --- Repo and paths ---
echo "Current conda environment: $CONDA_DEFAULT_ENV"
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
RUN_ID="${MODEL_VARIATION}-${group}"
LOG_DIR="${REPO}/experiments/logs/${MODEL_VARIATION}"
CFG_DIR="${REPO}/experiments"
OUT_ADAPTER="${CACHE}/${run_version}_${MODEL_VARIATION}_${group}-adapter"
mkdir -p "${OUT_ADAPTER}" "${LOG_DIR}"
echo "OUT_ADAPTER=${OUT_ADAPTER}"
ls -la "${OUT_ADAPTER}" || echo "(new dir)"

# --- Config selection based on whether checkpoints exist ---
if compgen -G "${OUT_ADAPTER}/checkpoint-*" > /dev/null; then
  CFG="${CFG_DIR}/${MODEL_VARIATION}_${group}.resume.yaml"
  echo "[train] Resuming with ${CFG}"
else
  CFG="${CFG_DIR}/${MODEL_VARIATION}_${group}.init.yaml"
  echo "[train] Fresh start with ${CFG}"
fi

# --- W&B parent run wiring (one parent per MODEL_VARIATION) ---
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
export WANDB_DIR="${LOG_DIR}"
export WANDB_RUN_GROUP="baseline/${MODEL_VARIATION}"    # parent group is just the model variation
export WANDB_NAME="train"                               # short, clean parent name
export WANDB_TAGS="baseline,model:${MODEL_VARIATION},group:${group}"

# --- Env ---
source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
echo "Activated conda environment: $CONDA_DEFAULT_ENV"
cd "$REPO"

# --- CUDA diag ---
echo "=== CUDA Debugging Information ==="
echo "HOST: $HOSTNAME"; nvidia-smi || true
nvcc --version
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================="
which python

set -euo pipefail

echo "Starting ${MODEL_VARIATION} ${group} workflow"
# ==========================================
# Train baseline, uncomment to retrain
# ==========================================
# llamafactory-cli train "${CFG}" \
#   > "${LOG_DIR}/train.log" 2>&1

# Record W&B parent for later tooling
# echo "$WANDB_RUN_ID"  > "${OUT_ADAPTER}/wandb_parent_id.txt"
# echo "$WANDB_PROJECT" > "${OUT_ADAPTER}/wandb_project.txt"

# echo variables from llamafactory-cli train call
# echo "Dummy call for debugging: $(date) on $(hostname)"
# echo "training variables for ${MODEL_VARIATION} ${group}:"
# echo "dataset: ${MODEL_VARIATION}_${group}_train"
# echo "eval_dataset: ${MODEL_VARIATION}_${group}_validation"
# echo "output_dir: ${OUT_ADAPTER}"
# echo "log: ${LOG_DIR}/${group}_train.log"

# ==========================================
# Inference: loop over dataset variations and grades
# ==========================================

# Configure dataset variations either via $2 or DATASET_VARIATIONS env
ds_variations=(original cleaned augmented)
echo "Dataset variations for inference: ${ds_variations[*]}"

echo "Dummy call for debugging: $(date) on $(hostname)"
echo "inference variables for ${MODEL_VARIATION} ${group}:"
echo "model_name_or_path: ${BASE_MODEL}"
echo "adapter_name_or_path: ${OUT_ADAPTER}"
echo "temperature: 0"
echo "template: llama3"

# Grades 02..12, zero padded
for DATASET_VARIATION in "${ds_variations[@]}"; do
  echo "== Inference on dataset variation: ${DATASET_VARIATION}"
  for grade in {02..12}; do
    echo "Infer ${MODEL_VARIATION} on ${DATASET_VARIATION}, grade ${grade}"
    grade_int=$((10#$grade))

    # Stable child run id per (DATASET_VARIATION, grade)
    IDFILE="${OUT_ADAPTER}/wandb_infer_${DATASET_VARIATION}_g${grade}.id"
    if [ -f "$IDFILE" ]; then
      export WANDB_RUN_ID="$(cat "$IDFILE")"
    else
      export WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
      echo "$WANDB_RUN_ID" > "$IDFILE"
    fi

    export WANDB_RESUME=allow
    export WANDB_JOB_TYPE="inference"
    export WANDB_RUN_GROUP="baseline/${MODEL_VARIATION}"
    export WANDB_NAME="baseline-${MODEL_VARIATION}-g${grade}@${DATASET_VARIATION}"
    export WANDB_TAGS="baseline,model:${MODEL_VARIATION},group:${group},dsvar:${DATASET_VARIATION},group2:${MODEL_VARIATION}-${DATASET_VARIATION},grade:${grade_int}"
    export WANDB_NOTES="DATASET_VARIATION=${DATASET_VARIATION}; grade=${grade_int}; adapter=${MODEL_VARIATION}"

    # Paths and names
    SAVE_DIR="${OUT_ADAPTER}/infer/${DATASET_VARIATION}/g${grade}"
    mkdir -p "${SAVE_DIR}"
    SAVE_NAME="baseline_model-${MODEL_VARIATION}_data-${DATASET_VARIATION}_g${grade}.jsonl"

    # Dataset naming pattern: "{variation}_grade{grade}_{split}", grades 2 digits, split in {train, validation, test}
    DATASET_NAME="${DATASET_VARIATION}_grade${grade}_validation"

    echo "Saving inference artifacts to: ${SAVE_DIR}"
    echo "Dataset: ${DATASET_NAME}"
    echo "Save name: ${SAVE_NAME}"

    # --- Actual inference call ---
    python3 scripts/vllm_infer_metrics.py \
      --model_name_or_path "${BASE_MODEL}" \
      --adapter_name_or_path "${OUT_ADAPTER}" \
      --save_path "${SAVE_DIR}" \
      --save_name "${SAVE_NAME}" \
      --template llama3 \
      --dataset "${DATASET_NAME}" \
      --temperature 0 \
      --grade "${grade_int}" \
      > "${LOG_DIR}/infer_model-${MODEL_VARIATION}_data-${DATASET_VARIATION}_g${grade}.log" 2>&1

  done
done

echo "Main ${MODEL_VARIATION} ${group} Workflow Completed!"
