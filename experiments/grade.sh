#!/usr/bin/env bash


# ---------------- User knobs ----------------
# MODEL_VARIATION="${1:?model variation required: original|cleaned|augmented}"
MODEL_VARIATION="cleaned"              # fixed for baseline runs
PROJECT_VERSION="v0-3"                 # used in WANDB_PROJECT
BASE_GROUP="${1:?model grade required: 02|03|04|05|06|07|08|09|10|11|12}"                  # logical family for this run
ENTITY=""                              # optional W&B entity

# ---------------- Paths & env ----------------
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
RUN_KEY="${MODEL_VARIATION}-graded"
LOG_DIR="${REPO}/experiments/logs/graded"
CFG_DIR="${REPO}/experiments/configs"
OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_${MODEL_VARIATION}_grade${BASE_GROUP}-adapter"
mkdir -p "${OUT_ADAPTER}" "${LOG_DIR}" "${LOG_DIR}/logs" "${LOG_DIR}/generated_predictions"

# ---------------- Config choose: fresh vs resume ----------------
# if compgen -G "${OUT_ADAPTER}/checkpoint-*" > /dev/null; then
#   CFG="${CFG_DIR}/${MODEL_VARIATION}_${BASE_GROUP}.resume.yaml"
#   echo "[train] Resuming with ${CFG}"
# else
CFG="${CFG_DIR}/grade${BASE_GROUP}.yaml"
echo "[train] Fresh start with ${CFG}"
#fi

# ---------------- Stable W&B run id per train variant ----------------
ID_DIR="${HOME}/.llf_wandb_ids"
mkdir -p "${ID_DIR}"
WBRUN_FILE="${ID_DIR}/${RUN_KEY}.id"

if [[ -f "${WBRUN_FILE}" ]]; then
  export WANDB_RUN_ID="$(cat "${WBRUN_FILE}")"
else
  # short stable id
  export WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
  echo "${WANDB_RUN_ID}" > "${WBRUN_FILE}"
fi

# Persist for other scripts and future resumes
echo "${WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_parent_id.txt"
echo "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

EXPERIMENT_GROUP="exp-$(date +%Y%m%d-%H%M%S)"

# ---------------- Core W&B env ----------------
export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}"
[[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
export WANDB_DIR="${LOG_DIR}"
export WANDB_RESUME=allow
export WANDB_RUN_GROUP="graded"
export WANDB_NAME="model=graded-infer"
export WANDB_TAGS="${BASE_GROUP},${MODEL_VARIATION}"

export WANDB_ENABLE_SERVICE=true
export WANDB_HTTP_TIMEOUT=300
export WANDB_START_METHOD=thread

# --------------- System info ---------------
source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
cd "$REPO"

# if ! python -c "import bert_score" >/dev/null 2>&1; then
#   python -m pip install -U bert-score
# fi

echo "=== ENV ==="
echo "Conda: $CONDA_DEFAULT_ENV"; which python
nvidia-smi || true; nvcc --version || true

set -euo pipefail

# --------------- TRAIN ---------------
# echo "[train] will now run llamafactory-cli train ${CFG}"
# llamafactory-cli train "${CFG}" \
#   > "${LOG_DIR}/train_grade${BASE_GROUP}.log" 2>&1


# --------------- INFER (same run; tag infer dataset + grade) ---------------


echo "staring run at $(date)"
run_start_time=$(date +%s)

DATASET_VARIATION="${MODEL_VARIATION}" # original augmented)

echo "[infer] dataset variation: ${DATASET_VARIATION}"
variation_start_time=$(date +%s)
grade="${BASE_GROUP}"
grade_start_time=$(date +%s)
echo "[infer]   grade: ${grade}"

# Keep SAME run id as training; do NOT create per-grade runs
export WANDB_RUN_ID
export WANDB_JOB_TYPE="infer"

# Rich tags & notes for grouping/filtering in the UI
export WANDB_TAGS="${BASE_GROUP},${MODEL_VARIATION},ds:${DATASET_VARIATION},grade:${grade}"
export WANDB_NOTES="infer_ds=${DATASET_VARIATION}; grade=${grade}; train_variant=${MODEL_VARIATION}"

# If your inference script forwards env to W&B config, also export custom hints
export TRAIN_VARIANT="${MODEL_VARIATION}"
export INFER_VARIANT="${DATASET_VARIATION}"
export INFER_GRADE="${grade}"

# echo the specific inference arguments

# Call your inference (must use wandb.init(resume='allow') or respect env id)
echo "[wandb] using project=${WANDB_PROJECT} id=${WANDB_RUN_ID} resume=${WANDB_RESUME}"

python3 scripts/vllm_infer_metrics.py \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_name_or_path "${OUT_ADAPTER}" \
    --save_path "${LOG_DIR}" \
    --save_name "graded_${MODEL_VARIATION}_grade${grade}@${DATASET_VARIATION}" \
    --template llama3 \
    --dataset "${DATASET_VARIATION}_grade${grade}_validation" \
    --temperature 0 \
    --grade "${grade}" \
    > "${LOG_DIR}/logs/infer_grade${grade}.log" 2>&1 || true

echo "[infer] completed grade ${grade} into run ${WANDB_RUN_ID}"
grade_end_time=$(date +%s)
echo "[infer]  ${DATASET_VARIATION} grade ${grade} took $((grade_end_time - grade_start_time)) seconds"
end_time=$(date +%s)
echo "Total infer time: $((end_time - run_start_time)) seconds"

/scratch/wlacroix/.cache/llama_factory/v0-3_graded_02-adapter
