#!/usr/bin/env bash

# Train a single grade-specific adapter starting from a shared baseline adapter
# Usage: ./grade_robust.sh <variation> <grade>
#   variation: original|cleaned|augmented
#   grade: grade02 .. grade12

# ---------------- User knobs ----------------
VARIATION="${1:?variation required: original|cleaned|augmented}"
GRADE="${2:?grade required, e.g., grade05}"
PROJECT_VERSION="v0-2"                 # used in WANDB_PROJECT
BASE_GROUP="grade"                     # logical family for this run
ENTITY=""                              # optional W&B entity

# ---------------- Paths & env ----------------
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh || true
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
CFG_DIR="${REPO}/experiments"
RUN_KEY="${VARIATION}-${GRADE}-${BASE_GROUP}"
LOG_DIR="${REPO}/experiments/logs/${VARIATION}"
OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_${VARIATION}_${GRADE}-${BASE_GROUP}-adapter"
BASELINE_ADAPTER="${CACHE}/${PROJECT_VERSION}_${VARIATION}_baseline-adapter"

mkdir -p "${OUT_ADAPTER}" "${LOG_DIR}" "${LOG_DIR}/logs" "${LOG_DIR}/generated_predictions"

# ---------------- Config choose: fresh vs resume ----------------
if compgen -G "${OUT_ADAPTER}/checkpoint-*" > /dev/null; then
  CFG="${CFG_DIR}/${VARIATION}_${BASE_GROUP}.resume.yaml"
  echo "[train] Resuming with ${CFG}"
else
  CFG="${CFG_DIR}/${VARIATION}_${BASE_GROUP}.init.yaml"
  echo "[train] Fresh start with ${CFG}"
fi

# ---------------- Guards ----------------
if [[ ! -d "${BASELINE_ADAPTER}" ]]; then
  echo "[guard] Missing baseline adapter: ${BASELINE_ADAPTER}" >&2
  echo "[hint] Run baseline.sh for ${VARIATION} first, which creates the shared warm start adapter." >&2
  exit 2
fi

# ---------------- Stable W&B run id per grade variant ----------------
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
printf '%s\n' "${WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_parent_id.txt"
printf '%s\n' "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

# ---------------- Core W&B env ----------------
# Group all grades of a given variation under one timestamped group for easy comparison
EXPERIMENT_GROUP="${VARIATION}-grades-$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}"
[[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
export WANDB_DIR="${LOG_DIR}"
export WANDB_RESUME=allow
export WANDB_RUN_GROUP="${EXPERIMENT_GROUP}"
export WANDB_NAME="model=${VARIATION},grade=${GRADE}"
export WANDB_TAGS="${BASE_GROUP},${VARIATION},${GRADE}"

# --------------- System info ---------------
source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
cd "${REPO}"

# if ! python -c "import bert_score" >/dev/null 2>&1; then
#   python -m pip install -U bert-score
# fi

set -euo pipefail

echo "=== ENV ==="
echo "Conda: $CONDA_DEFAULT_ENV"; which python
nvidia-smi || true; nvcc --version || true

# --------------- TRAIN ---------------
# Warm start this grade adapter from the shared baseline adapter of the same variation
export WANDB_JOB_TYPE="train"
DATASET_TRAIN="${VARIATION}_${GRADE}_train"
DATASET_VAL="${VARIATION}_${GRADE}_validation"

# Un-comment to run for real
llamafactory-cli train "${CFG}" \
  --dataset "${DATASET_TRAIN}" \
  --eval_dataset "${DATASET_VAL}" \
  --adapter_name_or_path "${BASELINE_ADAPTER}" \
  --output_dir "${OUT_ADAPTER}" \
  > "${LOG_DIR}/${VARIATION}_${GRADE}_train.log" 2>&1

# Always echo the planned command for auditability
cat <<EOF
[train] planned call
llamafactory-cli train ${CFG} \
  --dataset ${DATASET_TRAIN} \
  --eval_dataset ${DATASET_VAL} \
  --adapter_name_or_path ${BASELINE_ADAPTER} \
  --output_dir ${OUT_ADAPTER} \
  > ${LOG_DIR}/${VARIATION}_${GRADE}_train.log 2>&1
EOF

# --------------- INFER (on this grade’s test set) ---------------
export WANDB_JOB_TYPE="infer"
GNUM="${GRADE#grade}"   # "02".."12"

SAVE_NAME="${VARIATION}_${GRADE}_test_preds.jsonl"
DATASET_TEST="${VARIATION}_${GRADE}_test"

# Un-comment to run for real
python3 scripts/vllm_infer_metrics.py \
  --model_name_or_path "${BASE_MODEL}" \
  --adapter_name_or_path "${OUT_ADAPTER}" \
  --save_path "${OUT_ADAPTER}" \
  --save_name "${SAVE_NAME}" \
  --template llama3 \
  --dataset "${DATASET_TEST}" \
  --temperature 0 \
  --grade "${GNUM}" \
  > "${LOG_DIR}/${VARIATION}_${GNUM}_infer.log" 2>&1

cat <<EOF
[infer] planned call
python3 scripts/vllm_infer_metrics.py \
  --model_name_or_path ${BASE_MODEL} \
  --adapter_name_or_path ${OUT_ADAPTER} \
  --save_path ${OUT_ADAPTER} \
  --save_name ${SAVE_NAME} \
  --template llama3 \
  --dataset ${DATASET_TEST} \
  --temperature 0 \
  --grade ${GNUM} \
  > ${LOG_DIR}/${VARIATION}_${GNUM}_infer.log 2>&1
EOF

echo "[done] ${VARIATION} ${GRADE}"
