#!/usr/bin/env bash

# ---------------- User knobs ----------------
# MODEL_VARIATION="${1:?model variation required: original|cleaned|augmented}"
MODEL_VARIATION="cleaned"              # fixed for baseline runs
PROJECT_VERSION="v0-3"                 # used in WANDB_PROJECT
#BASE_GROUP="${1:?model grade required: 02|03|04|05|06|07|08|09|10|11|12}"                  # logical family for this run
ENTITY=""                              # optional W&B entity

# ---------------- Paths & env ----------------
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
RUN_KEY="graded-infer"
LOG_DIR="${REPO}/experiments/logs/graded"
CFG_DIR="${REPO}/experiments/configs"

mkdir -p "${LOG_DIR}" "${LOG_DIR}/logs" "${LOG_DIR}/generated_predictions"

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

echo "Starting sequential grade processing at $(date)"
total_start_time=$(date +%s)

GRADES=(02 03 04 05 06 07 08 09 10 11 12)

for GRADE in "${GRADES[@]}"; do
    echo "----- Starting grade ${GRADE} -----"
    echo "staring run at $(date)"
    run_start_time=$(date +%s)

    OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_${MODEL_VARIATION}_grade${GRADE}-adapter"
    mkdir -p "${OUT_ADAPTER}"

    # ---------------- Separate W&B run IDs for train vs infer ----------------
    ID_DIR="${HOME}/.llf_wandb_ids"
    mkdir -p "${ID_DIR}"
    
    # Shared inference run ID (consistent across all grades)
    INFER_WBRUN_FILE="${ID_DIR}/${RUN_KEY}-infer.id"
    if [[ -f "${INFER_WBRUN_FILE}" ]]; then
      INFER_WANDB_RUN_ID="$(cat "${INFER_WBRUN_FILE}")"
    else
      INFER_WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
      echo "${INFER_WANDB_RUN_ID}" > "${INFER_WBRUN_FILE}"
    fi
    
    # Unique training run ID for each grade
    TRAIN_WBRUN_FILE="${ID_DIR}/${RUN_KEY}-train-grade${GRADE}.id"
    if [[ -f "${TRAIN_WBRUN_FILE}" ]]; then
      TRAIN_WANDB_RUN_ID="$(cat "${TRAIN_WBRUN_FILE}")"
    else
      TRAIN_WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
      echo "${TRAIN_WANDB_RUN_ID}" > "${TRAIN_WBRUN_FILE}"
    fi

    # Persist for other scripts and future resumes
    echo "${INFER_WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_infer_id.txt"
    echo "${TRAIN_WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_train_id.txt"
    echo "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

    # ---------------- Core W&B env ----------------
    export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}"
    [[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
    export WANDB_DIR="${LOG_DIR}"
    export WANDB_RESUME=allow

    # --------------- TRAIN ---------------
    # Set training-specific W&B config
    export WANDB_RUN_ID="${TRAIN_WANDB_RUN_ID}"
    export WANDB_RUN_GROUP="graded-train"
    export WANDB_NAME="model=graded-train-grade${GRADE}"
    export WANDB_TAGS="${GRADE},${MODEL_VARIATION},train"
    export WANDB_JOB_TYPE="train"

    export WANDB_ENABLE_SERVICE=true
    export WANDB_HTTP_TIMEOUT=300


    # --------------- TRAIN ---------------
    CFG="${CFG_DIR}/grade${GRADE}.yaml" ### TODO: don't forget to change yamls when running secondary training
    echo "[train] Fresh start with ${CFG}"
    echo "[train] will now run llamafactory-cli train ${CFG}"
    llamafactory-cli train "${CFG}" \
      > "${LOG_DIR}/train_grade${GRADE}.log" 2>&1


    # --------------- INFER (same run; tag infer dataset + grade) ---------------
    # Switch to shared inference W&B config
    export WANDB_RUN_ID="${INFER_WANDB_RUN_ID}"
    export WANDB_RUN_GROUP="graded"
    export WANDB_NAME="model=graded-infer"
    export WANDB_TAGS="${BASE_GROUP},${MODEL_VARIATION},ds:${DATASET_VARIATION},grade:${grade}"
    export WANDB_NOTES="infer_ds=${DATASET_VARIATION}; grade=${grade}; train_variant=${MODEL_VARIATION}"
    export WANDB_JOB_TYPE="infer"

    export TRAIN_VARIANT="${MODEL_VARIATION}"
    export INFER_VARIANT="${DATASET_VARIATION}"
    export INFER_GRADE="${GRADE}"

    # echo the specific inference arguments
    echo "[infer]   grade: ${GRADE}"
    grade_start_time=$(date +%s)
    DATASET_VARIATION="${MODEL_VARIATION}" # original augmented)
    echo "[infer] dataset variation: ${DATASET_VARIATION}"
    echo "[wandb] using project=${WANDB_PROJECT} id=${WANDB_RUN_ID} resume=${WANDB_RESUME}"

    # -------------- INFERENCE ECHO --------------
    echo the script call for debug
    echo "python3 scripts/vllm_infer_metrics.py "
    echo "    --model_name_or_path \'${BASE_MODEL}\' "
    echo "    --adapter_name_or_path \'${OUT_ADAPTER}\' "
    echo "    --save_path \'${LOG_DIR}\' "
    echo "    --save_name \'graded_${MODEL_VARIATION}_grade${GRADE}@${DATASET_VARIATION}\' "
    echo "    --template llama3 "
    echo "    --dataset \'${DATASET_VARIATION}_grade${GRADE}_validation\' "
    echo "    --temperature 0 "
    echo "    --grade \'${GRADE}\' "
    echo " > \'${LOG_DIR}/logs/infer_grade${GRADE}.log\' 2>&1"
    # -------------- INFERENCE CALL --------------
    python3 scripts/vllm_infer_metrics.py \
        --model_name_or_path "${BASE_MODEL}" \
        --adapter_name_or_path "${OUT_ADAPTER}" \
        --save_path "${LOG_DIR}" \
        --save_name "graded_${MODEL_VARIATION}_grade${GRADE}@${DATASET_VARIATION}" \
        --template llama3 \
        --dataset "${DATASET_VARIATION}_grade${GRADE}_validation" \
        --temperature 0 \
        --grade "${GRADE}" \
        > "${LOG_DIR}/logs/infer_grade${GRADE}.log" 2>&1
    # -------------- INFERENCE END --------------

    echo "[infer] completed grade ${GRADE} into run ${WANDB_RUN_ID}"
    grade_end_time=$(date +%s)
    echo "[infer]  ${DATASET_VARIATION} grade ${GRADE} took $((grade_end_time - grade_start_time)) seconds"
    end_time=$(date +%s)
    echo "Run time: $((end_time - run_start_time)) seconds"
done

total_end_time=$(date +%s)
echo "=== ALL GRADES COMPLETED ==="
echo "Total processing time: $((total_end_time - total_start_time)) seconds"