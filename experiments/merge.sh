#!/usr/bin/env bash

# ---------------- User knobs ----------------
# MODEL_VARIATION="${1:?model variation required: original|cleaned|augmented}"
MERGE_METHOD="${1:?merge method required: debug|svd|linear|ties|ties_svd|dare_ties|dare_linear|dare_ties_svd|dare_linear_svd|magnitude_prune|magnitude_prune_svd}"
ADAPTER_SELECTION="all"          # fixed for baseline runs
WEIGHT_METHOD="uniform"         # fixed for baseline runs

MODEL_VARIATION="cleaned"              # fixed for baseline runs
PROJECT_VERSION="v0-3"                 # used in WANDB_PROJECT  
ENTITY=""                              # optional W&B entity


# ---------------- Paths & env ----------------
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
RUN_KEY="${MERGE_METHOD}_a@${ADAPTER_SELECTION}_w@${WEIGHT_METHOD}-infer"
LOG_DIR="${REPO}/experiments/logs/merged"
CFG_DIR="${REPO}/experiments/configs"

mkdir -p "${LOG_DIR}" "${LOG_DIR}/generated_predictions"

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

echo "Starting adapter merging at $(date)"
total_start_time=$(date +%s)

# --------------- MERGE ADAPTERS --------------- 
echo "Begin Merger"
python3 experiments/scripts/adapter_merging.py \
--merge_method="${MERGE_METHOD}" \
--adapter_selection="${ADAPTER_SELECTION}" \
--weight_method="${WEIGHT_METHOD}" \
--project_version="${PROJECT_VERSION}" \
> experiments/logs/merged/${MERGE_METHOD}_a@${ADAPTER_SELECTION}_w@${WEIGHT_METHOD}_merge.log 2>&1
# --------------- MERGE END ---------------

GRADES=(02 03 04 05 06 07 08 09 10 11 12)

for GRADE in "${GRADES[@]}"; do
    echo "----- Starting grade ${GRADE} -----"
    echo "staring run at $(date)"
    run_start_time=$(date +%s)
    
    OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_merge_${MERGE_METHOD}_g@${ADAPTER_SELECTION}_w@${WEIGHT_METHOD}"
    mkdir -p "${OUT_ADAPTER}"

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

    # Persist for other scripts and future resumes
    echo "${INFER_WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_infer_id.txt"
    echo "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

    # ---------------- Core W&B env ----------------
    export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}"
    [[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
    export WANDB_DIR="${LOG_DIR}"
    export WANDB_RESUME=allow
    export WANDB_ENABLE_SERVICE=true
    export WANDB_HTTP_TIMEOUT=300

    # --------------- INFER (same run; tag infer dataset + grade) ---------------
    # Switch to shared inference W&B config
    export WANDB_RUN_ID="${INFER_WANDB_RUN_ID}"
    export WANDB_RUN_GROUP="merged"
    export WANDB_NAME="${MERGE_METHOD}_a@${ADAPTER_SELECTION}_w@${WEIGHT_METHOD}-infer"
    # echo the specific inference arguments
    echo "[infer]   grade: ${GRADE}"
    grade_start_time=$(date +%s)
    DATASET_VARIATION="${MODEL_VARIATION}" # original augmented)
    export TRAIN_VARIANT="${MODEL_VARIATION}"
    export WANDB_TAGS="${MODEL_VARIATION},ds:${DATASET_VARIATION},grade:${GRADE}"
    export WANDB_NOTES="infer_ds=${DATASET_VARIATION}; grade=${GRADE}; train_variant=${MODEL_VARIATION}"
    export WANDB_JOB_TYPE="infer"
    export INFER_GRADE="${GRADE}"

    echo "[infer] dataset variation: ${DATASET_VARIATION}"
    echo "[wandb] using project=${WANDB_PROJECT} id=${WANDB_RUN_ID} resume=${WANDB_RESUME}"

    # -------------- INFERENCE ECHO --------------
    echo the script call for debug
    echo "python3 scripts/vllm_infer_metrics.py "
    echo "    --model_name_or_path \'${BASE_MODEL}\' "
    echo "    --adapter_name_or_path \'${OUT_ADAPTER}\' "
    echo "    --save_path \'${LOG_DIR}\' "
    echo "    --save_name \'${MERGE_METHOD}_a@${ADAPTER_SELECTION}_w@${WEIGHT_METHOD}_grade${GRADE}-infer\' "
    echo "    --template llama3 "
    echo "    --dataset \'${DATASET_VARIATION}_grade${GRADE}_validation\' "
    echo "    --temperature 0 "
    echo "    --grade \'${GRADE}\' "
    echo " > \'${LOG_DIR}/generated_predictions/${MERGE_METHOD}_a@${ADAPTER_SELECTION}_w@${WEIGHT_METHOD}_infer_grade${GRADE}.log\' 2>&1"
    # -------------- INFERENCE CALL --------------
    python3 scripts/vllm_infer_metrics.py \
        --model_name_or_path "${BASE_MODEL}" \
        --adapter_name_or_path "${OUT_ADAPTER}" \
        --save_path "${LOG_DIR}" \
        --save_name "${MERGE_METHOD}_a@${ADAPTER_SELECTION}_w@${WEIGHT_METHOD}_grade${GRADE}-infer" \
        --template llama3 \
        --dataset "${DATASET_VARIATION}_grade${GRADE}_validation" \
        --temperature 0 \
        --grade "${GRADE}" \
        > "${LOG_DIR}/generated_predictions/${MERGE_METHOD}_a@${ADAPTER_SELECTION}_w@${WEIGHT_METHOD}_infer_grade${GRADE}.log" 2>&1
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