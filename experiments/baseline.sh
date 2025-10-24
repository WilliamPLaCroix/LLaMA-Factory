#!/usr/bin/env bash


# ---------------- User knobs ----------------
# MODEL_VARIATION="${1:?model variation required: original|cleaned|augmented}"
MODEL_VARIATION="official"  # fixed for baseline runs
PROJECT_VERSION="v0-3"                 # used in WANDB_PROJECT
BASE_GROUP="baseline"                  # logical family for this run
ENTITY=""                              # optional W&B entity

# ---------------- Paths & env ----------------
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct"
CACHE="/scratch/wlacroix/.cache/llama_factory"
RUN_KEY="${MODEL_VARIATION}-${BASE_GROUP}"
LOG_DIR="${REPO}/experiments/logs/${MODEL_VARIATION}"
CFG_DIR="${REPO}/experiments"
OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_${MODEL_VARIATION}_${BASE_GROUP}-adapter"
mkdir -p "${OUT_ADAPTER}" "${LOG_DIR}" "${LOG_DIR}/logs" "${LOG_DIR}/generated_predictions"

# ---------------- Config choose: fresh vs resume ----------------
# if compgen -G "${OUT_ADAPTER}/checkpoint-*" > /dev/null; then
#   CFG="${CFG_DIR}/${MODEL_VARIATION}_${BASE_GROUP}.resume.yaml"
#   echo "[train] Resuming with ${CFG}"
# else
#   CFG="${CFG_DIR}/${MODEL_VARIATION}_${BASE_GROUP}.init.yaml"
#   echo "[train] Fresh start with ${CFG}"
# fi

# ---------------- Stable W&B run id per train variant ----------------
ID_DIR="${HOME}/.llf_wandb_ids"
mkdir -p "${ID_DIR}"
WBRUN_FILE="${ID_DIR}/${RUN_KEY}.id"

if [[ -f "${WBRUN_FILE}" ]]; then
  export WANDB_RUN_ID="$(cat "${WBRUN_FILE}")"
else
  # short stable id
  export WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' 
' | cut -c1-12)"
  echo "${WANDB_RUN_ID}" > "${WBRUN_FILE}"
fi

Persist for other scripts and future resumes
printf '%s
' "${WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_parent_id.txt"
printf '%s
' "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

# An experiment group id to compare the trio {original,cleaned,augmented} together
EXPERIMENT_GROUP="exp-$(date +%Y%m%d-%H%M%S)"

# ---------------- Core W&B env ----------------
export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}"
[[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
export WANDB_DIR="${LOG_DIR}"
export WANDB_RESUME=allow
export WANDB_RUN_GROUP="${EXPERIMENT_GROUP}"          # shared across the 3 variants for this run of experiments
export WANDB_NAME="model=${MODEL_VARIATION}"           # stable name per train variant
export WANDB_TAGS="${BASE_GROUP},${MODEL_VARIATION}"

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
#   > "${LOG_DIR}/train.log" 2>&1


# --------------- INFER (same run; tag infer dataset + grade) ---------------
export WANDB_JOB_TYPE="infer"

echo "staring run at $(date)"
run_start_time=$(date +%s)
ds_variations=(cleaned) # original augmented)
for DATASET_VARIATION in "${ds_variations[@]}"; do
  echo "[infer] dataset variation: ${DATASET_VARIATION}"
  variation_start_time=$(date +%s)
  for grade in {02..12}; do
    grade_start_time=$(date +%s)
    echo "[infer]   grade: ${grade}"

    # Keep SAME run id as training; do NOT create per-grade runs
    export WANDB_RUN_ID
    export WANDB_RESUME=allow
    export WANDB_NAME="model=${MODEL_VARIATION}"   # keep stable name for color-by-run

    # Rich tags & notes for grouping/filtering in the UI
    export WANDB_TAGS="${BASE_GROUP},${MODEL_VARIATION},ds:${DATASET_VARIATION},grade:${grade}"
    export WANDB_NOTES="infer_ds=${DATASET_VARIATION}; grade=${grade}; train_variant=${MODEL_VARIATION}"

    # If your inference script forwards env to W&B config, also export custom hints
    export TRAIN_VARIANT="${MODEL_VARIATION}"
    export INFER_VARIANT="${DATASET_VARIATION}"
    export INFER_GRADE="${grade}"

    # echo the specific inference arguments


    # Call your inference (must use wandb.init(resume='allow') or respect env id)
    # --adapter_name_or_path "${OUT_ADAPTER}" \
    python3 scripts/vllm_infer_metrics.py \
      --model_name_or_path "${BASE_MODEL}" \
      --save_path "${LOG_DIR}" \
      --save_name "baseline_${MODEL_VARIATION}_g${grade}@${DATASET_VARIATION}" \
      --template llama3 \
      --dataset "${DATASET_VARIATION}_grade${grade}_validation" \
      --temperature 0 \
      --grade "${grade}" \
      > "${LOG_DIR}/logs/infer_g${grade}@${DATASET_VARIATION}.log" 2>&1 || true

    echo "[infer] completed grade ${grade} into run ${WANDB_RUN_ID}"
    grade_end_time=$(date +%s)
    echo "[infer]   grade ${grade} took $((grade_end_time - grade_start_time)) seconds"
  done
    variation_end_time=$(date +%s)
    echo "[infer] dataset variation ${DATASET_VARIATION} took $((variation_end_time - variation_start_time)) seconds"
done

end_time=$(date +%s)
echo "Total infer time: $((end_time - run_start_time)) seconds"
echo "[infer] completed all 3×3×11 calls into run ${WANDB_RUN_ID}"

echo "Done. Tips in W&B UI:
  • Group by group: ${EXPERIMENT_GROUP} to compare the three runs.
  • Color by run to keep train variants consistent.
  • Filter by tag ds:<dataset> or grade:<n> to slice inference results."
