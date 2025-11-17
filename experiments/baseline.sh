#!/usr/bin/env bash
# export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ---------------- User knobs ----------------
# MODEL_VARIATION="${1:?model variation required: original|cleaned|augmented}"
MODEL_VARIATION="cleaned"              # fixed for baseline runs
PROJECT_VERSION="v1"                 # used in WANDB_PROJECT
BASE_GROUP="baseline"                  # logical family for this run
ENTITY=""                              # optional W&B entity
#ITERATION_NUM="${1:?ITERATION number required}"  # Get the raw number
#ITERATION="-${ITERATION_NUM}"

# ---------------- Paths & env ----------------
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct-greedy"
CACHE="/scratch/wlacroix/.cache/llama_factory"

LOG_DIR="${REPO}/experiments/logs/${MODEL_VARIATION}"
CFG_DIR="${REPO}/experiments/configs"
MERGED_MODEL="${CACHE}/${PROJECT_VERSION}_${MODEL_VARIATION}_${BASE_GROUP}_merged"
OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_${MODEL_VARIATION}_${BASE_GROUP}-adapter"
mkdir -p "${OUT_ADAPTER}" "${LOG_DIR}" "${LOG_DIR}/logs" "${LOG_DIR}/generated_predictions"

# ---------------- Config choose: fresh vs resume ----------------
if compgen -G "${OUT_ADAPTER}/checkpoint-*" > /dev/null; then
CFG="${CFG_DIR}/${MODEL_VARIATION}_${BASE_GROUP}.resume.yaml"
echo "[train] Resuming with ${CFG}"
else
CFG="${CFG_DIR}/${MODEL_VARIATION}_${BASE_GROUP}.init.yaml"
echo "[train] Fresh start with ${CFG}"
fi

# An experiment group id to compare the trio {original,cleaned,augmented} together
EXPERIMENT_GROUP="exp-$(date +%Y%m%d-%H%M%S)"

# ---------------- Core W&B env ----------------
export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}"
[[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
export WANDB_DIR="${LOG_DIR}"
export WANDB_RESUME=allow
export WANDB_RUN_GROUP="${EXPERIMENT_GROUP}"          # shared across the 3 variants for this run of experiments

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

# for loop to iterate through evals by ITERATION
# for ITERATION_NUM in {97..98}; do
ITERATION_NUM=100

ITERATION="-${ITERATION_NUM}"
echo "Starting experiment for iteration: ${ITERATION_NUM}"
RUN_KEY="${MODEL_VARIATION}-${BASE_GROUP}-v1${ITERATION}"
export WANDB_NAME="${MODEL_VARIATION}-${BASE_GROUP}${ITERATION}"           # stable name per train variant

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

# -------------------- Persist for other scripts and future resumes
printf '%s
' "${WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_parent_id.txt"
printf '%s
' "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

# --------------- TRAIN ---------------
# echo "[train] will now run llamafactory-cli train ${CFG}"
# llamafactory-cli train "${CFG}" \
#   > "${LOG_DIR}/train.log" 2>&1

# --------------- MERGE ---------------
echo "Begin Merge"
llamafactory-cli export \
  --model_name_or_path /scratch/common_models/Llama-3.2-3B-Instruct-greedy \
  --adapter_name_or_path /scratch/wlacroix/.cache/llama_factory/v1_cleaned_baseline-adapter \
  --trust_remote_code true \
  --template llama3 \
  --export_dir ${MERGED_MODEL} \
  --export_legacy_format: false \
  --export_size 5 \
  --export_device cpu \
  > "${LOG_DIR}/merge_cleaned_baseline.log" 2>&1

# --------------- manual eval ---------------
echo "[train] will now run llamafactory-cli train ${CFG} eval only"
echo "starting manual eval"
export WANDB_JOB_TYPE="eval"
# --model_name_or_path /scratch/common_models/Llama-3.2-3B-Instruct-greedy \
# --adapter_name_or_path "${OUT_ADAPTER}/checkpoint-1768" \
llamafactory-cli train \
  --model_name_or_path "${MERGED_MODEL}" \
  --trust_remote_code True \
  --template llama3 \
  --do_train False \
  --do_eval True \
  --finetuning_type lora \
  --eval_dataset cleaned_baseline_validation \
  --output_dir "${LOG_DIR}" \
  --overwrite_output_dir True \
  --cutoff_len 1024 \
  --seed 42 \
  --per_device_eval_batch_size 32 \
  --bf16 True \
  --predict_with_generate False \
  --do_sample False \
  --report_to wandb \
  --run_name "${WANDB_NAME}" \
  > "${LOG_DIR}/cleaned_baseline_validation${ITERATION}_eval.log" 2>&1
echo "[eval] completed eval for iteration ${ITERATION} into run ${WANDB_RUN_ID}"

# --------------- INFER (same run; tag infer dataset + grade) ---------------
echo "starting vllm eval"
export WANDB_JOB_TYPE="infer"
    # --model_name_or_path "${BASE_MODEL}" \
    # --adapter_name_or_path "${OUT_ADAPTER}" \
python3 scripts/vllm_infer_metrics.py \
  --model_name_or_path "${MERGED_MODEL}" \
  --save_path "${LOG_DIR}" \
  --save_name "cleaned_baseline_validation${ITERATION}_infer" \
  --template llama3 \
  --dataset "cleaned_baseline_validation" \
  --seed "42" \
  > "${LOG_DIR}/cleaned_baseline_validation${ITERATION}.log" 2>&1


# # # ------------- loop eval for all checkpoints -------------] 
# # # for checkpoint in ${OUT_ADAPTER}/checkpoint-*; do
# # #   checkpoint_name="$(basename "${checkpoint}")"
# # #   out="${OUT_ADAPTER}/${checkpoint_name}"
# # #   echo "[eval] Processing checkpoint: ${checkpoint_name}"
# # #   echo "[eval] Checkpoint path: ${checkpoint}"
# # #   echo "[eval] Output directory: ${out}"
  
# # #   # Create a temporary config file for this checkpoint evaluation
# # #   temp_cfg="${CFG_DIR}/temp_eval_$(basename "${checkpoint}").yaml"
  
# # #   # Copy the base config and modify for evaluation
# # #   cp "${CFG}" "${temp_cfg}"
# # #   echo "[eval] Original config content (relevant lines):"
# # #   grep -E "(do_train|output_dir|adapter_name_or_path)" "${CFG}" || echo "No matching lines found in original config"
  
  
# # #   # Modify the config for evaluation using sed or yq
# # #   sed -i "s|do_train: True|do_train: False|g" "${temp_cfg}"
# # #   sed -i "s|output_dir: .*|output_dir: ${out}|g" "${temp_cfg}"
# # #   sed -i "s|^adapter_name_or_path:.*|adapter_name_or_path: ${checkpoint}|g" "${temp_cfg}"
    
# # #   echo "[eval] Modified config content (relevant lines):"
# # #   grep -E "(do_train|output_dir|adapter_name_or_path|resume_from_checkpoint)" "${temp_cfg}" || echo "No matching lines found in modified config"
  
# # #   echo "[eval] Full temporary config file contents:"
# # #   echo "--- START CONFIG ---"
# # #   cat "${temp_cfg}"
# # #   echo "--- END CONFIG ---"

# # #   llamafactory-cli train "${temp_cfg}" \
# # #     > "${LOG_DIR}/logs/eval_$(basename "${checkpoint}").log" 2>&1
  
# # #   # Clean up temporary file
# # #   rm "${temp_cfg}"
# # # done

# --------------- INFER (same run; tag infer dataset + grade) ---------------
# export WANDB_JOB_TYPE="infer"

# echo "staring run at $(date)"
# run_start_time=$(date +%s)
# DATASET_VARIATION="cleaned"

# for grade in {02..12}; do
#     grade_start_time=$(date +%s)
#     echo "[infer]   grade: ${grade}"

#     # Keep SAME run id as training; do NOT create per-grade runs
#     export WANDB_RUN_ID
#     export WANDB_RESUME=allow
#     export WANDB_NAME="${MODEL_VARIATION}-${BASE_GROUP}${ITERATION}"   # keep stable name for color-by-run

#     # Rich tags & notes for grouping/filtering in the UI
#     export WANDB_TAGS="${BASE_GROUP},${MODEL_VARIATION},ds:${DATASET_VARIATION},grade:${grade}"
#     export WANDB_NOTES="infer_ds=${DATASET_VARIATION}; grade=${grade}; train_variant=${MODEL_VARIATION}"

#     # If your inference script forwards env to W&B config, also export custom hints
#     export TRAIN_VARIANT="${MODEL_VARIATION}"
#     export INFER_VARIANT="${DATASET_VARIATION}"
#     export INFER_GRADE="${grade}"

#     # echo the specific inference arguments


#     # Call your inference (must use wandb.init(resume='allow') or respect env id)
#     #echo full command:
#     echo "python3 scripts/vllm_infer_metrics.py \
#         --model_name_or_path '${BASE_MODEL}' \
#         --adapter_name_or_path '${OUT_ADAPTER}' \
#         --save_path '${LOG_DIR}' \
#         --save_name 'baseline_${MODEL_VARIATION}_g${grade}@${DATASET_VARIATION}${ITERATION}' \
#         --template llama3 \
#         --dataset '${DATASET_VARIATION}_grade${grade}_validation' \
#         --grade '${grade}'"

#     python3 scripts/vllm_infer_metrics.py \
#         --model_name_or_path "${BASE_MODEL}" \
#         --adapter_name_or_path "${OUT_ADAPTER}" \
#         --save_path "${LOG_DIR}" \
#         --save_name "baseline_${MODEL_VARIATION}_g${grade}@${DATASET_VARIATION}${ITERATION}" \
#         --template llama3 \
#         --dataset "${DATASET_VARIATION}_grade${grade}_validation" \
#         --grade "${grade}" \
#         --seed "42" \
#         > "${LOG_DIR}/logs/infer_g${grade}@${DATASET_VARIATION}${ITERATION}.log" 2>&1 || true

#     echo "[infer] completed grade ${grade} into run ${WANDB_RUN_ID}"
#     grade_end_time=$(date +%s)
#     echo "[infer]   grade ${grade} took $((grade_end_time - grade_start_time)) seconds"
# done

# end_time=$(date +%s)
# echo "Total infer time: $((end_time - run_start_time)) seconds"
# echo "[infer] completed all 3×3×11 calls into run ${WANDB_RUN_ID}"

# echo "Done. Tips in W&B UI:
# • Group by group: ${EXPERIMENT_GROUP} to compare the three runs.
# • Color by run to keep train variants consistent.
# • Filter by tag ds:<dataset> or grade:<n> to slice inference results."
