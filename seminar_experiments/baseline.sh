#!/bin/bash

source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
echo "Current conda environment: $CONDA_DEFAULT_ENV"
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
echo "Activated conda environment: $CONDA_DEFAULT_ENV"
cd /nethome/wlacroix/LLaMA-Factory

# Debugging: Check CUDA details
echo "=== CUDA Debugging Information ==="
nvcc --version
nvidia-smi
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================="
echo "HOSTNAME: $HOSTNAME"
which python

# Main Experiment Script
echo "Starting Main Experiment Workflow!"

#echo "Begin Training"
#llamafactory-cli train experiments/baseline.yaml \
#> experiments/logs/baseline_train.log 2>&1

#echo "Begin Merge"
#llamafactory-cli export experiments/baseline_merge.yaml \
#> experiments/logs/baseline_merge.log 2>&1

echo "Begin Inferences"

echo "Begin Baseline Inference 2"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_2.jsonl --template llama3 --dataset wikilarge_grade_2_test --temperature 0 --grade 2 \
> experiments/logs/retrain_baseline_3_infer_2.log 2>&1

echo "Begin Baseline Inference 3"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_3.jsonl --template llama3 --dataset wikilarge_grade_3_test --temperature 0 --grade 3 \
> experiments/logs/retrain_baseline_3_infer_3.log 2>&1

echo "Begin Baseline Inference 4"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_4.jsonl --template llama3 --dataset wikilarge_grade_4_test --temperature 0 --grade 4 \
> experiments/logs/retrain_baseline_3_infer_4.log 2>&1

echo "Begin Baseline Inference 5"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_5.jsonl --template llama3 --dataset wikilarge_grade_5_test --temperature 0 --grade 5 \
> experiments/logs/retrain_baseline_3_infer_5.log 2>&1

echo "Begin Baseline Inference 6"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_6.jsonl --template llama3 --dataset wikilarge_grade_6_test --temperature 0 --grade 6 \
> experiments/logs/retrain_baseline_3_infer_6.log 2>&1

echo "Begin Baseline Inference 7"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_7.jsonl --template llama3 --dataset wikilarge_grade_7_test --temperature 0 --grade 7 \
> experiments/logs/retrain_baseline_3_infer_7.log 2>&1

echo "Begin Baseline Inference 8"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_8.jsonl --template llama3 --dataset wikilarge_grade_8_test --temperature 0 --grade 8 \
> experiments/logs/retrain_baseline_3_infer_8.log 2>&1

echo "Begin Baseline Inference 9"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_9.jsonl --template llama3 --dataset wikilarge_grade_9_test --temperature 0 --grade 9 \
> experiments/logs/retrain_baseline_3_infer_9.log 2>&1

echo "Begin Baseline Inference 10"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_10.jsonl --template llama3 --dataset wikilarge_grade_10_test --temperature 0 --grade 10 \
> experiments/logs/retrain_baseline_3_infer_10.log 2>&1

echo "Begin Baseline Inference 11"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_11.jsonl --template llama3 --dataset wikilarge_grade_11_test --temperature 0 --grade 11 \
> experiments/logs/retrain_baseline_3_infer_11.log 2>&1

echo "Begin Baseline Inference 12"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/baseline_adapter/checkpoint-5247" --save_path "/scratch/wlacroix/.cache/llama_factory/baseline" --save_name retrain_baseline_3_infer_12.jsonl --template llama3 --dataset wikilarge_grade_12_test --temperature 0 --grade 12 \
> experiments/logs/retrain_baseline_3_infer_12.log 2>&1

echo "Main Experiment Workflow Completed!"
