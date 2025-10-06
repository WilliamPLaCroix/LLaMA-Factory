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

echo "Begin Merge"
python3 experiments/scripts/adapter_merging.py --adapters='["/scratch/wlacroix/.cache/llama_factory/2_adapter", "/scratch/wlacroix/.cache/llama_factory/3_adapter", "/scratch/wlacroix/.cache/llama_factory/4_adapter", "/scratch/wlacroix/.cache/llama_factory/5_adapter", "/scratch/wlacroix/.cache/llama_factory/6_adapter", "/scratch/wlacroix/.cache/llama_factory/7_adapter", "/scratch/wlacroix/.cache/llama_factory/8_adapter", "/scratch/wlacroix/.cache/llama_factory/9_adapter", "/scratch/wlacroix/.cache/llama_factory/10_adapter", "/scratch/wlacroix/.cache/llama_factory/11_adapter", "/scratch/wlacroix/.cache/llama_factory/12_adapter"]' --grades='[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]' --weights='[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]' --merge_method dare_ties --density 0.5 \
> experiments/logs/2_through_12_dareties_merge.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_2_test --temperature 0 --grade 2 \
> experiments/logs/2_through_12_dareties_infer_2.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_3_test --temperature 0 --grade 3 \
> experiments/logs/2_through_12_dareties_infer_3.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_4_test --temperature 0 --grade 4 \
> experiments/logs/2_through_12_dareties_infer_4.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_5_test --temperature 0 --grade 5 \
> experiments/logs/2_through_12_dareties_infer_5.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_6_test --temperature 0 --grade 6 \
> experiments/logs/2_through_12_dareties_infer_6.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_7_test --temperature 0 --grade 7 \
> experiments/logs/2_through_12_dareties_infer_7.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_8_test --temperature 0 --grade 8 \
> experiments/logs/2_through_12_dareties_infer_8.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_9_test --temperature 0 --grade 9 \
> experiments/logs/2_through_12_dareties_infer_9.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_10_test --temperature 0 --grade 10 \
> experiments/logs/2_through_12_dareties_infer_10.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_11_test --temperature 0 --grade 11 \
> experiments/logs/2_through_12_dareties_infer_11.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_12_test --temperature 0 --grade 12 \
> experiments/logs/2_through_12_dareties_infer_12.log 2>&1


echo "Main Experiment Workflow Completed!"
