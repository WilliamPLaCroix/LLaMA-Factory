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
#llamafactory-cli train experiments/3_11.yaml \
#> experiments/logs/3_11_train.log 2>&1

echo "Begin Merge"
#llamafactory-cli export experiments/3_11_merge.yaml \
#python3 experiments/scripts/adapter_merging.py --adapters='["/scratch/wlacroix/.cache/llama_factory/2_adapter", "/scratch/wlacroix/.cache/llama_factory/3_adapter", "/scratch/wlacroix/.cache/llama_factory/4_adapter", "/scratch/wlacroix/.cache/llama_factory/5_adapter", "/scratch/wlacroix/.cache/llama_factory/6_adapter", "/scratch/wlacroix/.cache/llama_factory/7_adapter", "/scratch/wlacroix/.cache/llama_factory/8_adapter", "/scratch/wlacroix/.cache/llama_factory/9_adapter", "/scratch/wlacroix/.cache/llama_factory/10_adapter", "/scratch/wlacroix/.cache/llama_factory/11_adapter", "/scratch/wlacroix/.cache/llama_factory/12_adapter"]' --grades='[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]' --weights='[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]' --merge_method dare_ties --density 0.5 \
#> experiments/logs/2_through_12_dareties_merge.log 2>&1

echo "Begin Inference"
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/common_models/Llama-3.2-3B-Instruct" --adapter_name_or_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --save_path "/scratch/wlacroix/.cache/llama_factory/2_through_12_dareties" --template llama3 --dataset wikilarge_grade_12_test --temperature 0 --grade 12 \
> experiments/logs/2_through_12_dareties_infer.log 2>&1

#or if you encounter error:
#FORCE_TORCHRUN=1 PTA/experiments_sarubi/llama3_lora_sft.yaml \
#> PTA/experiments_sarubi/logs_lora_sft 2>&1

echo "Main Experiment Workflow Completed!"
