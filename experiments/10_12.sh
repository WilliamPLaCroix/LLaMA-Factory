#!/bin/bash

# rename gpus
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh

source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
#conda create -p /nethome/wlacroix/miniconda3/envs/llama_factory_v2 python=3.10
echo "Current conda environment: $CONDA_DEFAULT_ENV"
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

cd /nethome/wlacroix/LLaMA-Factory
#pip install -e ".[torch,metrics,deepspeed,vllm,bitsandbytes]"

#conda install -c nvidia cuda-compiler  ##https://github.com/deepspeedai/DeepSpeed/issues/2772

# run misc. stuff
# Debugging: Check CUDA details
echo "=== CUDA Debugging Information ==="
nvcc --version
nvidia-smi
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================="
echo "HOSTNAME: $HOSTNAME"
which python
#python -m pip list


# Main Experiment Script
echo "Starting Main Experiment Workflow!"
##Lora fine-tuning example already given: examples/train_lora/llama3_lora_sft.yaml
#Supervised Fine-Tuning cmd:
#llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
echo "Begin Training"
llamafactory-cli train experiments/10_12.yaml \
> experiments/logs/10_12_train.log  2>&1

echo "Begin Merge"
llamafactory-cli export experiments/10_12_merge.yaml \
> experiments/logs/10_12_merge.log  2>&1

echo "Begin Inference"
#export CUDA_LAUNCH_BLOCKING=1
python3 scripts/vllm_infer_metrics.py --model_name_or_path "/scratch/wlacroix/.cache/llama_factory/10_12" --save_path "/scratch/wlacroix/.cache/llama_factory/10_12" --template llama3 --dataset wikilarge_grade_11_test --temperature 0 \
> experiments/logs/10_12_infer.log  2>&1

#or if you encounter error:
#FORCE_TORCHRUN=1 PTA/experiments_sarubi/llama3_lora_sft.yaml \
#> PTA/experiments_sarubi/logs_lora_sft  2>&1

echo "Main Experiment Workflow Completed!"
