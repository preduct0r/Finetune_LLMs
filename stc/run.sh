#!/bin/bash

source get_slurm_params.sh
 
export CUDA_HOME=/mnt/hs/dorado6/mamaev-n/cuda-11.8
export PATH=$PATH:/mnt/hs/dorado6/mamaev-n/cuda-11.8/bin
export LD_LIBRARY_PATH=/mnt/hs/dorado6/mamaev-n/cuda-11.8/lib64
 
eval "$(conda shell.bash hook)"
conda activate stc_llama

cd ~/projects/LLM/Finetune_LLMs_github

accelerate launch --config_file stc/accelerate.yaml --num_processes $NUM_GPUS finetuning_repo/trl_finetune.py --partition $PARTITION --num_gpus $NUM_GPUS

exec <&-