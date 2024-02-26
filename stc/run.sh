#!/bin/bash
 
export CUDA_HOME=/mnt/hs/dorado6/mamaev-n/cuda-11.8
export PATH=$PATH:/mnt/hs/dorado6/mamaev-n/cuda-11.8/bin
export LD_LIBRARY_PATH=/mnt/hs/dorado6/mamaev-n/cuda-11.8/lib64
 
eval "$(conda shell.bash hook)"
conda activate stc_llama
 
cd ~/projects/LLM/Finetune_LLMs

export NUM_GPU=4

# accelerate launch --config_file stc/accelerate.yaml --num_processes $NUM_GPU finetuning_repo/trl_finetune.py --block_size 50 --eval_steps 100 --save_steps 100 -tf quotes_dataset/llama/train.csv -vf quotes_dataset/llama/validation.csv -m /mnt/hs/dorado6/mamaev-n/saiga_2/llama2-7b -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --disable_lora

# --config_file stc/accelerate.yaml 

accelerate launch finetuning_repo/trl_finetune.py --block_size 2048 --eval_steps 100 --save_steps 1000000000000000 -tf quotes_dataset/small_llama/train.csv -vf quotes_dataset/small_llama/train.csv -m /mnt/hs/dorado6/mamaev-n/saiga_2/llama2-7b -b 1 --log_steps 5 -lr 5e-6 -e 1 --gradient_accumulation_steps 1 --pad_token_id=18636 --disable_lora --disable_flash_attention
