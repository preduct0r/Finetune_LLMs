#!/bin/sh

# set params
PARTITION=gpu
NUM_GPUS=1
CPUS_PER_TASK=12
JOB_NAME=sft_llama



# debugging flags (optional)
export NCCL_DEBUG=INFO
date

cd ~/projects/LLM/Finetune_LLMs
echo "$(date '+%m/%d/%y %H:%M:%S')" > date.txt

set | grep "SLURM"

export CUDA_HOME=/mnt/hs/dorado6/mamaev-n/cuda-11.8
export PATH=$PATH:/mnt/hs/dorado6/mamaev-n/cuda-11.8/bin
export LD_LIBRARY_PATH=/mnt/hs/dorado6/mamaev-n/cuda-11.8/lib64

# setup conda
__conda_setup="$(CONDA_REPORT_ERRORS=false '/mnt/cs/home/kotov-d/anaconda3/condabin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then 
    \eval "$__conda_setup"
else
    if [ -f "/mnt/cs/home/kotov-d/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/cs/home/kotov-d/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate pl_template
    else
        \export PATH="/mnt/cs/home/kotov-d/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# launch
conda activate stc_llama

sbatch --export=PATH,LD_LIBRARY_PATH \
    --cpus-per-task=$CPUS_PER_TASK \
    --partition=$PARTITION --gres=gpu:$NUM_GPUS \
    --time=0-08:00:00 \
    --job-name=$JOB_NAME \
    -o /mnt/cs/nlu/home/kotov/LLM/SFT/slurm/output/$(date '+%Y_%m_%d_%H_%M_%S').txt \
    accelerate launch --config_file stc/accelerate.yaml --num_processes $NUM_GPUS finetuning_repo/trl_finetune.py \
    slurm.partition=$PARTITION \
    slurm.gres=$NUM_GPUS \
    slurm.job-name=$JOB_NAME \
    slurm.cpus-per-task=$CPUS_PER_TASK



