#!/bin/bash

# MAX_JOBS=1

# while true
# do
#   if (("$(squeue | grep 'stcllama my_name' -c)" 
#   < MAX_JOBS))
#   then
#     sbatch --export=PATH,LD_LIBRARY_PATH \
#       --cpus-per-task=16 --partition=a100 --gres=gpu:1 \
#       --job-name=stcllama \
#       run.sh || exit 1
#   fi
#   sleep 10
# done

filename="finetuning_repo/config.yaml"
source get_slurm_params.sh

#  --nodes=1 

sbatch --export=PATH,LD_LIBRARY_PATH \
      --cpus-per-task="$CPUS_PER_TASK" \
      --partition="$PARTITION" \
      --gres=gpu:"$GRES" \
      --time=0-08:00:00 \
      --job-name="$JOB_NAME" \
      stc/run.sh || exit 1

exec <&-