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


export PARTITION='gpu'
export NUM_GPUS="1"

#  --nodes=1 

sbatch --export=PATH,LD_LIBRARY_PATH \
      --cpus-per-task=12 --partition=$PARTITION --gres=gpu:$NUM_GPU \
      --time=0-08:00:00 --job-name=stcllama3 \
      stc/run.sh || exit 1