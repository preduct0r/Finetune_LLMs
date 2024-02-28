#!/bin/bash
filename="finetuning_repo/config.yaml"
COUNT=0
while [ $COUNT -lt 6 ]
do
    read -r line
    if [ $COUNT -eq 2 ]; then
        export PARTITION=${line:10}
    fi
    
    if [ $COUNT -eq 3 ]; then
        export GRES=${line:5}
    fi

    if [ $COUNT -eq 4 ]; then
        export CPUS_PER_TASK=${line:15}
    fi

    if [ $COUNT -eq 5 ]; then
        export JOB_NAME=${line:9}
    fi

    COUNT=$((COUNT + 1))
done < "$filename"

