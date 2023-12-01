#!/usr/bin/env bash

model_dir=$1
rank_table_file=$2
num_p=$3
start_device_id=$4

mkdir -p logs

export RANK_TABLE_FILE=${rank_table_file}

for((i=0;i<$[num_p];i++))
do
    let j=i+start_device_id
    export DEVICE_ID=$j
    export RANK_ID=$i
    python mindformers/tools/export_parallel.py --model_dir ${model_dir} --device_id $j > logs/export_rank_${i}_dev_${j}.log 2>&1 &
done