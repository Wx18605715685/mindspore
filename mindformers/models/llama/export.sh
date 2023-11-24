#!/bin/bash

if [[ $# -ne 4 && $# -ne 5  ]]; then
    echo "bash export.sh config_path output_path rank_table_file start_device_id (num_parallel)"
    exit 1
fi

config_path=$1
output_path=$2
rank_table_file=$3
start_device_id=$4

if [[ $# -eq 4  ]]; then
  num_p=4
  echo "export.sh $config_path $output_path $rank_table_file $start_device_id (4)"
else
  num_p=$5
  echo "export.sh $config_path $output_path $rank_table_file $start_device_id $num_p"
fi

mkdir -p logs
mkdir -p output_path

export RANK_TABLE_FILE=${rank_table_file}

for((i=0;i<$[num_p];i++))
do
    let j=i+start_device_id
    export DEVICE_ID=$j
    export RANK_ID=$i
    python export.py -c ${config_path} -d ${j} -r ${i} -o ${output_path} > logs/export_rank_${i}_dev_${j}.log 2>&1 &
done
