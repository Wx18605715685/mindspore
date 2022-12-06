#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash examples/pretrain/pretrain_gpt.sh  DEVICE_ID EPOCH_SIZE DATA_DIR"
echo "for example: bash examples/pretrain/pretrain_gpt.sh 0 40 /path/zh-wiki/"
echo "=============================================================================================================="
export GLOG_v=3
export DEVICE_ID=$1
DATA_DIR=$2

python -m mindtransformer.models.gpt.gpt_stack_trainer \
    --stages=[3,6,12]  \
    --stage_epochs=[1,1,3]  \
    --train_data_path=$DATA_DIR \
    --optimizer="adam"  \
    --seq_length=1024 \
    --parallel_mode="stand_alone" \
    --checkpoint_prefix="gpt" \
    --global_batch_size=16 \
    --vocab_size=50257 \
    --hidden_size=768 \
    --num_heads=16 \
    --full_batch=False \
    --device_target="GPU" > standalone_train_gpu_log.txt 2>&1 &
