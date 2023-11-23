# Copyright 2023 Huawei Technologies Co., Ltd
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
"""export LLama2 model"""


import os
import argparse

import numpy as np
from mindformers import MindFormerConfig, LlamaConfig, LlamaForCausalLM, init_context, TransformerOpParallelConfig
import mindspore as ms
from mindspore import Tensor, export


def set_config(args):
    """setup MindFormerConfig"""
    config = MindFormerConfig(args.config_path)
    if args.device_id != -1:
        config.context.device_id = args.device_id
    config.model.model_config = LlamaConfig(**config.model.model_config)

    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    config.model.model_config.parallel_config = parallel_config
    config.model.model_config.checkpoint_name_or_path = config.load_checkpoint
    print(config)
    return config


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--rank_id', '-r', type=int, required=True)
    parser.add_argument('--output_path', '-o', type=str, required=True)

    args = parser.parse_args()
    print("-------------------------------------------------", args)
    return args


def dummy_tensor(shape, dtype):
    """create dummy tensor"""
    if None in shape:
        return Tensor(shape=shape, dtype=dtype)
    return Tensor(np.ones(shape=tuple(shape)), dtype=dtype)


def export_mindir():
    """export full and inc model on one device"""
    args = get_args()
    config = set_config(args)
    assert config.use_parallel
    os.makedirs(f"{args.output_path}", exist_ok=True)

    network = LlamaForCausalLM(config.model.model_config)
    network.set_train(False)
    network.phase = 'predict'
    print("...........Exporting Mindir.............", flush=True)

    bs = None
    seqlen = None
    activate_len_shape = None

    full_input_ids = dummy_tensor(shape=[bs, seqlen], dtype=ms.int32)
    full_input_position = dummy_tensor(shape=[bs], dtype=ms.int32)
    full_batch_valid_length = dummy_tensor(shape=[bs], dtype=ms.int64)
    full_batch_index = dummy_tensor(shape=[bs], dtype=ms.int64)
    full_activate_len = dummy_tensor(shape=[activate_len_shape], dtype=ms.int64)
    full_input_list = [full_input_ids, None, full_input_position, None, None, None, None, full_batch_valid_length,
                       full_batch_index, full_activate_len]

    inc_input_ids = dummy_tensor(shape=[bs, 1], dtype=ms.int32)
    inc_input_position = dummy_tensor(shape=[bs], dtype=ms.int32)
    inc_batch_valid_length = dummy_tensor(shape=[bs], dtype=ms.int64)
    inc_batch_index = dummy_tensor(shape=[bs], dtype=ms.int64)
    inc_activate_len = dummy_tensor(shape=[activate_len_shape], dtype=ms.int64)
    inc_input_list = [inc_input_ids, None, inc_input_position, None, None, None, None, inc_batch_valid_length,
                      inc_batch_index, inc_activate_len]

    def export_single(mode):
        save_path = os.path.join(args.output_path, f"rank_{args.rank_id}")
        save_path = os.path.join(save_path, f"{mode}")

        if mode == 'full':
            network.add_flags_recursive(is_first_iteration=True)
            input_list = full_input_list
        elif mode == 'inc':
            network.add_flags_recursive(is_first_iteration=False)
            input_list = inc_input_list
        else:
            assert False
        print(f"export {mode} model: {input_list}")
        export(network, *input_list, file_name=save_path, file_format='MINDIR')

    print("Start export full model...", flush=True)
    export_single(mode='full')
    print("Start export incr model...", flush=True)
    export_single(mode='inc')
    print("........Export mindir finished...........", flush=True)
    print(f"Mindir saved in \033[1;33m{args.output_path}\033[0m")


if __name__ == '__main__':
    export_mindir()
