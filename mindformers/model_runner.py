# Copyright 2024 Huawei Technologies Co., Ltd
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

"""
For text generation
"""
import os
import numpy as np

import mindspore as ms
from mindspore.communication.management import init

from mindformers.core.context import build_context
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.models.utils import convert_mstype
from mindformers.models.build_config import build_model_config
from mindformers.core.parallel_config import build_parallel_config

from mindformers.models.auto import AutoModel
from mindformers.generation import GenerationConfig

__all__ = ["ModelRunner"]

class ModelRunner:
    """Model runner"""

    def __init__(self, model_path, npu_mem_size, cpu_mem_size, block_size, rank_id=0, world_size=1, ip=None, port=None):
        self.config = None
        self.model_config = None
        self.generation_config = None

        # parallel predict with dynamic cluster.
        if world_size > 1:
            os.environ['MS_WORKER_NUM'] = str(world_size)
            os.environ['MS_ROLE'] = 'MS_WORKER'
            if rank_id == 0 and os.fork() == 0:
                os.environ['MS_ROLE'] = 'MS_SCHED'
                init()

        if os.path.isdir(model_path):
            yaml_list = [file for file in os.listdir(model_path)
                         if file.endswith(".yaml")]
            if not yaml_list:
                raise FileNotFoundError(f"There is no yaml file for model config in {model_path}.")
            model_path = os.path.join(model_path, yaml_list[0])
            self.config = MindFormerConfig(model_path)
        else:
            raise ValueError(f"The path {model_path} is not exist.")

        if self.config and self.config.model.model_config:
            self.config.model.model_config.block_size = block_size
            self.model_config = build_model_config(self.config.model.model_config)

            compute_dtype = convert_mstype(self.model_config.compute_dtype)
            self.num_layers = self.model_config.num_layers
            n_kv_heads = self.model_config.num_heads if self.model_config.n_kv_heads is None else self.model_config.n_kv_heads
            head_dim = self.model_config.hidden_size // self.model_config.num_heads
            self.npu_num_blocks = (npu_mem_size * 1024 * 1024 * 1024) // \
                                  (block_size * n_kv_heads * head_dim * 2 * 2 * self.num_layers)
            self.cpu_num_blocks = (cpu_mem_size * 1024 * 1024 * 1024) // \
                                  (block_size * n_kv_heads * head_dim * 2 * 2 * self.num_layers)
            self.model_config.num_blocks = self.npu_num_blocks
            self.config.model.model_config.num_blocks = self.npu_num_blocks

            if self.config.use_parallel:
                build_parallel_config(self.config)
                self.config.model.model_config.checkpoint_name_or_path = None
                self.config.model.model_config.parallel_config = self.config.parallel_config

        self.generation_config = GenerationConfig.from_model_config(self.model_config)

        build_context(self.config)
        self.model = AutoModel.from_config(self.config)

        if self.config.use_parallel:
            _model = ms.Model(self.model)
            inputs = self.model.prepare_inputs_for_export(True)
            transform_and_load_checkpoint(self.config, _model, self.model, inputs, do_predict=True)

        if self.model_config.is_dynamic:
            self.model.set_dynamic_inputs()



    def forward(self, **kwargs):
        return self.model.infer(**kwargs)


    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

