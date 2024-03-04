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
"""Blip2 Config API"""

from mindformers import BaseConfig, TransformerOpParallelConfig, CLIPVisionConfig, MindFormerRegister, \
    MindFormerModuleType
from mindformers.core.parallel_config import default_parallel_config
from mindformers.models.utils import convert_mstype
from qwen.qwen_config import QwenConfig


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class QwenVLConfig(BaseConfig):
    def __init__(self, vision_config: CLIPVisionConfig,
                 text_config: QwenConfig,
                 num_queries: int = 256,
                 proj_output_dim: int = 4096,
                 image_start_id: int = 151857,
                 image_pad_id: int = 151859,
                 freeze_vision: bool = False,
                 freeze_resampler: bool = False,
                 freeze_llm: bool = False,
                 checkpoint_name_or_path: str = None,
                 use_past: bool = False,
                 dtype: str = "float32",
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 param_init_type: str = "float16",
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 **kwargs):
        super().__init__(**kwargs)

        self.vision_config = vision_config
        self.text_config = text_config

        self.num_queries = num_queries
        self.proj_output_dim = proj_output_dim
        self.image_start_id = image_start_id
        self.image_pad_id = image_pad_id

        self.freeze_vision = freeze_vision
        self.freeze_resampler = freeze_resampler
        self.freeze_llm = freeze_llm
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.use_past = use_past

        self.parallel_config = parallel_config

        self.dtype = convert_mstype(dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.param_init_type = convert_mstype(param_init_type)

        self.text_config.parallel_config = parallel_config
        self.text_config.compute_dtype = self.compute_dtype
        self.text_config.layernorm_compute_type = self.layernorm_compute_type
        self.text_config.softmax_compute_type = self.softmax_compute_type
        self.text_config.param_init_type = self.param_init_type
        self.rotary_dtype = convert_mstype(text_config.rotary_dtype)

        self.pad_token_id = text_config.pad_token_id
        self.eos_token_id = text_config.eos_token_id
        self.ignore_token_id = text_config.ignore_token_id

        self.vocab_size = text_config.vocab_size
        self.seq_length = text_config.seq_length
        self.repetition_penalty = text_config.repetition_penalty
        self.max_decode_length = text_config.max_decode_length
        self.top_k = text_config.top_k
        self.top_p = text_config.top_p
        self.do_sample = text_config.do_sample
