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

"""KVCacheMgr custom layers"""
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore import nn, ops, Parameter
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
# from mindspore.context import ParallelMode
# from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
# from mindformers.tools.logger import logger as mindformer_logger
# from mindformers.modules import AttentionMask
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
# from mindformers.modules.transformer.moe import default_moe_config
# from mindformers.modules.transformer import MultiHeadAttention,\
#                                             TransformerEncoderLayer, TransformerEncoder
# from mindformers.modules.transformer.transformer import default_transformer_config, _get_lambda_func

class KVCacheMgrOp(nn.Cell):
    """The Embedding Layer of Bloom network."""
    def __init__(self,
                 hidden_size,
                 num_heads,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 parallel_config=default_dpmp_config,
                 max_cache_length=1024*32,
                 act_len=False):
        super(KVCacheMgrOp, self).__init__(auto_prefix=False)
        self.shape = P.Shape()
        self.max_cache_length = max_cache_length
        self.act_len = act_len
        self.sub = P.Sub()
        self.div = P.Div()
        self.pad = P.PadV3()
        self.concat = P.Concat(0)
        self.slice = P.StridedSlice()

        self.use_past = use_past
        self.n_head = num_heads
        self.size_per_head = hidden_size // self.n_head
        # pylint: disable=W0212
        self.prompt_kvcache = P._inner_ops.PromptKVCache().shard((
            (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
            (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
            (parallel_config.data_parallel,), (1,), (1,), (1,), (1,)))
        # pylint: disable=W0212
        self.decoder_kvcache = P._inner_ops.DecoderKVCache().shard((
            (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
            (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
            (parallel_config.data_parallel,), (1,), (1,), (1,), (1,)))
        self.kvcache_concat_dim0 = P.Concat(axis=0)
        self.concat_dim0 = ops.Concat(axis=0) # need shard
        self.kvcache_concat_dim2 = P.Concat(axis=2)
        self.kvcache_pad_tensor = ops.zeros((3,), mstype.int64)
        self.kvcache_pad8_tensor = ops.zeros((4,), mstype.int64)

        self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

        self.batch_index_pad_tmp = Tensor(np.array([0, 0, 0, 0]), mstype.int64)
        self.seq_length_tensor_pad = Tensor(np.array([1, 0, 0, 0, 0, 0, 0, 0]), mstype.int64)
        # self.seq_length_tensor = Tensor([self.seq_length], dtype=ms.int64)
        # self.seq_length_tensor_pad = self.kvcache_concat_dim0((self.seq_length_tensor, self.kvcache_pad_tensor))
        self.seq_length_axis_tensor = Tensor([2], dtype=mstype.int64)
        self.seq_length_axis_tensor_pad = self.kvcache_concat_dim0((self.seq_length_axis_tensor,
                                                                    self.kvcache_pad_tensor))

        self.input_mask_ones = Tensor(np.ones(self.max_cache_length), mstype.float32)

        self.mul1 = P.Mul().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                   (1, 1, 1, 1)))
        self.add1 = P.TensorAdd().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                         (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))

        self.dtype = compute_dtype
        self.key_shape = (1, num_heads, max_cache_length, self.size_per_head)
        self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
        self.value_shape = (1, num_heads, max_cache_length, self.size_per_head)
        self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")

    def construct(self, key, value, batch_valid_length=None, batch_index=None):
        """The forward compute of KvCacheMsg."""
        # attention_mask : is_first_iteration = True : [bs, seq_length, seq_length], is_first_iteration = False : [bs, 1, max_seq_length]
        batch_size, _, seq_length, _ = self.shape(key)

        # incremental inference needs to pad k,v to k_past and v_past
        if self.use_past:
            batch_index_pad = self.kvcache_concat_dim0((batch_index.astype(mstype.int64), self.kvcache_pad_tensor))

            max_seq_length = self.div(Tensor(self.max_cache_length, mstype.int32),
                                      batch_size).reshape((1,)).astype(mstype.int64)
            seq_length_tensor_pad = self.kvcache_concat_dim0((max_seq_length, self.kvcache_pad_tensor))
            if self.is_first_iteration:
                pad_length = self.sub(max_seq_length.astype(mstype.int32),
                                      seq_length).reshape((1,)).astype(mstype.int32)
                pad_config = self.concat((Tensor([0, 0, 0, 0, 0], mstype.int32),
                                          pad_length, Tensor([0, 0], mstype.int32)))
                # pad to max_seq_length
                # key : [bs, num_heads, seq_length, size_per_head]
                key_present = self.pad(key, pad_config, Tensor(0, mstype.float16))
                value_present = self.pad(value, pad_config, Tensor(0, mstype.float16))

                bvl = F.reshape(batch_valid_length, (-1,)).astype(mstype.int64)
                self.prompt_kvcache(self.key_past, key_present, bvl, batch_index_pad,
                                    self.seq_length_axis_tensor_pad,
                                    seq_length_tensor_pad, seq_length_tensor_pad)
                self.prompt_kvcache(self.value_past, value_present, bvl, batch_index_pad,
                                    self.seq_length_axis_tensor_pad,
                                    seq_length_tensor_pad, seq_length_tensor_pad)
                key_present = ops.depend(key_present, key_present)
                value_present = ops.depend(value_present, value_present)
            else:
                self.decoder_kvcache(self.key_past, key, batch_valid_length.astype(mstype.int64),
                                     self.kvcache_pad8_tensor, self.seq_length_axis_tensor_pad,
                                     seq_length_tensor_pad, seq_length_tensor_pad)

                self.decoder_kvcache(self.value_past, value, batch_valid_length.astype(mstype.int64),
                                     self.kvcache_pad8_tensor, self.seq_length_axis_tensor_pad,
                                     seq_length_tensor_pad, seq_length_tensor_pad)
                key, value = self.key_past, self.value_past
                key = F.reshape(key, (batch_size, self.n_head, -1, self.size_per_head))
                value = F.reshape(value, (batch_size, self.n_head, -1, self.size_per_head))
                key_present, value_present = None, None
        return key, value
