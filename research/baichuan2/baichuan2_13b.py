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
"""Baichuan2_13b models' APIs."""
from typing import Optional
import math
import numpy as np
import mindspore.common.dtype as mstype

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn, ops
from mindspore.common.parameter import Parameter
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer, HeUniform
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

try:
    from mindspore.nn.layer.flash_attention import FlashAttention

    FLASHATTENTION_VALID = True
except ImportError:
    FLASHATTENTION_VALID = False

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.base_model import BaseModel
from mindformers.models.utils import cell_reuse
from mindformers.modules.transformer.op_parallel_config import _check_config, default_dpmp_config
from mindformers.modules.transformer import AttentionMask, TransformerOpParallelConfig
from mindformers.modules.layers import Linear, _check_input_dtype, _check_past_none_input_none, build_alibi_tensor_v2
from mindformers.modules.kv_cache_mgr_baichuan2 import KVCacheMgrOp
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.models.llama.llama import layer_compute_dtype
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaFeedForward, LlamaRMSNorm
from mindformers.tools.logger import logger

__all__ = ['Baichuan13BV2ForCausalLM', 'Baichuan13BV2Model']


class CausalMask(AttentionMask):
    r"""
        Get the Lower triangular matrix from the input mask. The input mask is a 2D tensor (batch_size, seq_length)
        with 1 and 0, where 1 indicates the current position is a valid token, otherwise not.

        Args:
            seq_length(int): The sequence length of the input tensor.
            parallel_config(OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                                               an instance of `OpParallelConfig` with default args.

        Inputs:
            - **input_mask** (Tensor) - The mask indicating whether each position is a valid input with
              (batch_size, seq_length).

        Outputs:
            Tensor. The attention mask matrix with shape (batch_size, seq_length, seq_length).
    """

    def __init__(self, seq_length, is_dynamic=False, parallel_config=default_dpmp_config):
        super(CausalMask, self).__init__(seq_length, parallel_config)
        # self.seq_length = seq_length
        self.parallel_config = parallel_config
        self.is_dynamic = is_dynamic
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()

    def construct(self, input_mask):
        """Forward process of the AttentionMask"""
        input_mask = self.cast(self.not_equal(input_mask, 0), mstype.float16)
        bs, seq_len = self.shape(input_mask)
        shape_right = (bs, 1, seq_len)
        shape_left = (bs, seq_len, 1)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        if not self.is_dynamic:
            lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        else:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
            lower_traiangle = self.expand_dim(lower_triangle_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.multiply(attention_mask, lower_traiangle)
        return attention_mask


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Baichuan13BV2ForCausalLM(BaseModel):
    r"""
        Provide baichuan2_13B training loss or logits through network.
        Args:
            config (LlamaConfig): The config of baichuan2_13B model.

        Inputs:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
              prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            Tensor, the loss or logits of the network.

        Examples:
            >>> from mindformers.models.llama import LlamaConfig
            >>> from research.baichuan2.baichuan2_13b import Baichuan13BV2ForCausalLM
            >>> config = LlamaConfig(batch_size=2)
            >>> network = Baichuan13BV2ForCausalLM(config=config)
        """

    @cell_reuse
    def __init__(self, config: LlamaConfig = None):
        super(Baichuan13BV2ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.seq_length = config.seq_length
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.is_first_iteration = True
        self.vocab_size = config.vocab_size
        self.dtype = config.compute_dtype

        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather()
        self.argmax = P.Argmax(-1)
        self.is_sample_acceleration = config.is_sample_acceleration
        self.p_all_ones = Tensor(np.ones((config.batch_size, 1), np.float32), mstype.float32)

        self.model = Baichuan13BV2Model(config=config)
        self.lm_head = NormHead(hidden_size=config.hidden_size,
                                vocab_size=config.vocab_size,
                                compute_dtype=config.compute_dtype)
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1), (dp,)))
            self.lm_head.shard(config.parallel_config)

            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.lm_head.set_comm_fusion(2)
            else:
                self.lm_head.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    # pylint: disable=W0613
    def prepare_inputs_for_export(self, full_model=True):
        """Get Baichuan13BV2 model input tuple for export."""
        seq_length = self.seq_length
        if full_model:
            logger.info('\nexporting with batch_size = %s, seq = %s ...', self.config.batch_size, seq_length)
            input_ids = Tensor(np.ones([self.config.batch_size, seq_length]), dtype=mstype.int32)
            input_position = Tensor([1] * self.config.batch_size, dtype=mstype.int32)
            init_reset = Tensor([False], mstype.bool_)
            batch_valid_length = Tensor([[1] * self.config.batch_size], dtype=mstype.int32)
        else:
            logger.info('\nexporting with batch_size = %s, seq = 1 ...', self.config.batch_size)
            input_ids = Tensor(np.ones([self.config.batch_size, 1]), dtype=mstype.int32)
            input_position = Tensor([1] * self.config.batch_size, dtype=mstype.int32)
            init_reset = Tensor([True], mstype.bool_)
            batch_valid_length = Tensor([[1] * self.config.batch_size], dtype=mstype.int32)
        return input_ids, None, input_position, None, None, None, init_reset, batch_valid_length

    def _in_graph_gather(self, logits, input_position):
        if (not self.use_past or self.is_first_iteration) and input_position is not None:
            logits = self.gather(logits.view(-1, self.vocab_size), input_position, 0)
        return logits

    def _in_graph_argmax(self, logits):
        return self.reshape(self.argmax(logits), (-1, 1))

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None):
        """Baichuan13BV2ForCausalLM forward."""
        bs, seq_len = self.shape(input_ids)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
        else:
            tokens = input_ids

        output = self.model(tokens, input_position, init_reset, batch_valid_length, batch_index)
        is_prefilling_phase = (not self.use_past or self.is_first_iteration) and input_position is not None
        if is_prefilling_phase:
            output = self.gather(self.reshape(output, (-1, output.shape[-1])), input_position, 0)  # axis=0
        logits = self.lm_head(output)

        if not self.training:
            if is_prefilling_phase:
                logits = self.reshape(logits, (bs, -1))
            else:
                logits = self.reshape(logits, (bs, seq_len, -1))

            if self.is_sample_acceleration:
                return self.get_top_token_id(logits, current_index=input_position)

            # if (not self.use_past or self.is_first_iteration) and input_position is not None:
            #     logits = logits.reshape(-1, logits.shape[-1])
            #     index = input_position.view(-1, )
            #     logits = self.gather(logits, index, 0)
            return logits, tokens

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bs, seq_len), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bs, seq_len), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        logits = self.cast(logits, mstype.float32)
        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

    def get_top_token_id(self, logits, current_index=None):
        """get_top_token_id"""
        # logits = logits.reshape(-1, logits.shape[-1])
        # if self.use_past and not self.is_first_iteration:
        #     logits = logits
        # elif current_index is not None:
        #     index = current_index.view(-1, )
        #     logits = P.Gather()(logits, index, 0)
        # probabilities = P.Softmax(-1)(logits)
        top_token_id = P.Argmax(-1)(logits)
        top_token_id = top_token_id.view(-1, 1)
        return self.p_all_ones, top_token_id


class Baichuan13BV2Model(BaseModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Baichuan13BV2DecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of baichuan2_13b decoderlayer
    """

    def __init__(self,
                 config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        if config.batch_size or config.use_past:
            Validator.check_positive_int(config.batch_size)
        self.dtype = config.compute_dtype
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.qkv_concat = config.qkv_concat
        self.max_cache_length = config.max_cache_length
        self.use_kvcache_mgr = config.use_kvcache_mgr
        self.use_flash_attention = config.use_flash_attention and FLASHATTENTION_VALID
        self.use_causal_attention = config.use_causal_attention
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        elif config.use_flash_attention:
            logger.info("Current MindSpore do not support flash attention.")

        self.get_attention_mask = CausalMask(self.seq_length, is_dynamic=self.is_dynamic,
                                             parallel_config=config.parallel_config.dp_mp_config
                                             ).to_float(config.compute_dtype)
        self.multiply_data = Tensor([-10000.0], dtype=config.compute_dtype)
        self.one = Tensor([1.0], dtype=config.compute_dtype)
        self.all_ones_attention_mask_alibi = P.Ones()((1, config.seq_length), mstype.float32)
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.mul_mask = P.Mul()
        self.mul_alibi = P.Mul()
        self.mul_alibi1 = P.Mul()
        self.sub = P.Sub()
        self.expand_dims = P.ExpandDims()
        self.not_equal = P.NotEqual()
        self.gather = P.Gather()
        self.transpose = P.Transpose()
        self.slice = P.StridedSlice()
        self.shape = P.Shape()
        self.add_alibi = P.Add()

        self.tok_embeddings = LlamaEmbedding(
            config.vocab_size, config.hidden_size, param_init_type=config.param_init_type)
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = Baichuan13BDecodeLayer(config.batch_size,
                                           config.seq_length,
                                           layer_id,
                                           dim=config.hidden_size,
                                           n_heads=config.num_heads,
                                           multiple_of=config.multiple_of,
                                           n_kv_heads=config.n_kv_heads,
                                           ffn_dim_multiplier=config.ffn_dim_multiplier,
                                           norm_eps=config.rms_norm_eps,
                                           compute_dtype=config.compute_dtype,
                                           layernorm_compute_dtype=config.layernorm_compute_type,
                                           softmax_compute_dtype=config.softmax_compute_type,
                                           param_init_type=config.param_init_type,
                                           use_past=config.use_past,
                                           use_flash_attention=config.use_flash_attention,
                                           use_causal_attention=self.use_causal_attention,
                                           is_dynamic=self.is_dynamic,
                                           qkv_concat=self.qkv_concat,
                                           max_cache_length=self.max_cache_length,
                                           use_past_shard=config.use_past_shard,
                                           use_kvcache_mgr=config.use_kvcache_mgr,
                                           parallel_config=config.parallel_config)
            layer_compute_dtype(layer, layer_id, config.offset, config.parallel_config,
                                config.num_layers, select_recompute=config.parallel_config.recompute.select_recompute)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)

        self.alibi_tensor = build_alibi_tensor_v2(seq_len=config.seq_length,
                                                  num_heads=config.num_heads,
                                                  return_tensors='ms',
                                                  dtype=self.dtype)
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.tok_embeddings.set_comm_fusion(2)
                self.norm_out.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.tok_embeddings.shard(config.parallel_config)
            self.sub.shard(((1,), (dp, 1, 1)))
            self.mul_mask.shard(((dp, 1, 1, 1), (1,)))
            self.mul_alibi.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))  # (dp, mp, 1, 1)
            self.mul_alibi1.shard(((1, mp, 1, 1), (dp, 1, 1, 1)))
            self.expand_dims.shard(((dp, 1, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.gather.shard(((dp, mp, 1, 1), (1,)))
            self.add_alibi.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
            self.transpose.shard(((1, mp, dp, 1),))
            self.norm_out.shard((dp, 1, 1))

        if self.use_past:
            if self.is_dynamic:
                self.range = Tensor(np.arange(self.max_cache_length).reshape(1, 1, -1), mstype.int32)
            else:
                self.range = Tensor(np.arange(config.seq_length).reshape(1, 1, -1), mstype.int32)
            self.ones = P.Ones()
            self.gather_past = P.Gather()
            self.expand_dims = P.ExpandDims()
            self.le_past = P.LessEqual()

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, input_position=None, init_reset=True, batch_valid_length=None,
                  batch_index=None):
        """Forward of baichuan2_13b model."""
        # preprocess
        bs, seq_len = self.shape(tokens)
        if self.use_past and not self.use_kvcache_mgr:
            if not isinstance(init_reset, Tensor):
                init_reset = Tensor([init_reset], mstype.bool_)
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bs,), mstype.int32)

        if self.is_dynamic:
            dyn_seq = self.max_cache_length // bs
            seq_range = self.slice(self.range, (0, 0, 0), (bs, 1, dyn_seq), (1, 1, 1))
        else:
            seq_range = self.range

        if self.is_first_iteration:
            cur_pos = batch_valid_length
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float16)
            mask = self.get_attention_mask(input_mask)
            alibi_tensor = self.mul_alibi1(self.alibi_tensor, self.reshape(input_mask, (bs, 1, -1, 1)))
            alibi_tensor = self.mul_alibi(alibi_tensor, self.reshape(input_mask, (bs, 1, 1, -1)))
            # mask: [bs, seq, seq]
            if self.is_dynamic:
                mask = self.slice(mask, (0, 0, 0), (bs, seq_len, seq_len), (1, 1, 1))
        else:
            cur_pos = batch_valid_length - 1
            valid_length = self.reshape(cur_pos, (-1, 1, 1))
            mask_range = self.reshape(seq_range, (1, 1, -1))
            mask = self.le_past(mask_range, valid_length)
            alibi_tensor = self.gather(self.alibi_tensor, cur_pos, 2)
            alibi_tensor = self.transpose(alibi_tensor, (2, 1, 0, 3))
            alibi_tensor = self.mul_alibi1(alibi_tensor, self.expand_dims(mask, 1))
            # mask: [bs, 1, 1]
        # mask = self.sub(self.one, self.cast(mask, mstype.float32))
        mask = self.sub(self.one, mask)
        if not self.use_flash_attention:
            mask = self.expand_dims(mask, 1)
            mask = self.mul_mask(mask, self.multiply_data)
            mask = self.add_alibi(mask, alibi_tensor)

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            if self.use_kvcache_mgr:
                h, _ = self.layers[i](h, alibi_tensor, mask, init_reset=init_reset,
                                      batch_valid_length=cur_pos, batch_index=batch_index)
            else:
                h, _ = self.layers[i](h, alibi_tensor, mask, init_reset=init_reset,
                                      batch_valid_length=batch_valid_length, batch_index=batch_index)
        output = self.norm_out(h)
        return output


class Baichuan13BDecodeLayer(nn.Cell):
    r"""
        Transformer Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            seq_length(int): The input sequence length.
            layer_id(int): The layer id of current transformer block layer.
            dim(int): The hidden size of the input.
            num_heads(int): The number of the heads.
            multiple_of(int): The SwiGLU hidden layer size multiple of large power of 2.
            norm_eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_dtype(dtype.Number): The computation type of the layer.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            layernorm_compute_type(dtype.Number): The computation type of the norm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **alibi_tensor** (Tensor) - Alibi Tensor for position embedding used in attention.
            - **mask** (Tensor) - Float Tensor, If the use_past is
            False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, head_dim, seq_length),
              (batch_size, num_heads, seq_length, head_dim)).

    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 n_heads: int = 8,
                 multiple_of: int = 256,
                 n_kv_heads: Optional[int] = None,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 use_flash_attention=False,
                 use_causal_attention=False,
                 is_dynamic=False,
                 qkv_concat=False,
                 max_cache_length: int = 4096,
                 # compute_in_2d=False,
                 use_past_shard=False,
                 use_kvcache_mgr=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads

        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.is_dynamic = is_dynamic
        self.key_past = None
        self.value_past = None
        self.use_seq_parallel = parallel_config.use_seq_parallel
        self.use_kvcache_mgr = use_kvcache_mgr

        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.add = P.Add()
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = Baichuan13BAttention(batch_size=batch_size,
                                              seq_length=seq_length,
                                              dim=dim,
                                              n_heads=n_heads,
                                              n_kv_heads=n_kv_heads,
                                              compute_dtype=compute_dtype,
                                              softmax_compute_dtype=softmax_compute_dtype,
                                              param_init_type=param_init_type,
                                              use_past=use_past,
                                              use_flash_attention=use_flash_attention,
                                              use_causal_attention=use_causal_attention,
                                              is_dynamic=is_dynamic,
                                              qkv_concat=qkv_concat,
                                              max_cache_length=max_cache_length,
                                              # compute_in_2d=compute_in_2d,
                                              use_past_shard=use_past_shard,
                                              use_kvcache_mgr=use_kvcache_mgr,
                                              parallel_config=parallel_config)
        self.feed_forward = LlamaFeedForward(dim=self.hidden_size,
                                             hidden_dim=4 * self.hidden_size,
                                             multiple_of=multiple_of,
                                             ffn_dim_multiplier=ffn_dim_multiplier,
                                             compute_dtype=compute_dtype,
                                             param_init_type=param_init_type)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.feed_forward.shard(parallel_config)
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.attention_norm.shard((dp, 1, 1))
            self.ffn_norm.shard((dp, 1, 1))
            self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))

            if parallel_config.use_seq_parallel and self.is_first_iteration:
                self.add.shard(((dp, mp, 1), (dp, mp, 1)))
                self.attention_norm.shard((dp, mp, 1))
                self.ffn_norm.shard((dp, mp, 1))
                self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        if self.use_past:
            if not self.use_kvcache_mgr:
                kv_shape = (batch_size, self.n_kv_head, seq_length, self.head_dim)
                self.key_past = Parameter(Tensor(np.zeros(kv_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(kv_shape), self.dtype), name="value_past")
            self.mul_past = P.Mul().shard(((dp, 1, 1, 1), (1,)))
            self.assign_past = P.Assign().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if use_past_shard:
                self.mul_past.shard(((dp, mp, 1, 1), (1,)))
                self.assign_past.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))

    def construct(self, x, alibi_tensor, mask=None, init_reset=True, batch_valid_length=None, batch_index=None):
        """ Forward of transformer block. """
        bs = x.shape[0]
        if self.use_past and not self.is_dynamic:
            if not isinstance(init_reset, Tensor):
                init_reset = Tensor([init_reset], mstype.bool_)
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bs,), mstype.int32)
        self._check_input(x, alibi_tensor, mask, init_reset, batch_valid_length)
        input_x = self.attention_norm(x)

        key_reset = None
        value_reset = None
        if self.use_past and self.is_first_iteration:
            if not self.use_kvcache_mgr:
                # reset states, init_reset True for reuse and False for reset
                self.assign_past(self.key_past, self.mul_past(self.key_past, self.cast(init_reset, self.dtype)))
                self.assign_past(self.value_past, self.mul_past(self.value_past, self.cast(init_reset, self.dtype)))
                key_reset = self.key_past
                value_reset = self.value_past
                # add dependency for desired execution order
                input_x = ops.depend(input_x, key_reset)
                input_x = ops.depend(input_x, value_reset)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        h, layer_present = self.attention(input_x, alibi_tensor, mask, self.key_past,
                                          self.value_past, batch_valid_length, batch_index)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)

        if not self.use_kvcache_mgr:
            value_update = None
            key_update = None
            if self.use_past:
                # current key and value
                key_present, value_present = layer_present
                # update key and value calculated this step
                self.assign_past(self.key_past, key_present)
                self.assign_past(self.value_past, value_present)
                key_update = self.key_past
                value_update = self.value_past
                # add dependency for desired execution order
                key_update = ops.depend(key_update, key_reset)
                value_update = ops.depend(value_update, value_reset)

            # add dependency for desired execution order
            ffn_out = ops.depend(ffn_out, value_update)
            ffn_out = ops.depend(ffn_out, key_update)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        return out, layer_present

    def _check_input(self, x, alibi_tensor, mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(alibi_tensor.dtype, "alibi_tensor",
                           [mstype.float32, mstype.float16], self.cls_name)
        if mask is not None:
            _check_input_dtype(mask.dtype, "input_mask", [mstype.float32, mstype.float16], self.cls_name)

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
                                    init_reset_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)

        if self.use_past and not self.is_dynamic:
            _check_input_dtype(init_reset.dtype, "init_reset", [mstype.bool_], self.cls_name)
            _check_input_dtype(batch_valid_length.dtype, "batch_valid_length", [mstype.int32], self.cls_name)
        return True


class Baichuan13BAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in Baichuan.

    Args:
            - **batch_size** (int): The batch size of the input tensor when do increnmental prediction. Should be a
                positive value.
                When do training or prediction, the argument will not work and the user can just pass None to the
                argument.
            - **src_seq_length** (int): The sequence length of the query vector.
            - **tgt_seq_length** (int): The sequence length of the key and value vector.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words' state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default False.
            - **parallel_config** (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **alibi_tensor** (Tensor) - Alibi Tensor for position embedding used in attention.
            - **mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, head_dim, tgt_seq_length).
                The past calculated key vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **value_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, tgt_seq_length,
                head_dim).
                The past calculated value vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.

    Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
                shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
                if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
                ((batch_size, num_heads, head_dim, tgt_seq_length),
                (batch_size, num_heads, tgt_seq_length, head_dim)).
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 use_flash_attention=False,
                 use_causal_attention=False,
                 is_dynamic=False,
                 qkv_concat=False,
                 max_cache_length: int = 4096,
                 # compute_in_2d=False,
                 use_past_shard=False,
                 use_kvcache_mgr=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_cache_length = max_cache_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head

        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        # self.compute_in_2d = compute_in_2d
        self.use_flash_attention = use_flash_attention and FLASHATTENTION_VALID
        self.is_dynamic = is_dynamic
        self.qkv_concat = qkv_concat
        self.use_causal_attention = use_causal_attention
        self.use_kvcache_mgr = use_kvcache_mgr

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))

        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.add_alibi = P.Add()
        self.softmax = P.Softmax()
        self.cast = P.Cast()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()

        self.wo = Linear(in_channels=self.hidden_size,
                         out_channels=self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wq = Linear(self.hidden_size,
                         self.hidden_size,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wk = Linear(self.hidden_size,
                         self.n_kv_head * self.head_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wv = Linear(self.hidden_size,
                         self.n_kv_head * self.head_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        if self.qkv_concat:
            self.wqkv = Linear(self.hidden_size,
                               self.hidden_size + self.n_kv_head * self.head_dim + self.n_kv_head * self.head_dim,
                               has_bias=False,
                               compute_dtype=compute_dtype,
                               param_init_type=param_init_type)
        if self.use_causal_attention:
            self.causal_attention = CausalAttention(parallel_config,
                                                    local_size=65536,
                                                    block_size=512,
                                                    inv_norm_factor=self.inv_norm_factor).to_float(mstype.float16)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.transpose.shard(((dp, 1, mp, 1),))
            self.merger_head_transpose.shard(((dp, mp, 1, 1),))
            self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.mul.shard(((dp, mp, 1, 1), ()))
            self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
            self.add_alibi.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.softmax.shard(((dp, mp, 1, 1),))
            # self.tile_kv.shard(((dp * mp, 1, 1, 1),))
            self.tile_kv.shard(((dp, mp, 1, 1),))

            self.wq.shard(((dp, 1), (mp, 1)))
            self.wk.shard(((dp, 1), (mp, 1)))
            self.wv.shard(((dp, 1), (mp, 1)))
            self.wo.shard(((dp, mp), (1, mp)))
            if self.qkv_concat:
                self.wqkv.shard(((dp, 1), (mp, 1)))
        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
        if parallel_config.recompute.select_recompute:
            self.tile_kv.recompute()
            self.batch_matmul_q_k.recompute()
            self.mul.recompute()
            self.add.recompute()
            self.cast_attn.recompute()
            self.softmax.recompute()
            self.batch_matmul.recompute()

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(self.head_dim, dp=dp, mp=mp, next_block_num=0)
            # self.flash_attention.shard(((dp, mp, 1, 1), (dp, mp, 1, 1), (dp, mp, 1, 1), (dp, 1, 1), ()))
            # if parallel_config.recompute.select_recompute:
            #     self.flash_attention.recompute()

        if self.use_kvcache_mgr:
            self.kvcache_mgr = KVCacheMgrOp(hidden_size=self.n_kv_head * self.head_dim,
                                            num_heads=self.n_kv_head,
                                            compute_dtype=self.dtype,
                                            use_past=self.use_past,
                                            parallel_config=parallel_config,
                                            max_cache_length=self.max_cache_length)

        if self.use_past:
            # operators used for state reuse
            seq_range = np.arange(seq_length).reshape(1, 1, -1)
            self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
            self.expand_dims = P.ExpandDims().shard(((dp, 1, 1),))
            self.add_past = P.Add().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.equal = P.Equal().shard(((dp, 1, 1), (dp, 1, 1)))
            self.less = P.Less().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul_past = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if use_past_shard:
                self.add_past.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.mul_past.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))

    # pylint: disable=W0613
    def construct(self, x: Tensor, alibi_tensor: Tensor, mask=None,
                  key_past=None, value_past=None, batch_valid_length=None, batch_index=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        bs, seq_len, _ = self.shape(x)

        if not self.qkv_concat:
            # [bs * seq/1, hidden_dim]
            query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
            key = self.cast(self.wk(x), self.dtype)  # dp, 1 -> dp, mp
            value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp
        else:
            # qkv_concat
            qkv = self.cast(self.wqkv(x), self.dtype)
            query, key, value = ops.split(qkv, (self.hidden_size, self.n_kv_head * self.head_dim,
                                                self.n_kv_head * self.head_dim), axis=2)

        # [bs * seq/1, hidden_dim]
        if self.use_past and not self.is_first_iteration:
            query = self.reshape(query, (bs, self.n_head, 1, self.head_dim))
            key = self.reshape(key, (bs, self.n_kv_head, 1, self.head_dim))
            value = self.reshape(value, (bs, self.n_kv_head, 1, self.head_dim))
        else:
            query = self.reshape(query, (bs, seq_len, self.n_head, self.head_dim))
            key = self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim))
            value = self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim))
            # [bs, seq/1, n_head/n_kv_head, head_dim]
            query = self.transpose(query, (0, 2, 1, 3))
            key = self.transpose(key, (0, 2, 1, 3))
            value = self.transpose(value, (0, 2, 1, 3))

        # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
        key_present = key
        value_present = value
        if self.use_kvcache_mgr:
            key, value = self.kvcache_mgr(key, value, batch_valid_length, batch_index)
        elif self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = (
                    self.less(self.range, batch_valid_length.view(-1, 1, 1))).astype(self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul_past(key, self.expand_dims(valid_length_vector, 3))
                value_present = self.mul_past(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            else:
                # Get the current token position index
                valid_length = batch_valid_length - 1
                valid_length = self.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = (self.equal(self.range, valid_length)).astype(self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul_past(key, self.expand_dims(valid_length_vector, 3))
                current_value = self.mul_past(value, self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add_past(key_past, current_key)
                value = self.add_past(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value

        layer_present = (key_present, value_present)
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        key = self._repeat_kv(key, self.n_rep)
        value = self._repeat_kv(value, self.n_rep)
        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        if self.use_flash_attention:
            attention = self.flash_attention(query, key, value, mask)
            attention = self._merge_heads(attention)
        else:
            if self.use_causal_attention and (not self.use_past or self.is_first_iteration):
                attention = self.causal_attention(query, key, value, mask)
                attention = self._merge_heads(attention)
            else:
                attention = self._attn(query, key, value, mask)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(attention)  # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)

        return output, layer_present

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        bs, seq_len, n_head, head_dim = self.shape(x)
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        # score = self.add_alibi(score, alibi_tensor)

        score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class NormHead(nn.Cell):
    """
    NormHead Layer.

        Args:
            hidden_size (int): The hidden size of the input.
            vocab_size (int): Size of the dictionary of embeddings.
            compute_type (dtype.Number): The compute type.
            eps (number): A small positive value prevents division by zero.

        Inputs:
            - hidden_states (Tensor) - Tensor of shape :math:`(batch, seq_length, hidden_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, vocab_size)`.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 compute_dtype=mstype.float32,
                 eps=1e-5):
        super().__init__()
        self.weight = Parameter(
            initializer(HeUniform(negative_slope=math.sqrt(5)),
                        [vocab_size, hidden_size],
                        mstype.float32),
            name='weight',
            parallel_optimizer=False)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.add = P.Add()
        self.real_div = P.RealDiv()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.sum = P.ReduceSum()
        self.eps = Tensor([eps], mstype.float32)

        self.matmul = P.MatMul(transpose_b=True)
        self.cast = P.Cast()
        self.compute_dtype = compute_dtype
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.assign = P.Assign()

    def construct(self, hidden_states):
        """Forward process of the NormHead"""
        out_shape = P.Shape()(hidden_states)[:-1] + (self.vocab_size,)
        hidden_states = self.reshape(hidden_states, (-1, self.hidden_size))

        if self.is_first_iteration:
            # 全量推理
            variance = self.square(self.weight)
            variance = self.sum(variance, 1)
            variance = self.reshape(variance, (-1, 1))
            variance_eps = self.sqrt(self.add(variance, self.eps))
            # 更新self.weight
            norm_weight = self.real_div(self.weight, variance_eps)
            self.assign(self.weight, norm_weight)
            norm_weight = ops.depend(norm_weight, norm_weight)
        else:
            # 增量推理，直接用已归一化的权重
            norm_weight = self.weight

        ori_type = hidden_states.dtype
        out = self.matmul(hidden_states.astype(self.compute_dtype),
                          norm_weight.astype(self.compute_dtype))
        out = self.reshape(out, out_shape)
        return self.cast(out, ori_type)

    def shard(self, parallel_config):
        """sharding for norm head"""
        if parallel_config.vocab_emb_dp:
            self.square.shard(((1, 1),))
            self.sqrt.shard(((1, 1),))
            self.add.shard(((1, 1), (1,)))
            self.real_div.shard(((1, 1), (1, 1)))
            self.sum.shard(((1, 1),))
            self.matmul.shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.square.shard(((parallel_config.model_parallel, 1),))
            self.sqrt.shard(((parallel_config.model_parallel, 1),))
            self.add.shard(((parallel_config.model_parallel, 1), (1,)))
            self.real_div.shard(((parallel_config.model_parallel, 1),
                                 (parallel_config.model_parallel, 1)))
            self.sum.shard(((parallel_config.model_parallel, 1),))
            self.matmul.shard(((parallel_config.data_parallel, 1),
                               (parallel_config.model_parallel, 1)))


class CausalAttention(nn.Cell):
    """
    CausalAttention Layer.
    """

    def __init__(self, parallel_config, dropout_rate=0.1, local_size=65536, block_size=128, inv_norm_factor=1):
        super(CausalAttention, self).__init__()

        self.local_size = local_size
        self.block_size = block_size
        self.qkv_slice = P.StridedSlice().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.attn_mask_slice = P.StridedSlice().shard(((parallel_config.data_parallel, 1, 1, 1),))
        self.qk_bmm = P.BatchMatMul(transpose_b=True).shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.attn_mask_mul = P.Mul().shard(((parallel_config.data_parallel, 1, 1, 1), (1,)))
        self.attn_mask_expand_dims = P.ExpandDims().shard(((parallel_config.data_parallel, 1, 1),))
        self.attn_mask_add = P.Add().shard(((parallel_config.data_parallel, 1, 1, 1),
                                            (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.softmax = P.Softmax().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.attention_dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        self.attention_dropout.dropout.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.pv_bmm = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.o_concat = P.Concat(axis=2).shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.inv_norm_factor = inv_norm_factor
        self.multiply_data = Tensor([-10000.0,], dtype=mstype.float16)
        self.mul = P.Mul()

    def construct(self, q, k, v, attention_mask):
        """Forward process of the CausalAttention"""
        bsz, head_num, tgt_len, head_dim = q.shape
        sparse_groups = tgt_len // self.block_size
        # attention_mask = self.attn_mask_mul(attention_mask, self.multiply_data)
        prev_block_num = self.local_size // self.block_size - 1
        output = None
        for i in range(sparse_groups):
            q_begin = i * self.block_size
            q_end = (i + 1) * self.block_size
            kv_begin = max(0, i - prev_block_num) * self.block_size
            kv_end = (i + 1) * self.block_size
            # slice
            cur_q = self.qkv_slice(q, (0, 0, q_begin, 0), (bsz, head_num, q_end, head_dim), (1, 1, 1, 1))
            cur_k = self.qkv_slice(k, (0, 0, kv_begin, 0), (bsz, head_num, kv_end, head_dim), (1, 1, 1, 1))
            cur_v = self.qkv_slice(v, (0, 0, kv_begin, 0), (bsz, head_num, kv_end, head_dim), (1, 1, 1, 1))
            adder = self.attn_mask_slice(attention_mask, (0, 0, q_begin, kv_begin),
                                         (bsz, attention_mask.shape[1], q_end, kv_end), (1, 1, 1, 1))
            # cur_alibi_tensor = self.attn_mask_slice(alibi_tensor, (0, 0, q_begin, kv_begin),
            #                                         (bsz, alibi_tensor.shape[1], q_end, kv_end), (1, 1, 1, 1))
            # q * k.T
            cur_score = self.qk_bmm(cur_q, cur_k)
            cur_score = self.mul(cur_score, self.inv_norm_factor)
            # adder = self.attn_mask_expand_dims(adder, 1)
            # cur_score = self.attn_mask_add(cur_alibi_tensor, cur_score)
            cur_score = self.attn_mask_add(adder, cur_score)
            cur_probs = self.softmax(cur_score)
            # p * v
            cur_o = self.pv_bmm(cur_probs, cur_v)
            if output is None:
                output = cur_o
            else:
                output = self.o_concat((output, cur_o))
        return output
