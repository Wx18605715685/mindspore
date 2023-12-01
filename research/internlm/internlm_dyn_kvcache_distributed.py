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
"""InternLM models' APIs."""
import numpy as np

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

import mindspore as ms
from mindspore import Tensor, nn, ops
import mindspore.common.dtype as mstype
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
try:
    # pylint: disable=W0611
    from mindspore.nn.layer.flash_attention import FlashAttention
    FLASHATTENTION_VALID = True
except ImportError:
    FLASHATTENTION_VALID = False

from mindformers import logger
from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.base_model import BaseModel
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaRMSNorm, precompute_freqs_cis
from mindformers.models.llama.llama import layer_compute_dtype
from mindformers.models.llama import LlamaConfig
from mindformers.modules.layers import Linear
from mindformers.modules.transformer.op_parallel_config import _check_config, default_dpmp_config
from mindformers.modules.transformer.transformer import AttentionMask
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.pet.tuners.lora_adapter import LoraAdapter
from mindformers.models.utils import cell_reuse
from research.internlm.internlm_transformer_dyn_kvcache_distributed import InternLMDecodeLayer


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


class InternLMModel(BaseModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLMDecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the output of internlm decoderlayer
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
        self.act_len = config.act_len
        self.max_cache_length = config.max_cache_length
        self.use_kvcache_mgr = config.use_kvcache_mgr
        self.use_flash_attention = config.use_flash_attention and FLASHATTENTION_VALID
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        elif config.use_flash_attention:
            logger.info("Current MindSpore do not support flash attention.")

        seq_length = self.seq_length
        self.freqs_cos, self.freqs_sin, self.swap_mask = precompute_freqs_cis(
            self.head_dim, seq_length, dtype=config.rotary_dtype,
            pretrain_seqlen=config.pretrain_seqlen, extend_method=config.extend_method)
        self.get_attention_mask = CausalMask(seq_length, is_dynamic=self.is_dynamic,
                                             parallel_config=config.parallel_config.dp_mp_config
                                             ).to_float(config.compute_dtype)

        self.multiply_data = Tensor([-10000.0], dtype=config.compute_dtype)
        self.one = Tensor([1.0], dtype=config.compute_dtype)
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.mul_mask = P.Mul()
        self.sub = P.Sub()
        self.expand_dims = P.ExpandDims()
        self.not_equal = P.NotEqual()
        self.gather = P.Gather()
        self.slice = P.StridedSlice()
        self.shape = P.Shape()

        self.tok_embeddings = LlamaEmbedding(
            config.vocab_size, config.hidden_size, param_init_type=config.param_init_type)
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = InternLMDecodeLayer(config.batch_size,
                                        config.seq_length,
                                        layer_id,
                                        dim=config.hidden_size,
                                        n_heads=config.num_heads,
                                        multiple_of=config.multiple_of,
                                        n_kv_heads=config.n_kv_heads,
                                        ffn_dim_multiplier=config.ffn_dim_multiplier,
                                        norm_eps=config.rms_norm_eps,
                                        bias=config.attention_bias,
                                        compute_dtype=config.compute_dtype,
                                        layernorm_compute_dtype=config.layernorm_compute_type,
                                        softmax_compute_dtype=config.softmax_compute_type,
                                        rotary_dtype=config.rotary_dtype,
                                        param_init_type=config.param_init_type,
                                        use_past=config.use_past,
                                        use_flash_attention=config.use_flash_attention,
                                        is_dynamic=self.is_dynamic,
                                        qkv_concat=self.qkv_concat,
                                        act_len=self.act_len,
                                        max_cache_length=self.max_cache_length,
                                        # compute_in_2d=config.compute_in_2d,
                                        use_past_shard=config.use_past_shard,
                                        use_rope_slice=config.use_rope_slice,
                                        use_kvcache_mgr=config.use_kvcache_mgr,
                                        parallel_config=config.parallel_config)
            layer_compute_dtype(layer, layer_id, config.offset, config.parallel_config,
                                config.num_layers, select_recompute=config.parallel_config.recompute.select_recompute)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)

        dp = config.parallel_config.data_parallel
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

            self.tile.shard(((1, 1, 1, 1),))
            self.sub.shard(((1,), (dp, 1, 1)))
            self.mul_mask.shard(((dp, 1, 1, 1), (1,)))
            self.expand_dims.shard(((dp, 1, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1), (1,)))
            self.norm_out.shard((dp, 1, 1))

        if self.use_past:
            self.range = Tensor(np.arange(config.seq_length).reshape((1, 1, -1)), mstype.int32)  # 1, 1, maxseq
            self.equal_past = P.Equal().shard(((dp, 1, 1), (dp, 1, 1)))
            self.less_past = P.Less().shard(((dp, 1, 1), (dp, 1, 1)))
            self.gather_past = P.Gather()
            self.le_past = P.LessEqual()

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, input_position=None, batch_valid_length=None,
                  batch_index=None, zactivate_len=None):
        """Forward of internlm model."""
        # preprocess
        bs, seq_len = self.shape(tokens)
        seq_range = self.range  # 1, 1, maxseq

        if self.is_first_iteration:
            cur_pos = batch_valid_length
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
            mask = self.get_attention_mask(input_mask)
            if self.use_past:
                valid_length_vector = self.cast(self.less_past(
                    seq_range, self.reshape(batch_valid_length, (-1, 1, 1))), self.dtype)
            else:
                valid_length_vector = None
            if self.is_dynamic:
                mask = self.slice(mask, (0, 0, 0), (bs, seq_len, seq_len), (1, 1, 1))
                freqs_cos_value = self.slice(self.freqs_cos, (0, 0), (seq_len, self.head_dim), (1, 1))
                freqs_sin_value = self.slice(self.freqs_sin, (0, 0), (seq_len, self.head_dim), (1, 1))
            else:
                freqs_cos_value = self.freqs_cos
                freqs_sin_value = self.freqs_sin

            freqs_cis = (self.tile(self.reshape(freqs_cos_value, (1, 1, seq_len, self.head_dim)), (bs, 1, 1, 1)),
                         self.tile(self.reshape(freqs_sin_value, (1, 1, seq_len, self.head_dim)), (bs, 1, 1, 1)),
                         self.swap_mask)

            # mask: [bs, seq, seq]
        else:
            if self.act_len:
                seq_range = self.slice(self.range, (0, 0, 0), (1, 1, ops.shape(zactivate_len)[0]), (1, 1, 1))
            cur_pos = batch_valid_length - 1
            valid_length = self.reshape(cur_pos, (-1, 1, 1))
            valid_length_vector = self.cast((self.equal_past(seq_range, valid_length)), self.dtype)
            freqs_cis = (self.reshape(self.gather_past(self.freqs_cos, cur_pos, 0), (bs, 1, seq_len, self.head_dim)),
                         self.reshape(self.gather_past(self.freqs_sin, cur_pos, 0), (bs, 1, seq_len, self.head_dim)),
                         self.swap_mask)

            mask_range = self.reshape(seq_range, (1, 1, -1))
            mask = self.le_past(mask_range, valid_length)

        mask = self.sub(self.one, self.cast(mask, self.dtype))
        valid_length_vector = self.expand_dims(valid_length_vector, 3)

        if not self.use_flash_attention:
            mask = self.expand_dims(mask, 1)
            mask = self.mul_mask(mask, self.multiply_data)

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        # h: [bs, seq/1, hidden_dim]
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        for i in range(self.num_layers):
            if self.use_kvcache_mgr:
                h = self.layers[i](h, freqs_cis, mask, valid_length_vector=cur_pos,
                                   batch_index=batch_index, zactivate_len=zactivate_len)

            else:
                h = self.layers[i](h, freqs_cis, mask, valid_length_vector=valid_length_vector)
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InternLMForCausalLM(BaseModel):
    r"""
        Provide internlm training loss or logits through network.
        Args:
            config (LlamaConfig): The config of llama model.

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
        """

    @cell_reuse
    def __init__(self, config: LlamaConfig = None):
        super(InternLMForCausalLM, self).__init__(config, auto_prefix=True)
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
        self.model = InternLMModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")  # meta default: xavier_normal
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1), (dp,)))
            if config.parallel_config.vocab_emb_dp:
                self.lm_head.shard(strategy_matmul=((dp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp, 1), (mp, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    def dummy_tensor(self, shape, dtype):
        if None in shape:
            return ms.Tensor(shape=shape, dtype=dtype)
        return ms.Tensor(np.ones(shape=tuple(shape)), dtype=dtype)

    # pylint: disable=W0613
    def prepare_inputs_for_export(self, full_model=True):
        """Get InternLM model input tuple for export."""
        batch_size = None
        seq_length = None
        act_len = False
        if not full_model:
            seq_length = 1
        init_reset = not full_model
        input_ids = self.dummy_tensor(shape=[batch_size, seq_length], dtype=mstype.int32)
        input_position = self.dummy_tensor(shape=[batch_size], dtype=mstype.int32)
        init_reset = ms.Tensor([init_reset], mstype.bool_)
        batch_valid_length = self.dummy_tensor(shape=[batch_size], dtype=mstype.int64)
        batch_index = self.dummy_tensor(shape=[batch_size], dtype=mstype.int64)
        print(f'input_ids shape: {input_ids.shape}', flush=True)
        print(f'input_position shape: {input_position.shape}', flush=True)
        print(f'batch_valid_length shape: {batch_valid_length.shape}', flush=True)
        if act_len:
            zactivate_len = self.dummy_tensor(shape=[None], dtype=mstype.int64)
            print(f'zactivate_len shape: {zactivate_len.shape}', flush=True)
        else:
            zactivate_len = None
        return input_ids, None, input_position, None, None, None, None, batch_valid_length, batch_index, zactivate_len

    def _in_graph_gather(self, logits, input_position):
        if (not self.use_past or self.is_first_iteration) and input_position is not None:
            logits = self.gather(logits.view(-1, self.vocab_size), input_position, 0)
        return logits

    def _in_graph_argmax(self, logits):
        return self.reshape(self.argmax(logits), (-1, 1))

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None):
        """InternLMForCausalLM forward."""
        bs, seq_len = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bs,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
        else:
            tokens = input_ids

        output = self.model(tokens, input_position, batch_valid_length, batch_index, zactivate_len)
        logits = self.lm_head(output)

        if self.phase == 'predict':
            logits = self.reshape(logits, (bs, seq_len, -1))
            logits = self._in_graph_gather(logits, input_position)
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
        if not self.training:
            logits = self.reshape(logits, (bs, seq_len, -1))
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            self._in_graph_gather(logits, input_position)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InternLMForCausalLMWithLora(InternLMForCausalLM):
    """InternLM Model for finetuning with LoRA

    Args:
        config (LlamaConfig): The config of network.
    """

    def __init__(self, config: LlamaConfig = None):
        ckpt_cfg = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = None
        super().__init__(config)
        # get Pet tuning model.
        config.pet_config.reg_rules = r'.*wq|.*wk|.*wv|.*wo'
        self.model = LoraAdapter.get_pet_model(self.model, config.pet_config)
        # load lora ckpt
        config.checkpoint_name_or_path = ckpt_cfg
        self.load_checkpoint(config)
        # freeze pretrained model
        PetAdapter.freeze_pretrained_model(self, config.pet_config.pet_type)
