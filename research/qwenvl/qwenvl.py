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
# This file was refer to project:
# https://huggingface.co/Qwen/Qwen-VL
# ============================================================================
"""QwenVL models' APIs."""

import os
import math
from collections import OrderedDict
from typing import Optional

import mindspore as ms
import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, Parameter, Tensor
from mindspore import ops
from mindspore.common.initializer import initializer, TruncatedNormal, Normal
from mindspore.ops import operations as P
from qwen.qwen_model import QwenForCausalLM

from mindformers import BaseModel, MultiHeadAttention, MindFormerRegister, MindFormerModuleType
from mindformers.models.vit.vit_modules import get_2d_sincos_pos_embed
from mindformers.modules.layers import LayerNorm, Linear
from mindspore.parallel._utils import _get_parallel_mode
from mindformers.tools.logger import logger
from qwenvl_config import QwenVLConfig


class AbsPos(nn.Cell):
    def __init__(self, src_size: int, tgt_size: int):
        super().__init__()
        self.src_size = int(math.sqrt(src_size))
        self.tgt_size = int(math.sqrt(tgt_size))

        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.flatten = nn.Flatten(start_dim=0, end_dim=2)
        self.resize_shape = ms.Tensor([self.tgt_size, self.tgt_size], ms.int32)
        self.resize = P.ResizeBicubic(align_corners=False, half_pixel_centers=False)
        self.transpose = P.Transpose()

    def construct(self, x: Tensor):
        if self.src_size != self.tgt_size:
            ori_dtype = x.dtype
            x = self.cast(x, ms.float32)
            x = self.reshape(x, (1, self.src_size, self.src_size, -1))
            x = self.transpose(x, (0, 3, 1, 2))
            x = self.resize(x, self.resize_shape)
            x = self.transpose(x, (0, 2, 3, 1))
            x = self.flatten(x)
            x = self.cast(x, ori_dtype)
        return x


class Resampler(nn.Cell):
    def __init__(self, config: QwenVLConfig):
        super().__init__()

        self.num_queries = config.num_queries
        self.query_grid_size = int(math.sqrt(config.num_queries))

        self.embed_dim = config.proj_output_dim
        self.num_heads = self.embed_dim // 128

        self.kv_dim = config.vision_config.hidden_size

        self.pos_embed = Parameter(get_2d_sincos_pos_embed(self.embed_dim, self.query_grid_size), requires_grad=False)

        self.query = Parameter(initializer(TruncatedNormal(mean=0.0, sigma=0.02), [self.num_queries, self.embed_dim]),
                               requires_grad=False)

        if self.kv_dim is not None and self.kv_dim != self.embed_dim:
            self.kv_proj = Linear(in_channels=self.kv_dim, out_channels=self.embed_dim,
                                  has_bias=False)  # TODO: weight init
        else:
            self.kv_proj = nn.Identity()

        self.img_grid_size = config.vision_config.image_size // config.vision_config.patch_size
        self.attn = MultiHeadAttention(hidden_size=self.embed_dim,
                                       num_heads=self.num_heads,
                                       batch_size=None,
                                       src_seq_length=self.num_queries,
                                       tgt_seq_length=self.img_grid_size ** 2,
                                       hidden_dropout_rate=0.0,
                                       attention_dropout_rate=0.0,
                                       softmax_compute_type=config.softmax_compute_type,
                                       use_past=False,
                                       param_init_type=config.param_init_type,
                                       parallel_config=config.parallel_config.dp_mp_config)

        self.ln_q = LayerNorm((self.embed_dim,), eps=1e-6)
        self.ln_kv = LayerNorm((self.embed_dim,), eps=1e-6)

        self.abs_pos = AbsPos(self.pos_embed.shape[0], self.img_grid_size ** 2)

        self.shape = P.Shape()
        self.tile = P.Tile()
        self.add = P.Add()

    def construct(self, x, attn_mask=None):
        # x: [bs, img_grid_size**2, width], [bs, 1024, 1664]
        bs, _, _ = self.shape(x)
        pos_embed = self.abs_pos(self.pos_embed)  # (img_grid_size**2, embed_dim) (1024, 4096)

        x = self.kv_proj(x)  # [bs, img_grid_size**2, embed_dim] [bs, 1024, 4096]
        x = self.ln_kv(x)  # [bs, img_grid_size**2, embed_dim] [bs, 1024, 4096]

        q = self.ln_q(self.query)  # [num_queries, embed_dim] [256, 4096]]

        out = self.attn(
            self.add(self.tile(q.unsqueeze(0), (bs, 1, 1)), self.pos_embed.unsqueeze(0)),  # [bs, 256, 4096]
            self.add(x, pos_embed.unsqueeze(0)),  # [bs, 1024, 4096]
            x,  # [bs, 1024, 4096]
            attention_mask=attn_mask
        )[0]

        return out  # (bs, num_queries, embed_dim)


class VisionTransformer(nn.Cell):
    r"""VisionTransformer Of CLIPModel

    Args:
        input_resolution (int): The image size of input.
        patch_size (int): The patch size of vision transformer.
        width (int): The dimension of vision transformer.
        layers (int): The number of layers of vision transformer.
        heads (int): The number of attention heads.
        output_dim (int): The output dimension of vision transformer.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    # def __init__(self, input_resolution: int, patch_size: int, width: int,
    #              layers: int, heads: int, output_dim: int, dtype: mstype):
    def __init__(self, config: QwenVLConfig):
        super(VisionTransformer, self).__init__()

        input_resolution = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        width = config.vision_config.hidden_size
        layers = config.vision_config.num_hidden_layers
        heads = config.vision_config.num_attention_heads
        output_dim = config.proj_output_dim
        dtype = config.compute_dtype

        self.conv1 = \
            nn.Conv2d(
                in_channels=3, out_channels=width, kernel_size=patch_size,
                stride=patch_size, has_bias=False, pad_mode='pad').to_float(dtype)

        scale = width ** -0.5
        self.positional_embedding = \
            Parameter(scale * Tensor(
                np.random.normal(0, 1, size=(256, width))).astype(ms.float32))
        self.ln_pre = LayerNorm([width], eps=1e-6)
        self.transformer = Transformer(width, layers, heads, config, dtype)

        self.attn_pool = Resampler(config)

        self.ln_post = LayerNorm([output_dim], eps=1e-6)
        self.proj = \
            Parameter(scale * Tensor(np.random.normal(0, 1,
                                                      size=(output_dim, output_dim))).astype(ms.float32))
        self.dtype = dtype
        self.add = P.Add()
        img_grid_size = input_resolution // patch_size
        self.abs_pos = AbsPos(self.positional_embedding.shape[0], img_grid_size ** 2)

    def construct(self, input_x: ms.Tensor):
        r"""Construct

        Args:
            input_x (ms.Tensor): Input tensor.

        Returns:
            input_x (ms.Tensor): Output tensor.
        """
        input_x = self.conv1(input_x)
        input_x = input_x.reshape(input_x.shape[0], input_x.shape[1], -1)
        input_x = input_x.transpose(0, 2, 1)

        abs_pos = self.abs_pos(self.positional_embedding)
        input_x = self.add(input_x, abs_pos)
        input_x = self.ln_pre(input_x)
        input_x = input_x.transpose(1, 0, 2)
        input_x = self.transformer(input_x)
        input_x = input_x.transpose(1, 0, 2)
        input_x = self.attn_pool(input_x)
        input_x = self.ln_post(input_x)
        input_x = ops.matmul(input_x, self.proj)
        return input_x


class MultiheadAttention(nn.Cell):
    r"""MultiheadAttention, With Layers As Input For Initialization

    Args:
        d_model (int): The feature dimension
        n_head (int): The number of attention heads
        layers (int): The number of transformers, used for weight initialization
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, d_model: int, n_head: int, layers: int, dtype: mstype):
        super(MultiheadAttention, self).__init__()

        self.num_heads = n_head
        self.head_dim = d_model // n_head

        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.split = ops.Split(axis=-1, output_num=3)
        self.scaling = self.head_dim ** -0.5

        proj_std = (d_model ** -0.5) * ((2 * layers) ** -0.5)
        attn_std = d_model ** -0.5
        self.out_proj = nn.Dense(d_model, d_model,
                                 weight_init=Normal(mean=0.0, sigma=proj_std)).to_float(dtype)
        self.in_proj = nn.Dense(d_model, 3 * d_model,
                                weight_init=Normal(mean=0.0, sigma=attn_std)).to_float(dtype)

    def construct(self, query: ms.Tensor, attn_mask: Optional[ms.Tensor] = None):
        r"""Construct

        Args:
            query (ms.Tensor): query of attention.
            attn_mask (Optional[ms.Tensor]): attention mask.

        Returns:
            attn_output (ms.Tensor): attention output.
        """
        len_tgt, batch_size, width = query.shape
        qkv = self.in_proj(query)
        qkv = qkv.view(len_tgt, batch_size, self.num_heads, 3 * self.head_dim)

        att_q, att_k, att_v = self.split(qkv)

        att_q = att_q * self.scaling
        att_q = att_q.view(len_tgt, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))
        att_k = att_k.view(-1, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))
        att_v = att_v.view(-1, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        if attn_mask is not None:
            attn_output_weights = attn_mask + ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        else:
            attn_output_weights = ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        attn_output_weights = self.softmax(attn_output_weights)
        attn_output = ops.matmul(attn_output_weights, att_v)
        attn_output = attn_output.view(batch_size, self.num_heads, len_tgt, self.head_dim)
        attn_output = self.transpose(attn_output, (2, 0, 1, 3))
        attn_output = attn_output.view(len_tgt, batch_size, width)
        attn_output = self.out_proj(attn_output)
        return attn_output



class MLP(nn.Cell):
    def __init__(self, d_model, layers, dtype, config):
        super(MLP, self).__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        proj_std = (d_model ** -0.5) * ((2 * layers) ** -0.5)
        fc_std = (2 * d_model) ** -0.5
        mlp_width = int(d_model * config.vision_config.mlp_ratio)
        c_fc = Linear(d_model, mlp_width, weight_init=Normal(mean=0.0, sigma=fc_std)).to_float(dtype)
        c_fc.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.c_fc = c_fc
        c_proj = Linear(mlp_width, d_model, weight_init=Normal(mean=0.0, sigma=proj_std)).to_float(dtype)
        c_proj.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.c_proj = c_proj
        self.gelu = NoApproximateGELU(config.parallel_config)
        self.cast = P.Cast()
        self.gelu_dtype = config.vision_config.gelu_dtype
        self.dtype = P.DType()

    def construct(self, x):
        x = self.c_fc(x)
        ori_dtype = self.dtype(x)
        x = self.cast(x, self.gelu_dtype)
        x = self.gelu(x)
        x = self.cast(x, ori_dtype)
        x = self.c_proj(x)
        return x


class NoApproximateGELU(nn.Cell):
    def __init__(self, parallel_config):
        super().__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.factor = Tensor(np.sqrt(2.0))
        self.div = P.Div().shard(((dp, 1, mp), ()))
        self.erf = P.Erf().shard(((dp, 1, mp),))
        self.dtype = P.DType()
        self.add = P.Add().shard(((dp, 1, mp), ()))
        self.mul_const = P.Mul().shard(((dp, 1, mp), ()))
        self.mul = P.Mul().shard(((dp, 1, mp), (dp, 1, mp)))

    def construct(self, input_x):
        x_dtype = self.dtype(input_x)
        output = self.div(input_x, self.factor.astype(x_dtype))
        output = self.add(self.erf(output), Tensor(1.0, x_dtype))
        output = self.mul_const(self.mul(input_x, output), Tensor(0.5, x_dtype))
        return output

class ResidualAttentionBlock(nn.Cell):
    r"""
    ResidualAttentionBlock of CLIP

    Args:
        d_model (int): The dimension of features.
        n_head (int): The number of attention heads.
        layers (int): The number of transformer layers for weight initialization.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
        attn_mask (Optional[ms.Tensor]): attention mask.
    """

    def __init__(self, d_model: int, n_head: int, layers: int, config: QwenVLConfig,
                 dtype: mstype, attn_mask: Optional[ms.Tensor] = None):
        super(ResidualAttentionBlock, self).__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.dtype = dtype
        self.attn = MultiheadAttention(d_model, n_head, layers, dtype)
        self.ln_1 = LayerNorm((d_model,), eps=1e-6)
        self.ln_1.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.mlp = MLP(d_model, layers, dtype, config)
        self.ln_2 = LayerNorm((d_model,), eps=1e-6)
        self.ln_2.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.attn_mask = attn_mask

    def construct(self, input_x: ms.Tensor):
        r"""Construct"""
        input_x = ops.Add()(input_x, self.attention(self.ln_1(input_x)))
        input_x = ops.Add()(input_x, self.mlp(self.ln_2(input_x)))
        return input_x

    def attention(self, input_x: ms.Tensor):
        r"""Attention"""
        return self.attn(input_x, self.attn_mask)


class Transformer(nn.Cell):
    r"""
    Text Transformer of CLIP

    Args:
        width (int): The dimension of input features.
        layers (int): The number of transformer layers.
        heads (int): The number of attention heads.
        attn_mask (ms.Tensor):  Attention mask.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, width, layers, heads, config, dtype, attn_mask=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[ResidualAttentionBlock(width, heads, layers, config, dtype, attn_mask) for _ in range(layers)]
        )

    def construct(self, input_x):
        r"""Construct"""
        return self.resblocks(input_x)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class QwenVL(BaseModel):
    def __init__(self, config: QwenVLConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        self.vision_encoder = VisionTransformer(config)
        self.llm_model = QwenForCausalLM(config.text_config)

        self.image_start_id = self.config.image_start_id
        self.image_pad_id = self.config.image_pad_id

        self.num_queries = self.config.num_queries
        self.image_embeds_element_size = self.num_queries * config.text_config.hidden_size

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.ones_like = P.OnesLike()
        self.cast = P.Cast()
        self.not_equal = P.NotEqual()
        self.slice = P.StridedSlice()
        self.masked_fill = P.MaskedFill()
        self.tensor_scatter_update = ops.TensorScatterUpdate()
        self.gather = P.Gather().shard(((1, 1, 1), ()))

        self.freeze_component()

        self.is_first_iteration = True
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.ignore_token_id = ms.Tensor(config.ignore_token_id, mstype.int32)
        self.use_past = config.use_past

        batch_size = config.text_config.batch_size
        micro_batch_interleave_num = config.micro_batch_interleave_num
        if _get_parallel_mode() in ["semi_auto_parallel", "auto_parallel"]:
            full_batch = ms.get_auto_parallel_context("full_batch")
            if full_batch:
                self.batch_size = batch_size * config.parallel_config.data_parallel * micro_batch_interleave_num
            else:
                card_num = int(os.getenv('RANK_SIZE', '1'))
                self.batch_size = int(card_num * batch_size / micro_batch_interleave_num)
        else:
            self.batch_size = batch_size
        self.batch_index_adder = ms.Tensor([[i, 0] for i in range(self.batch_size)], ms.int32).reshape(self.batch_size, 1, 1, 2)
        self.img_pos_add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def freeze_component(self):
        if self.config.freeze_vision:
            logger.info("freeze vision encoder")
            for param in self.vision_encoder.trainable_params():
                if not self.config.freeze_resampler and "vision_encoder.attn_pool" in param.name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if self.config.freeze_llm:
            logger.info("freeze llm model")
            for param in self.llm_model.trainable_params():
                param.requires_grad = False

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        input_position = kwargs.get("current_index", None)
        img_pos = kwargs.get("img_pos", None)

        if input_position is not None:
            input_position = ms.Tensor(input_position, mstype.int32)

        if self.is_first_iteration or not self.use_past:
            images = kwargs.pop("images")
        else:
            images = None

        if img_pos is not None:
            img_pos = ms.Tensor(img_pos, ms.int32)

        return {
            "input_ids": ms.Tensor(input_ids, mstype.int32),
            "images": images,
            "input_position": input_position,
            "img_pos": img_pos
        }

    def concat_image_text(self, text_embeds, image_embeds, img_pos, is_multi_img=False):
        img_pos = self.img_pos_add(img_pos, self.batch_index_adder).reshape((-1,) + img_pos.shape[-2:])
        image_embeds = self.cast(image_embeds, text_embeds.dtype)
        text_embeds = self.tensor_scatter_update(text_embeds, img_pos, image_embeds)
        return text_embeds

    def generate_index(self, start):
        indices = self.base_index.copy()
        indices[:, 0] = indices[:, 0] + start
        return indices

    def construct(self, input_ids, images, img_pos: Tensor = None, labels=None,
                  input_position=None, position_ids=None, attention_mask=None, init_reset=True, batch_valid_length=None,
                  batch_index=None, zactivate_len=None):

        bs, seq_len = self.shape(input_ids)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
            if labels is None:
                labels = self.masked_fill(input_ids, input_ids == self.pad_token_id, self.ignore_token_id)
                labels = self.masked_fill(labels, labels == self.image_pad_id, self.ignore_token_id)
        else:
            tokens = input_ids

        input_embeds = self.llm_model.to_embeddings(tokens)
        input_attn_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)

        if images is not None:
            is_multi_img = False
            if images.ndim == 5:
                images_shape = self.shape(images)
                new_shape = (images_shape[0] * images_shape[1], images_shape[2], images_shape[3], images_shape[4])
                images = self.reshape(images, new_shape)
                is_multi_img = True

            image_embeds = self.vision_encoder(images)
            input_embeds = self.concat_image_text(input_embeds, image_embeds, img_pos, is_multi_img=is_multi_img)

        return self.llm_model(
            input_embeds=input_embeds,
            attention_mask=input_attn_mask,
            labels=labels,
            input_position=input_position,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length
        )
