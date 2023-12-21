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
"""test PretrainedModel using GPT2LMHeadModel"""
import os
import shutil
import unittest
import numpy as np

from mindformers import GPT2LMHeadModel

class TestPretrainedModel(unittest.TestCase):
    """test PretrainedModel"""
    def test_load_and_save(self):
        """test load and save GPT2LMHeadModel"""
        model_1 = GPT2LMHeadModel.from_pretrained("mf-ut/pretrianedmodel_ut")
        state_dict_1 = {}
        for item in model_1.get_parameters():
            state_dict_1[item.name] = item.data

        shard_model_save_dir = './test_pretrained_model_gpt2'
        os.makedirs(shard_model_save_dir)
        model_1.save_pretrained(shard_model_save_dir)
        file_names = os.listdir(shard_model_save_dir)
        assert "config.json" in file_names, "config.json not found!"
        assert "mindspore_model.ckpt" in file_names, "mindspore_model.ckpt not found!"

        model_2 = GPT2LMHeadModel.from_pretrained(shard_model_save_dir)
        state_dict_2 = {}
        for item in model_2.get_parameters():
            state_dict_2[item.name] = item.data

        self.compare_state_dict(state_dict_1, state_dict_2)

        shutil.rmtree(shard_model_save_dir)

    def test_load_and_save_shard(self):
        """test load and save GPT2LMHeadModel with shard"""
        model_1 = GPT2LMHeadModel.from_pretrained("mf-ut/pretrianedmodel_ut")
        state_dict_1 = {}
        for item in model_1.get_parameters():
            state_dict_1[item.name] = item.data

        shard_model_save_dir = './test_pretrained_model_gpt2_shard'
        os.makedirs(shard_model_save_dir)
        model_1.save_pretrained(shard_model_save_dir, max_shard_size="100MB")
        file_names = os.listdir(shard_model_save_dir)
        assert "config.json" in file_names, "config.json not found!"
        assert "mindspore_model.ckpt.index.json" in file_names, "mindspore_model.ckpt.index.json not found!"
        assert "mindspore_model-00001-of-00005.ckpt" in file_names, "mindspore_model-00001-of-00005.ckpt not found!"
        assert "mindspore_model-00002-of-00005.ckpt" in file_names, "mindspore_model-00002-of-00005.ckpt not found!"
        assert "mindspore_model-00003-of-00005.ckpt" in file_names, "mindspore_model-00003-of-00005.ckpt not found!"
        assert "mindspore_model-00004-of-00005.ckpt" in file_names, "mindspore_model-00004-of-00005.ckpt not found!"
        assert "mindspore_model-00005-of-00005.ckpt" in file_names, "mindspore_model-00005-of-00005.ckpt not found!"

        model_2, loading_info = GPT2LMHeadModel.from_pretrained(shard_model_save_dir, output_loading_info=True)
        state_dict_2 = {}
        for item in model_2.get_parameters():
            state_dict_2[item.name] = item.data
        self.compare_state_dict(state_dict_1, state_dict_2)
        assert "missing_keys" in loading_info, "missing_keys not found!"
        assert "unexpected_keys" in loading_info, "unexpected_keys not found!"
        assert "mismatched_keys" in loading_info, "mismatched_keys not found!"

        shutil.rmtree(shard_model_save_dir)

    def compare_state_dict(self, state_dict1, state_dict2):
        assert state_dict1.keys() == state_dict2.keys()
        for key in state_dict1:
            value1 = state_dict1[key].asnumpy()
            value2 = state_dict1[key].asnumpy()
            assert np.array_equal(value1, value2), f"The value of {key} is not match!"
