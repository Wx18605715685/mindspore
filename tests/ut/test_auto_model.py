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
"""test AutoModel using GPT2Model"""

import os
import shutil
import unittest
import numpy as np
from mindformers.models.auto.configuration_auto import AutoConfig
from mindformers.models.auto.modeling_auto import AutoModelForCausalLM


def compare_state_dict(state_dict1, state_dict2):
    assert state_dict1.keys() == state_dict2.keys()
    for key in state_dict1:
        value1 = state_dict1[key].asnumpy()
        value2 = state_dict1[key].asnumpy()
        assert np.array_equal(value1, value2), f"The value of {key} is not match!"


class TestAutoModel(unittest.TestCase):
    '''test AutoModel'''
    def test_automodel_load_from_repo(self):
        '''test AutoModel load from repo'''
        gpt2model_from_repo = "mf-ut/pretrianedmodel_ut"
        gpt2model_from_local = "./GPT2Model_ut"

        model_1 = AutoModelForCausalLM.from_pretrained(gpt2model_from_repo)
        model_1.save_pretrained(gpt2model_from_local)

        file_names = os.listdir(gpt2model_from_local)
        assert "config.json" in file_names, "config.json not found!"
        assert "mindspore_model.ckpt" in file_names, "mindspore_model.ckpt not found!"
        shutil.rmtree(gpt2model_from_local)

    def test_automodel_load_from_local(self):
        '''test AutoModel load from local'''
        gpt2model_from_repo = "mf-ut/pretrianedmodel_ut"
        gpt2model_from_local = "./GPT2Model_ut"

        model_1 = AutoModelForCausalLM.from_pretrained(gpt2model_from_repo)  # load from repo
        model_1.save_pretrained(gpt2model_from_local)
        state_dict_model_1 = {}
        for item in model_1.get_parameters():
            state_dict_model_1[item.name] = item.data

        model_2 = AutoModelForCausalLM.from_pretrained(gpt2model_from_local)  # load from local path
        state_dict_model_2 = {}
        for item in model_2.get_parameters():
            state_dict_model_2[item.name] = item.data

        compare_state_dict(state_dict_model_1, state_dict_model_2)
        shutil.rmtree(gpt2model_from_local)

    def test_automodel_load_from_config(self):
        '''test AutoModel load from config'''
        gpt2model_from_repo = "mf-ut/pretrianedmodel_ut"

        config_1 = AutoConfig.from_pretrained(gpt2model_from_repo)
        model_1 = AutoModelForCausalLM.from_config(config_1)
        state_dict_model_1 = {}
        for item in model_1.get_parameters():
            state_dict_model_1[item.name] = item.data

        model_2 = AutoModelForCausalLM.from_pretrained(gpt2model_from_repo)
        state_dict_model_2 = {}
        for item in model_2.get_parameters():
            state_dict_model_2[item.name] = item.data

        compare_state_dict(state_dict_model_1, state_dict_model_2)
