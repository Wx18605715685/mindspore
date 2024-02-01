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
"""test AutoConfig"""
import os
os.environ["MODELFOUNDRY_HUB_ENDPOINT"] = "https://giteash.test.osinfra.cn/"
import tempfile
import unittest
from mindformers import AutoConfig
from mindformers.models import GPT2Config

repo_id = "mindformersinfra/test_auto_config_ms"
num_layers = 4

dynamic_repo_id = "mindformersinfra/test_dynamic_config"
dynamic_class_name = "Baichuan2Config"

class TestPretrainedConfig(unittest.TestCase):
    """test PretrainedConfig"""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name
        self.config_path = self.path + "/config.json"

    def test_download_from_repo(self):
        """Test init config by from_pretrained('usr_name/repo_name')"""
        config = AutoConfig.from_pretrained(repo_id)
        self.assertEqual(config.num_layers, num_layers)
        self.assertTrue(isinstance(config, GPT2Config))

    def test_save_and_read(self):
        """Test save config by save_pretrained() and read it by from_pretrained()"""
        config = GPT2Config(num_layers=num_layers)

        config.save_pretrained(self.path)

        config = AutoConfig.from_pretrained(self.path)
        print(type(config))
        self.assertEqual(config.num_layers, num_layers)
        self.assertTrue(isinstance(config, GPT2Config))

    def test_dynamic_config(self):
        """Test init dynamic config by from_pretrained('usr_name/repo_name')"""
        model = AutoConfig.from_pretrained(dynamic_repo_id, trust_remote_code=True)
        self.assertEqual(model.__class__.__name__, dynamic_class_name)
