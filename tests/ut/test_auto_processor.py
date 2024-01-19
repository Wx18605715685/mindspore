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
""" test autoprocessor """
import os
import shutil
import unittest
from mindformers import AutoProcessor
from mindformers import Blip2Processor
from mindformers import Blip2ImageProcessor
from mindformers import BertTokenizerFast
from mindformers.tools.image_tools import load_image

os.environ["MODELFOUNDRY_HUB_ENDPOINT"] = "https://giteash.test.osinfra.cn/"
cache_dir = "../mindformersinfra/test_auto_processor_ms"
save_dir = "test_auto_processor_ms"
repo_id = "mindformersinfra/test_auto_processor_ms"
test_img = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png"


class TestAutoProcessor(unittest.TestCase):
    """test auto_processor"""

    def test_save_and_load(self):
        """test from_pretrained() and save_pretarined()"""
        processor_repo = AutoProcessor.from_pretrained(repo_id, cache_dir=cache_dir)
        assert isinstance(processor_repo, Blip2Processor)
        assert isinstance(processor_repo.image_processor, Blip2ImageProcessor)
        assert isinstance(processor_repo.tokenizer, BertTokenizerFast)

        processor_repo.save_pretrained(save_directory=save_dir)

        processor_local = AutoProcessor.from_pretrained(save_dir)
        assert isinstance(processor_local, Blip2Processor)
        assert isinstance(processor_local.image_processor, Blip2ImageProcessor)
        assert isinstance(processor_local.tokenizer, BertTokenizerFast)

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test_processor(self):
        """test auto processor processing data"""
        image = load_image(test_img)
        text = ["a boy", "a girl"]
        auto_processor = AutoProcessor.from_pretrained(repo_id, cache_dir=cache_dir)
        auto_processor(image, text)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
