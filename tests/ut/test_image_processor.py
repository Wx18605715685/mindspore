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
""" test image processor """
import os
import shutil
from mindformers.models.auto import AutoImageProcessor
from mindformers.models.blip2 import Blip2ImageProcessor
from mindformers.tools.image_tools import load_image

REMOTE_PATH = "mf-ut/px_image_processor_test"
OUTPUT_PATH = "./test_image_processors"
OUTPUT_SUM = -54167.402
DIFF_THRESHOLD = 1e-7

def test_blip2_image_processor():
    """
    Feature: Test BLIP2 Image Processor API
    Description: Test BLIP2 Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
                       "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

    blip2_processor = Blip2ImageProcessor(image_size=224)
    result = blip2_processor(image)
    diff = abs(result.sum() - OUTPUT_SUM)
    assert result.shape == (1, 3, 224, 224)
    assert diff < DIFF_THRESHOLD
    blip2_processor.save_pretrained(OUTPUT_PATH)

    blip2_processor = Blip2ImageProcessor.from_pretrained(OUTPUT_PATH)
    result = blip2_processor(image)
    diff = abs(result.sum() - OUTPUT_SUM)
    assert result.shape == (1, 3, 224, 224)
    assert diff < DIFF_THRESHOLD

    blip2_processor = Blip2ImageProcessor.from_pretrained(REMOTE_PATH)
    result = blip2_processor(image)
    diff = abs(result.sum() - OUTPUT_SUM)
    assert result.shape == (1, 3, 224, 224)
    assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)


def test_auto_image_processor():
    """
    Feature: Test Auto Image Processor API
    Description: Test Auto Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
                       "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

    auto_processor = AutoImageProcessor.from_pretrained(REMOTE_PATH)
    result = auto_processor(image)
    diff = abs(result.sum() - OUTPUT_SUM)
    assert result.shape == (1, 3, 224, 224)
    assert diff < DIFF_THRESHOLD
    auto_processor.save_pretrained(OUTPUT_PATH)

    auto_processor = AutoImageProcessor.from_pretrained(OUTPUT_PATH)
    result = auto_processor(image)
    diff = abs(result.sum() - OUTPUT_SUM)
    assert result.shape == (1, 3, 224, 224)
    assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
