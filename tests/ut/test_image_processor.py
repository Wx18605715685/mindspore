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
import shutil
import os

os.environ["MODELFOUNDRY_HUB_ENDPOINT"] = "https://giteash.test.osinfra.cn/"

# pylint: disable=C0413
import numpy as np
# pylint: disable=C0413
from mindformers.models.auto import AutoImageProcessor
# pylint: disable=C0413
from mindformers.models.blip2 import Blip2ImageProcessor
# pylint: disable=C0413
from mindformers.models.mae import ViTMAEImageProcessor
# pylint: disable=C0413
from mindformers.models.clip import CLIPImageProcessor
# pylint: disable=C0413
from mindformers.models.swin import SwinImageProcessor
# pylint: disable=C0413
from mindformers.models.vit import ViTImageProcessor
# pylint: disable=C0413
from mindformers.models.sam import SAMImageProcessor
# pylint: disable=C0413
from mindformers.tools.image_tools import load_image


TEST_IMAGE_URL = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png"

BLIP2_REMOTE_PATH = "mindformersinfra/test_blip2_image_processor"
MAE_REMOTE_PATH = "mindformersinfra/test_mae_image_processor"
CLIP_REMOTE_PATH = "mindformersinfra/test_clip_image_processor"
SWIN_REMOTE_PATH = "mindformersinfra/test_swin_image_processor"
VIT_REMOTE_PATH = "mindformersinfra/test_vit_image_processor"
SAM_REMOTE_PATH = "mindformersinfra/test_sam_image_processor"

BLIP2_OUTPUT_SHAPE = (1, 3, 224, 224)
MAE_OUTPUT_SHAPE = [(1, 3, 224, 224), (1, 196), (1, 196), (1, 49)]
CLIP_OUTPUT_SHAPE = (1, 3, 224, 224)
SWIN_OUTPUT_SHAPE = (1, 3, 224, 224)
VIT_OUTPUT_SHAPE = (1, 3, 224, 224)
SAM_OUTPUT_SHAPE = [(1, 3, 1024, 1024), (773, 1024)]

BLIP2_OUTPUT_SUM = -54167.402
MAE_OUTPUT_SUM = [-64566.43, 147, 19110]
CLIP_OUTPUT_SUM = -40270.586
SWIN_OUTPUT_SUM = -26959.375
VIT_OUTPUT_SUM = -26959.375
SAM_OUTPUT_SUM = -1019078.3

OUTPUT_PATH = "./test_image_processors"
DIFF_THRESHOLD = 0.3


def test_blip2_image_processor():
    """
    Feature: Test BLIP2 Image Processor API
    Description: Test BLIP2 Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image(TEST_IMAGE_URL)

    blip2_processor = Blip2ImageProcessor(image_size=224)
    result = blip2_processor(image)
    diff = abs(result.sum() - BLIP2_OUTPUT_SUM)
    assert result.shape == BLIP2_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD
    blip2_processor.save_pretrained(OUTPUT_PATH)

    blip2_processor = Blip2ImageProcessor.from_pretrained(OUTPUT_PATH)
    result = blip2_processor(image)
    diff = abs(result.sum() - BLIP2_OUTPUT_SUM)
    assert result.shape == BLIP2_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    blip2_processor = Blip2ImageProcessor.from_pretrained(BLIP2_REMOTE_PATH)
    result = blip2_processor(image)
    diff = abs(result.sum() - BLIP2_OUTPUT_SUM)
    assert result.shape == BLIP2_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)


def test_mae_image_processor():
    """
    Feature: Test MAE Image Processor API
    Description: Test MAE Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image(TEST_IMAGE_URL)

    mae_processor = ViTMAEImageProcessor()
    result = mae_processor(image)
    for i in range(len(result)):
        assert result[i].shape == MAE_OUTPUT_SHAPE[i]
        if i < len(result)-1:
            diff = abs(result[i].sum() - MAE_OUTPUT_SUM[i])
            assert diff < DIFF_THRESHOLD
    mae_processor.save_pretrained(OUTPUT_PATH)

    mae_processor = ViTMAEImageProcessor.from_pretrained(OUTPUT_PATH)
    result = mae_processor(image)
    for i in range(len(result)):
        assert result[i].shape == MAE_OUTPUT_SHAPE[i]
        if i < len(result)-1:
            diff = abs(result[i].sum() - MAE_OUTPUT_SUM[i])
            assert diff < DIFF_THRESHOLD

    mae_processor = ViTMAEImageProcessor.from_pretrained(MAE_REMOTE_PATH)
    result = mae_processor(image)
    for i in range(len(result)):
        assert result[i].shape == MAE_OUTPUT_SHAPE[i]
        if i < len(result)-1:
            diff = abs(result[i].sum() - MAE_OUTPUT_SUM[i])
            assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)


def test_clip_image_processor():
    """
    Feature: Test CLIP Image Processor API
    Description: Test CLIP Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image(TEST_IMAGE_URL)

    clip_processor = CLIPImageProcessor()
    result = clip_processor(image)
    diff = abs(result.sum() - CLIP_OUTPUT_SUM)
    assert result.shape == CLIP_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD
    clip_processor.save_pretrained(OUTPUT_PATH)

    clip_processor = CLIPImageProcessor.from_pretrained(OUTPUT_PATH)
    result = clip_processor(image)
    diff = abs(result.sum() - CLIP_OUTPUT_SUM)
    assert result.shape == CLIP_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    clip_processor = CLIPImageProcessor.from_pretrained(CLIP_REMOTE_PATH)
    result = clip_processor(image)
    diff = abs(result.sum() - CLIP_OUTPUT_SUM)
    assert result.shape == CLIP_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)


def test_swin_image_processor():
    """
    Feature: Test Swin Image Processor API
    Description: Test Swin Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image(TEST_IMAGE_URL)

    swin_processor = SwinImageProcessor()
    result = swin_processor(image)
    diff = abs(result.sum() - SWIN_OUTPUT_SUM)
    assert result.shape == SWIN_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD
    swin_processor.save_pretrained(OUTPUT_PATH)

    swin_processor = SwinImageProcessor.from_pretrained(OUTPUT_PATH)
    result = swin_processor(image)
    diff = abs(result.sum() - SWIN_OUTPUT_SUM)
    assert result.shape == SWIN_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    swin_processor = SwinImageProcessor.from_pretrained(SWIN_REMOTE_PATH)
    result = swin_processor(image)
    diff = abs(result.sum() - SWIN_OUTPUT_SUM)
    assert result.shape == SWIN_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)


def test_vit_image_processor():
    """
    Feature: Test VIT Image Processor API
    Description: Test VIT Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image(TEST_IMAGE_URL)

    vit_processor = ViTImageProcessor()
    result = vit_processor(image)
    diff = abs(result.sum() - VIT_OUTPUT_SUM)
    assert result.shape == VIT_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD
    vit_processor.save_pretrained(OUTPUT_PATH)

    vit_processor = ViTImageProcessor.from_pretrained(OUTPUT_PATH)
    result = vit_processor(image)
    diff = abs(result.sum() - VIT_OUTPUT_SUM)
    assert result.shape == VIT_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    vit_processor = ViTImageProcessor.from_pretrained(VIT_REMOTE_PATH)
    result = vit_processor(image)
    diff = abs(result.sum() - VIT_OUTPUT_SUM)
    assert result.shape == VIT_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)


def test_sam_image_processor():
    """
    Feature: Test SAM Image Processor API
    Description: Test SAM Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image(TEST_IMAGE_URL)

    sam_processor = SAMImageProcessor()
    result = sam_processor(np.array(image))
    diff = abs(result[0].sum() - SAM_OUTPUT_SUM)
    assert result[0].shape == SAM_OUTPUT_SHAPE[0]
    assert result[1] == SAM_OUTPUT_SHAPE[1]
    assert diff < DIFF_THRESHOLD
    sam_processor.save_pretrained(OUTPUT_PATH)

    sam_processor = SAMImageProcessor.from_pretrained(OUTPUT_PATH)
    result = sam_processor(np.array(image))
    diff = abs(result[0].sum() - SAM_OUTPUT_SUM)
    assert result[0].shape == SAM_OUTPUT_SHAPE[0]
    assert result[1] == SAM_OUTPUT_SHAPE[1]
    assert diff < DIFF_THRESHOLD

    sam_processor = SAMImageProcessor.from_pretrained(SAM_REMOTE_PATH)
    result = sam_processor(np.array(image))
    diff = abs(result[0].sum() - SAM_OUTPUT_SUM)
    assert result[0].shape == SAM_OUTPUT_SHAPE[0]
    assert result[1] == SAM_OUTPUT_SHAPE[1]
    assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)


def test_auto_image_processor():
    """
    Feature: Test Auto Image Processor API
    Description: Test Auto Image Processor functions, including init, save_pretrained, from_pretrained, preprocess.
    Expectation: No exception
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = load_image(TEST_IMAGE_URL)

    auto_processor = AutoImageProcessor.from_pretrained(BLIP2_REMOTE_PATH)
    result = auto_processor(image)
    diff = abs(result.sum() - BLIP2_OUTPUT_SUM)
    assert result.shape == BLIP2_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD
    auto_processor.save_pretrained(OUTPUT_PATH)

    auto_processor = AutoImageProcessor.from_pretrained(OUTPUT_PATH)
    result = auto_processor(image)
    diff = abs(result.sum() - BLIP2_OUTPUT_SUM)
    assert result.shape == BLIP2_OUTPUT_SHAPE
    assert diff < DIFF_THRESHOLD

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
