# Copyright 2018 The HuggingFace Inc. team.
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

"""Constant Declaration of Pipeline Registry"""
from mindformers.auto_class import AutoModel
from .image_classification_pipeline import ImageClassificationPipeline
from .zero_shot_image_classification_pipeline import ZeroShotImageClassificationPipeline
from .image_to_text_generation_pipeline import ImageToTextGenerationPipeline
from .translation_pipeline import TranslationPipeline
from .fill_mask_pipeline import FillMaskPipeline
from .text_classification_pipeline import TextClassificationPipeline
from .token_classification_pipeline import TokenClassificationPipeline
from .question_answering_pipeline import QuestionAnsweringPipeline
from .text_generation_pipeline import TextGenerationPipeline
from .masked_image_modeling_pipeline import MaskedImageModelingPipeline
from .segment_anything_pipeline import SegmentAnythingPipeline
from .pipeline_registry import PipelineRegistry


TASK_ALIASES = {
    "text_classification": "text-classification",
    "sentiment_analysis": "text-classification",
    "ner": "token-classification",
    "fill_mask": "fill-mask",
    "image_classification": "image-classification",
    "image_to_text_generation": "image-to-text-generation",
    "masked_image_modeling": "masked-image-modeling",
    "question_answering": "question-answering",
    "segment_anything": "segment-anything",
    "text_generation": "text-generation",
    "token_classification": "token-classification",
    "zero_shot_image_classification": "zero-shot-image-classification",
}

SUPPORTED_TASKS = {
    "fill-mask": {
        "impl": FillMaskPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "text",
    },
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "image-to-text-generation": {
        "impl": ImageToTextGenerationPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "masked-image-modeling": {
        "impl": MaskedImageModelingPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "segment-anything": {
        "impl": SegmentAnythingPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "text-classification": {
        "impl": TextClassificationPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "translation": {
        "impl": TranslationPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
    "zero-shot-image-classification": {
        "impl": ZeroShotImageClassificationPipeline,
        "ms": (AutoModel,),
        "default": {"model": {"ms": ()}},
        "type": "image",
    },
}

PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)
