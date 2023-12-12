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

"""test pipeline registry"""
import unittest
from mindformers.pipeline.pipeline_registry import PipelineRegistry
from mindformers.auto_class import AutoModel


class TestPipelineRegistry(unittest.TestCase):
    """
    Test the registration function of the pipeline.
    """
    def setUp(self):
        self.supported_tasks = {
            "image_classification":
                {"impl": "ImageClassificationPipeline",
                 "ms": "AutoModel",
                 "default": {"model": ()},
                 "type": "image"
                 },
            "text_classification":
                {"impl": "TextClassificationPipeline",
                 "ms": "AutoModel",
                 "default": {"model": ()},
                 "type": "text"
                 },
            "translation":
                {"impl": "TranslationPipeline",
                 "ms": "AutoModel",
                 "default": {"model": ()},
                 "type": "text"
                 },
        }

        self.task_aliases = {"alias1": "image_classification", "alias2": "text_classification"}
        self.registry = PipelineRegistry(supported_tasks=self.supported_tasks, task_aliases=self.task_aliases)

    def test_get_supported_tasks(self):
        """test get_supported_tasks function"""
        expected_tasks = ['alias1', 'alias2', 'image_classification', 'text_classification', "translation"]
        self.assertEqual(self.registry.get_supported_tasks(), expected_tasks)

    def test_check_task(self):
        """test check_task() function"""
        task1 = "image_classification"
        expected_task1 = ("image_classification", self.supported_tasks["image_classification"], None)
        self.assertEqual(self.registry.check_task(task1), expected_task1)

        task2 = "text_classification"
        expected_task2 = ("text_classification", self.supported_tasks["text_classification"], None)
        self.assertEqual(self.registry.check_task(task2), expected_task2)

        task3 = "translation_en_to_fr"
        expected_task3 = ("translation", self.supported_tasks["translation"], ("en", "fr"))
        self.assertEqual(self.registry.check_task(task3), expected_task3)

        task4 = "invalid_task"
        with self.assertRaises(KeyError):
            self.registry.check_task(task4)

    def test_register_pipeline(self):
        """test register_pipeline() function"""
        class TextPipeline:
            pass

        self.registry.register_pipeline("test_text", TextPipeline, AutoModel, {"model": ()}, "text")

        expected_tasks = ['alias1', 'alias2', 'image_classification', 'test_text', 'text_classification', "translation"]
        self.assertEqual(self.registry.get_supported_tasks(), expected_tasks)

        expected_task3 = {
            "impl": TextPipeline,
            "ms": AutoModel,
            "default": {"model": ()},
            "type": "text"
        }
        self.assertEqual(self.registry.supported_tasks["test_text"], expected_task3)
        