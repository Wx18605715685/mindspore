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
"""
Test module for testing pipeline interface.
How to run this:
pytest tests/st/test_pipeline/test_px_pipeline.py
"""
import mindspore as ms

from mindformers import TextGenerationPipeline, pipeline
from mindformers.models.auto.modeling_auto import AutoModelForCausalLM
from mindformers.models.auto.tokenization_auto import AutoTokenizer

ms.set_context(mode=0)


class TestPipeline:
    """A test class for testing pipeline features."""
    def setup_method(self):
        """setup method."""
        self.task_name = "text_generation"
        self.model_name = "senzhen/mindspore_gpt2"
        self.use_past = True

    def test_text_generation(self):
        """
        Feature: text_generation pipeline.
        Description: Test generate for text_generation pipeline.
        Expectation: TypeError, ValueError, RuntimeError
        """
        input_text = "An increasing sequence: one,"
        model = AutoModelForCausalLM.from_pretrained(self.model_name, use_past=self.use_past)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        task_pipeline = TextGenerationPipeline(model, tokenizer)
        res = task_pipeline(input_text, max_new_tokens=20, do_sample=False)
        print(res)

    def test_pipeline(self):
        """
        Feature: pipeline interface.
        Description: Test generate for pipeline interface.
        Expectation: TypeError, ValueError, RuntimeError
        """
        input_text = "An increasing sequence: one,"
        model_kwargs = {"use_past": self.use_past}
        task_pipeline = pipeline(self.task_name,
                                 self.model_name,
                                 model_kwargs=model_kwargs)
        res = task_pipeline(input_text, max_new_tokens=20, do_sample=False)
        print(res)
