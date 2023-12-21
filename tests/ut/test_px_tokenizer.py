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
"""px tokenizer ut"""

import os
from mindformers.models.auto import AutoTokenizer
from mindformers import GPT2TokenizerFast, GPT2Tokenizer

def test_px_tokenizer():
    """

        Feature: px_tokenizer

        Description: Test px_tokenizer

        Expectation: No Error

    """
    short_string = "I love Beijing, because I'm a Chinese."

    short_string_ids = [40, 1842, 11618, 11, 780, 314, 1101, 257, 3999, 13]

    special_token = "hahahaha"
    special_token_ids = [71, 36225, 12236]
    special_token_ids_spe = [50257]

    slow_tokenizer = AutoTokenizer.from_pretrained("zyw-hw/tokenizer_test")
    fast_tokenizer = AutoTokenizer.from_pretrained("zyw-hw/tokenizer_test", use_fast=True)

    assert slow_tokenizer(short_string)["input_ids"] == short_string_ids
    assert fast_tokenizer(short_string)["input_ids"] == short_string_ids

    assert slow_tokenizer.encode(short_string) == short_string_ids
    assert fast_tokenizer.encode(short_string) == short_string_ids

    assert slow_tokenizer.decode(short_string_ids, skip_special_tokens=True) == short_string
    assert fast_tokenizer.decode(short_string_ids, skip_special_tokens=True) == short_string

    if not os.path.exists("./slow_tokenizer_pretrained"):
        os.mkdir("./slow_tokenizer_pretrained")

    if not os.path.exists("./slow_tokenizer_vocabulary"):
        os.mkdir("./slow_tokenizer_vocabulary")

    if not os.path.exists("./fast_tokenizer_pretrained"):
        os.mkdir("./fast_tokenizer_pretrained")

    if not os.path.exists("./fast_tokenizer_vocabulary"):
        os.mkdir("./fast_tokenizer_vocabulary")

    slow_tokenizer.save_pretrained("./slow_tokenizer_pretrained")
    slow_tokenizer.save_vocabulary("./slow_tokenizer_vocabulary")

    fast_tokenizer.save_pretrained("./fast_tokenizer_pretrained")
    fast_tokenizer.save_vocabulary("./fast_tokenizer_vocabulary")

    slow_tokenizer = GPT2Tokenizer.from_pretrained("./slow_tokenizer_vocabulary")
    fast_tokenizer = GPT2TokenizerFast.from_pretrained("./fast_tokenizer_vocabulary")

    assert slow_tokenizer(short_string)["input_ids"] == short_string_ids
    assert fast_tokenizer(short_string)["input_ids"] == short_string_ids

    assert slow_tokenizer.encode(short_string) == short_string_ids
    assert fast_tokenizer.encode(short_string) == short_string_ids

    assert slow_tokenizer.decode(short_string_ids, skip_special_tokens=True) == short_string
    assert fast_tokenizer.decode(short_string_ids, skip_special_tokens=True) == short_string

    assert slow_tokenizer(special_token)["input_ids"] == special_token_ids
    assert slow_tokenizer.encode(special_token) == special_token_ids
    assert slow_tokenizer.decode(special_token_ids) == special_token
    slow_tokenizer.add_tokens(special_token, special_tokens=True)
    assert slow_tokenizer(special_token)["input_ids"] == special_token_ids_spe
    assert slow_tokenizer.encode(special_token) == special_token_ids_spe
    assert slow_tokenizer.decode(special_token_ids_spe) == special_token

    assert fast_tokenizer(special_token)["input_ids"] == special_token_ids
    assert fast_tokenizer.encode(special_token) == special_token_ids
    assert fast_tokenizer.decode(special_token_ids) == special_token
    fast_tokenizer.add_tokens(special_token, special_tokens=True)
    assert fast_tokenizer(special_token)["input_ids"] == special_token_ids_spe
    assert fast_tokenizer.encode(special_token) == special_token_ids_spe
    assert fast_tokenizer.decode(special_token_ids_spe) == special_token

    # test 外挂
    tokenizer = AutoTokenizer.from_pretrained("zyw-hw/baichuan2", use_fast=False, trust_remote_code=True)

    res = [92346, 2950, 19801, 92323, 2201, 1406, 92404, 92326, 1346, 6840, 72]

    assert tokenizer(short_string)["input_ids"] == res
    assert tokenizer.encode(short_string) == res
    assert tokenizer.decode(res) == short_string

    if not os.path.exists("./tokenizer_pretrained"):
        os.mkdir("./tokenizer_pretrained")

    if not os.path.exists("./tokenizer_vocabulary"):
        os.mkdir("./tokenizer_vocabulary")

    tokenizer.save_pretrained("./tokenizer_pretrained")
    tokenizer.save_vocabulary("./tokenizer_vocabulary")
