# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generic utilities
"""
import tempfile

from contextlib import contextmanager

@contextmanager
def working_or_temp_dir(working_dir, use_temp_dir: bool = False):
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        yield working_dir

def add_model_info_to_auto_map(auto_map, repo_id):
    """
    Adds the information of the repo_id to a given auto map.
    """
    for key, value in auto_map.items():
        if isinstance(value, (tuple, list)):
            auto_map[key] = [f"{repo_id}--{v}" if (v is not None and "--" not in v) else v for v in value]
        elif value is not None and "--" not in value:
            auto_map[key] = f"{repo_id}--{value}"

    return auto_map
