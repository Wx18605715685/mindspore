# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023, All rights reserved.
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contain helper class to retrieve/store token from/to local cache."""
import os
from pathlib import Path
from typing import Optional

from ..constant import MDS_TOKEN_PATH


class MdsFolder:
    path_token = Path(MDS_TOKEN_PATH)

    @classmethod
    def save_token(cls, token: str) -> None:
        """
        Save token, creating folder as needed.

        Token is saved in the mindseed home folder. You can configure it by setting
        the `MDS_HOME` environment variable.

        Args:
            token (`str`):
                The token to save to the [`MdsFolder`]
        """
        cls.path_token.parent.mkdir(parents=True, exist_ok=True)
        cls.path_token.write_text(token)

    @classmethod
    def get_token(cls) -> Optional[str]:
        """
        Get token or None if not existent.

        A token can be also provided using the `MDS_TOKEN` environment variable.

        Token is saved in the mindseed home folder. You can configure it by setting
        the `MDS_HOME` environment variable. Previous location was `~/.mindseeed/token`.

        Returns:
            `str` or `None`: The token, `None` if it doesn't exist.
        """
        # 1. Is it set by environment variable ?
        token: Optional[str] = os.environ.get("MDS_TOKEN")
        if token is not None:
            token = token.replace("\r", "").replace("\n", "").strip()
            return token

        # 2. Is it set in token path ?
        try:
            token = cls.path_token.read_text()
            token = token.replace("\r", "").replace("\n", "").strip()
            return token
        except FileNotFoundError:
            return None

    @classmethod
    def delete_token(cls) -> None:
        """
        Deletes the token from storage. Does not fail if token does not exist.
        """
        try:
            cls.path_token.unlink()
        except FileNotFoundError:
            pass
