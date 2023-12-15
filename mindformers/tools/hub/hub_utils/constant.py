# Copyright 2020 The HuggingFace Team. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023, All rights reserved.
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
"""hub constant"""
import os
import re
from typing import Callable, Any
import requests

BIG_FILE_SIZE = 5 * 1024 * 1024
default_home = os.path.join(os.path.expanduser("~"), ".cache")
MDS_HOME = os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "mindseed")

default_cache_path = os.path.join(MDS_HOME, "hub")
MDS_HUB_CACHE = default_cache_path

DEFAULT_REVISION = "main"
ENDPOINT = "https://gitea.test.osinfra.cn/"
MINDSEED_CO_URL_TEMPLATE = ENDPOINT + "/{repo_id}/media/branch/{revision}/{filename}"
DEFAULT_REQUEST_TIMEOUT = 10
HTTP_METHOD_T = ["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]
BACKEND_FACTORY_T = Callable[[], requests.Session] # pylint: disable=C0103
MDS_HUB_DISABLE_SYMLINKS_WARNING: bool = False
MDS_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD: int = 5 * 1024 * 1024
MINDSEED_HEADER_X_LINKED_ETAG = "X-Linked-Etag"
MINDSEED_HEADER_X_LINKED_SIZE = "X-Linked-Size"
DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024
HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD = 5 * 1024 * 1024
DEFAULT_ETAG_TIMEOUT = 10
REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")
REGEX_COMMIT_OID = re.compile(r"[A-Fa-f0-9]{5,40}")
_CACHED_NO_EXIST = object()
_CACHED_NO_EXIST_T = Any
MULTIUPLOAD_SIZE = 20000000
