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
import os
import re
import requests
from typing import Callable, Optional, Any, TypeVar


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES


BIG_FILE_SIZE = 5 * 1024 * 1024
default_home = os.path.join(os.path.expanduser("~"), ".cache")
MDS_HOME = os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "mindseed")

default_cache_path = os.path.join(MDS_HOME, "hub")
MDS_HUB_CACHE = default_cache_path

MDS_TOKEN_PATH = os.path.join(MDS_HOME, "token")

# Disable sending the cached token by default is all HTTP requests to the Hub
MDS_HUB_DISABLE_IMPLICIT_TOKEN: bool = _is_true(os.environ.get("MDS_HUB_DISABLE_IMPLICIT_TOKEN"))

DEFAULT_REVISION = "main"
ENDPOINT = "https://gitea.test.osinfra.cn/"
MINDSEED_CO_URL_TEMPLATE = ENDPOINT + "/{repo_id}/media/branch/{revision}/{filename}"
DEFAULT_REQUEST_TIMEOUT = 10
HTTP_METHOD_T = ["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]
BACKEND_FACTORY_T = Callable[[], requests.Session]
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
ONE_MEGABYTE = 1024 * 1024
GIT_LFS_SPE = "version https://git-lfs.github.com/spec/v1"

# error code:
RevisionNotFound = 1001
EntryNotFound = 1002
GatedRepo = 1003
ValidationField = 1004
RepositoryNotFound = 1005
IGNORE_GIT_FOLDER_PATTERNS = [".git", ".git/*", "*/.git", "**/.git/**"]
R = TypeVar("R")

REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPE_MODEL = "model"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_SPACE]

REPO_TYPES_URL_PREFIXES = {
    REPO_TYPE_DATASET: "datasets/",
    REPO_TYPE_SPACE: "spaces/",
}
REPO_TYPES_MAPPING = {
    "datasets": REPO_TYPE_DATASET,
    "spaces": REPO_TYPE_SPACE,
    "models": REPO_TYPE_MODEL,
}
