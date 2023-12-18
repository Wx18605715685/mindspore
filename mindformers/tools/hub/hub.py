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
Hub utilities: utilities related to download and cache models
"""
# import json
import os
# import re
# import shutil
# import sys
# import tempfile
# import traceback
# import warnings
# from concurrent import futures
from pathlib import Path
import re
from typing import Dict, Optional, Union
from urllib.parse import urlparse
# from uuid import uuid4

from .hub_utils import (
    # mds_hub_url,
    mds_hub_download,
    MDS_HUB_CACHE,
    MDS_HOME,
    REGEX_COMMIT_HASH,
    try_to_load_from_cache,
    _CACHED_NO_EXIST,
)
from .. import logger
# from ..generic import working_or_temp_dir
from ... import __version__

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"} # TODO: import from import_utils
_is_offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES


MINDSEED_CACHE = os.getenv("MINDSEED_CACHE", MDS_HUB_CACHE)

MS_MODULES_CACHE = os.getenv("MS_MODULES_CACHE", os.path.join(MDS_HOME, "modules"))
MINDSEED_DYNAMIC_MODULE_NAME = "mindseed_modules"


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def is_offline_mode():
    return _is_offline_mode


# pylint: disable=W0613
def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    # TODO this function will be completed soon
    return ""


# pylint: disable=C0103
# pylint: disable=W0613
def cached_file(
        path_or_repo_id: Union[str, os.PathLike],
        filename: str,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        subfolder: str = "",
        repo_type: Optional[str] = None,
        user_agent: Optional[Union[str, Dict[str, str]]] = None,
        _raise_exceptions_for_missing_entries: bool = True,
        _raise_exceptions_for_connection_errors: bool = True,
        _commit_hash: Optional[str] = None,
        **deprecated_kwargs,
) -> Optional[str]:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:
            # TODO add community url to replace {xxx}
            - a string, the *model id* of a model repo on {xxx}.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions
            if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a
            file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            # TODO add cli login command for generating token to replace first {xxx}, add local path where
            # TODO token is saved to replace second {xxx}
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token
            generated when running `{xxx}` (stored in `{xxx}`).
        revision (`str`, *optional*, defaults to `"main"`):
            # TODO add community url to replace {xxx}
            The specific model version to use. It can be a branch name, a tag name, or a commit id,
            since we use a git-based system for storing models and other artifacts on {xxx},
            so `revision` can be any identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            # TODO add community url to replace {xxx}
            In case the relevant files are located inside a subfolder of the model repo on {xxx},
            you can specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("bert-base-uncased", "pytorch_model.bin")
    ```
    """
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(  # TODO add community url to replace xxx
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://xxx/{path_or_repo_id}/{revision}' for available files."
                )
            return None
        return resolved_file

    if cache_dir is None:
        cache_dir = MINDSEED_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if _commit_hash is not None and not force_download:
        # If the file is cached under that commit hash, we return it directly.
        # FIXME param `repo_type` is not supported now, remove it temporary
        resolved_file = try_to_load_from_cache(
            path_or_repo_id,
            full_filename,
            cache_dir=cache_dir,
            revision=_commit_hash,
        )
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            if not _raise_exceptions_for_missing_entries:
                return None
            raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")

    user_agent = http_user_agent(user_agent)  # noqa

    try:
        # Load from URL or cache if already cached
        resolved_file = mds_hub_download(
            path_or_repo_id,
            filename,
            subfolder=None if not subfolder else subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
    except:
        raise SystemError("下载功能暂不可用")
    # # TODO replace this error with corresponding one in hub_utils after code merged
    # except GatedRepoError as e:
    #     raise EnvironmentError(
    #         # TODO add community url to replace xxx, add cli login command for generating token to replace {xxx}
    #         "You are trying to access a gated repo.\nMake sure to request access at "
    #         f"https://xxx/{path_or_repo_id} and pass a token having permission to this repo either "
    #         "by logging in with `{xxx}` or by passing `token=<your_token>`."
    #     ) from e
    # # TODO replace this error with corresponding one in hub_utils after code merged
    # except RepositoryNotFoundError as e:
    #     raise EnvironmentError(
    #         # TODO add community url to replace xxx, add cli login command for generating token to replace {xxx}
    #         f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
    #         "listed on 'https://xxx/models'\nIf this is a private repository, make sure to pass a token "
    #         "having permission to this repo either by logging in with `{xxx}` or by passing "
    #         "`token=<your_token>`"
    #     ) from e
    # # TODO replace this error with corresponding one in hub_utils after code merged
    # except RevisionNotFoundError as e:
    #     raise EnvironmentError(  # TODO add community url to replace xxx
    #         f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
    #         "for this model name. Check the model page at "
    #         f"'https://xxx/{path_or_repo_id}' for available revisions."
    #     ) from e
    # # TODO replace this error with corresponding one in hub_utils after code merged
    # except LocalEntryNotFoundError as e:
    #     # We try to see if we have a cached version (not up to date):
    #     resolved_file = try_to_load_from_cache(
    #         path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision
    #     )
    #     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    #         return resolved_file
    #     if not _raise_exceptions_for_missing_entries or not _raise_exceptions_for_connection_errors:
    #         return None
    #     raise EnvironmentError(
    #         # TODO add mindseed offline-mode document address to replace xxx
    #         f"We couldn't connect to '{MINDSEED_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
    #         f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file "
    #         f"named {full_filename}.\nCheckout your internet connection or see how to run the library in offline "
    #         f"mode at 'https://xxx'."
    #     ) from e
    # # TODO replace this error with corresponding one in hub_utils after code merged
    # except EntryNotFoundError as e:
    #     if not _raise_exceptions_for_missing_entries:
    #         return None
    #     if revision is None:
    #         revision = "main"
    #     raise EnvironmentError(  # TODO add community url to replace xxx
    #         f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
    #         f"'https://xxx/{path_or_repo_id}/{revision}' for available files."
    #     ) from e
    # # TODO replace this error with corresponding one in hub_utils after code merged
    # except HTTPError as err:
    #     # First we try to see if we have a cached version (not up to date):
    #     resolved_file = try_to_load_from_cache(
    #         path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision
    #     )
    #     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    #         return resolved_file
    #     if not _raise_exceptions_for_connection_errors:
    #         return None

    #     raise EnvironmentError(
    #         f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}"
    #     )
    # # TODO replace this error with corresponding one in hub_utils after code merged
    # except HFValidationError as e:
    #     raise EnvironmentError(
    #         f"Incorrect path_or_model_id: '{path_or_repo_id}'. Please provide either the path to a "
    #         f"local folder or the repo_id of a model on the Hub."
    #     ) from e

    return resolved_file

def download_url(url):
    raise NotImplementedError()

def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]) -> Optional[str]:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


class PushToHubMixin:
    """to do"""
    # pylint: disable=R1711
    def _create_repo(
            self,
            repo_id: str,
            private: Optional[bool] = None,
            token: Optional[Union[bool, str]] = None,
            repo_url: Optional[str] = None,
            organization: Optional[str] = None,
    ) -> str:
        """
        Create the repo if needed, cleans up repo_id with deprecated kwargs `repo_url` and `organization`, retrieves
        the token.
        """
        repo_id = repo_id
        private = private
        token = token
        repo_url = repo_url
        organization = organization
        return "repo"

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    # pylint: disable=R1711
    def _upload_modified_files(
            self,
            working_dir: Union[str, os.PathLike],
            repo_id: str,
            files_timestamps: Dict[str, float],
            commit_message: Optional[str] = None,
            token: Optional[Union[bool, str]] = None,
            create_pr: bool = False,
            revision: str = None,
            commit_description: str = None,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        working_dir = working_dir
        repo_id = repo_id
        files_timestamps = files_timestamps
        commit_message = commit_message
        token = token
        create_pr = create_pr
        revision = revision
        commit_description = commit_description
        return None

    # pylint: disable=R1711
    def push_to_hub(
            self,
            repo_id: str,
            use_temp_dir: Optional[bool] = None,
            commit_message: Optional[str] = None,
            private: Optional[bool] = None,
            token: Optional[Union[bool, str]] = None,
            max_shard_size: Optional[Union[int, str]] = "5GB",
            create_pr: bool = False,
            safe_serialization: bool = True,
            revision: str = None,
            commit_description: str = None,
            **deprecated_kwargs,
    ) -> str:
        """push to hub"""
        repo_id = repo_id
        use_temp_dir = use_temp_dir
        commit_message = commit_message
        private = private
        token = token
        max_shard_size = max_shard_size
        create_pr = create_pr
        safe_serialization = safe_serialization
        revision = revision
        commit_description = commit_description
        deprecated_kwargs = deprecated_kwargs
        return None
