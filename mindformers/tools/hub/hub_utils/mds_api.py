# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023, All rights reserved.
# Copyright 2019-present, the HuggingFace Inc. team.
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
import io
import inspect
import hashlib
import functools
import itertools
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    BinaryIO,
)
from pathlib import Path
from typing import Iterable, TypeVar
from requests.auth import HTTPBasicAuth
from dataclasses import dataclass, field
from tqdm.contrib.concurrent import thread_map

from .utils import build_mds_headers, get_token_to_send, get_session
from .constant import (
    REPO_TYPES,
    ENDPOINT,
    MULTIUPLOAD_SIZE,
    BIG_FILE_SIZE,
    IGNORE_GIT_FOLDER_PATTERNS,
)
from .utils._commit_api import (
    CommitOperationAdd,
    _warn_on_overwriting_operations,
    _prepare_commit_payload,
    _kwargs,
    UploadInfo,
)
from .utils._path import filter_repo_objects
from .utils.lfs import (
    _validate_batch_actions,
    _upload_multi_part,
)
from .utils._error import mds_raise_for_status

T = TypeVar("T")
sha256 = functools.partial(hashlib.sha256, **_kwargs)


def chunk_iterable(iterable: Iterable[T], chunk_size: int) -> Iterable[Iterable[T]]:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("`chunk_size` must be a strictly positive integer (>0).")

    iterator = iter(iterable)
    while True:
        try:
            next_item = next(iterator)
        except StopIteration:
            return
        yield itertools.chain((next_item,), itertools.islice(iterator, chunk_size - 1))


def post_lfs_batch_info(
    upload_infos: Iterable[UploadInfo],
    token: Optional[str],
    repo_id: str,
    endpoint: Optional[str] = None,
) -> List[dict]:
    LFS_HEADERS = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
    }
    endpoint = endpoint if endpoint is not None else ENDPOINT
    url_prefix = ""
    batch_url = f"{endpoint}{url_prefix}{repo_id}.git/info/lfs/objects/batch"
    payload: Dict = {
        "operation": "upload",
        "transfers": ["basic", "multipart"],
        "objects": [
            {
                "oid": upload.sha256.hex(),
                "size": upload.size,
            }
            for upload in upload_infos
        ],
        "hash_algo": "sha256",
    }

    resp = get_session().post(
        batch_url,
        headers=LFS_HEADERS,
        json=payload,
        auth=HTTPBasicAuth(
            "access_token",
            get_token_to_send(token or True),
        ),
    )
    batch_info = resp.json().get("objects", None)

    return batch_info


def lfs_upload(operation: "CommitOperationAdd", lfs_batch_action: Dict) -> None:
    # 0. If LFS file is already present, skip upload
    _validate_batch_actions(lfs_batch_action)
    actions = lfs_batch_action.get("actions")
    if actions is None:
        return

    # 1. Validate server response, skip the update process if file is already on obs.
    if lfs_batch_action["actions"].get("parts"):
        upload_action = lfs_batch_action["actions"]["parts"]

        verify_action = lfs_batch_action["actions"].get("verify")

        # 2. Upload file
        upload_url_dict = {d.get("index"): [d.get("etag", ""), d.get("href", "")] for d in upload_action}

        _upload_multi_part(
            operation=operation, chunk_size=MULTIUPLOAD_SIZE, upload_url_dict=upload_url_dict, verify_inf=verify_action
        )


def _upload_lfs_files(
    *,
    additions: List[CommitOperationAdd],
    repo_id: str,
    token: Optional[str],
    endpoint: Optional[str] = None,
    num_threads: int = 5,
):
    # Step 1: retrieve upload instructions from the LFS batch endpoint.
    #         Upload instructions are retrieved by chunk of 256 files to avoid reaching
    #         the payload limit.
    batch_actions: List[Dict] = []
    for chunk in chunk_iterable(additions, chunk_size=256):
        batch_actions_chunk = post_lfs_batch_info(
            upload_infos=[op.upload_info for op in chunk],
            token=token,
            repo_id=repo_id,
            endpoint=endpoint,
        )

        batch_actions += batch_actions_chunk
    oid2addop = {add_op.upload_info.sha256.hex(): add_op for add_op in additions}

    # Step 2: ignore files that have already been uploaded
    filtered_actions = []
    for action in batch_actions:
        if action.get("actions") is None:
            return
        else:
            filtered_actions.append(action)

    if len(filtered_actions) == 0:
        return

    # Step 3: upload files concurrently according to these instructions
    def _wrapped_lfs_upload(batch_action) -> None:
        try:
            operation = oid2addop[batch_action["oid"]]
            lfs_upload(operation=operation, lfs_batch_action=batch_action)
        except Exception as exc:
            raise RuntimeError("Error while uploading to the Hub.") from exc

    if len(filtered_actions) == 1:
        _wrapped_lfs_upload(filtered_actions[0])
    else:
        thread_map(
            _wrapped_lfs_upload,
            filtered_actions,
            desc=f"Upload {len(filtered_actions)} LFS files",
            max_workers=num_threads,
        )


def _fetch_upload_modes(
    commit_operations: List[CommitOperationAdd],
) -> List[CommitOperationAdd]:
    """
    keep CommitOperationAdd instance in a list which has size > 5mb
    Args:
        commit_operations (List[CommitOperationAdd]): CommitOperationAdd instance

    Returns:
        List[CommitOperationAdd]: filtered list of CommitOperationAdd instance
    """
    filtered_operations = []
    for operation in commit_operations:
        file_size = 0

        if isinstance(operation.path_or_fileobj, str):
            if os.path.isfile(operation.path_or_fileobj):
                file_size = os.path.getsize(operation.path_or_fileobj)
        elif isinstance(operation.path_or_fileobj, bytes):
            file_size = len(operation.path_or_fileobj)
        elif isinstance(operation.path_or_fileobj, io.BufferedIOBase):
            operation.path_or_fileobj.seek(0, os.SEEK_END)
            file_size = operation.path_or_fileobj.tell()
            operation.path_or_fileobj.seek(0)

        if file_size >= BIG_FILE_SIZE:
            filtered_operations.append(operation)
    for ops in filtered_operations:
        ops._upload_mode = "lfs"
    return filtered_operations


@dataclass
class CommitInfo:
    """Data structure containing information about a newly created commit.

    Returned by [`create_commit`].

    Attributes:
        commit_url (`str`):
            Url where to find the commit.

        commit_message (`str`):
            The summary (first line) of the commit that has been created.

        commit_description (`str`):
            Description of the commit that has been created. Can be empty.

        oid (`str`):
            Commit hash id. Example: `"91c54ad1727ee830252e457677f467be0bfd8a57"`.
    """

    commit_url: str
    commit_message: str
    commit_description: str
    oid: str
    pr_url: Optional[str] = None

    # Computed from `pr_url` in `__post_init__`
    pr_revision: Optional[str] = field(init=False)
    pr_num: Optional[str] = field(init=False)

    def __post_init__(self):
        """Populate pr-related fields after initialization.

        See https://docs.python.org/3.10/library/dataclasses.html#post-init-processing.
        """
        if self.pr_url is not None:
            self.pr_revision = "main"
            self.pr_num = str(int(self.pr_revision.split("/")[-1]))
        else:
            self.pr_revision = None


def _prepare_upload_folder_additions(
    folder_path: Union[str, Path],
    path_in_repo: str,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
) -> List[CommitOperationAdd]:
    """Generate the list of Add operations for a commit to upload a folder.

    Files not matching the `allow_patterns` (allowlist) and `ignore_patterns` (denylist)
    constraints are discarded.
    """
    folder_path = Path(folder_path).expanduser().resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Provided path: '{folder_path}' is not a directory")

    # List files from folder
    relpath_to_abspath = {
        path.relative_to(folder_path).as_posix(): path
        for path in sorted(folder_path.glob("**/*"))  # sorted to be deterministic
        if path.is_file()
    }

    # Filter files and return
    # Patterns are applied on the path relative to `folder_path`. `path_in_repo` is prefixed after the filtering.
    prefix = f"{path_in_repo.strip('/')}/" if path_in_repo else ""
    return [
        CommitOperationAdd(
            path_or_fileobj=relpath_to_abspath[relpath],  # absolute path on disk
            path_in_repo=prefix + relpath,  # "absolute" path in repo
        )
        for relpath in filter_repo_objects(
            relpath_to_abspath.keys(), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
        )
    ]


def future_compatible(fn):
    """Wrap a method of `MdsApi` to handle `run_as_future=True`.

    A method flagged as "future_compatible" will be called in a thread if `run_as_future=True` and return a
    `concurrent.futures.Future` instance. Otherwise, it will be called normally and return the result.
    """
    sig = inspect.signature(fn)
    args_params = list(sig.parameters)[1:]  # remove "self" from list

    @wraps(fn)
    def _inner(self, *args, **kwargs):
        # Get `run_as_future` value if provided (default to False)
        if "run_as_future" in kwargs:
            run_as_future = kwargs["run_as_future"]
            kwargs["run_as_future"] = False  # avoid recursion error
        else:
            run_as_future = False
            for param, value in zip(args_params, args):
                if param == "run_as_future":
                    run_as_future = value
                    break

        # Call the function in a thread if `run_as_future=True`
        if run_as_future:
            return self.run_as_future(fn, self, *args, **kwargs)

        # Otherwise, call the function normally
        return fn(self, *args, **kwargs)

    _inner.is_future_compatible = True  # type: ignore
    return _inner  # type: ignore


class MdsApi:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Union[Dict, str, None] = None,
    ) -> None:
        """Create a MDS client to interact with the Hub via HTTP.

        The client is initialized with some high-level settings used in all requests
        made to the Hub (Gitea endpoint, authentication, user agents...).

        Args:
            token (`str`):
                Gitea token. Will default to the locally saved token if
                not provided.
            library_name (`str`, *optional*):
                The name of the library that is making the HTTP request. Will be
                added to the user-agent header.
            library_version (`str`, *optional*):
                The version of the library that is making the HTTP request. Will be
                added to the user-agent header. Example: `"4.24.0"`.
            user_agent (`str`, `dict`, *optional*):
                The user agent info in the form of a dictionary or a single string.
                It will be completed with information about the installed packages.
        """
        self.endpoint = endpoint if endpoint is not None else ENDPOINT
        self.token = token
        self.library_name = library_name
        self.library_version = library_version
        self.user_agent = user_agent
        self._thread_pool: Optional[ThreadPoolExecutor] = None

    def _build_mds_headers(
        self,
        token: Optional[Union[bool, str]] = None,
        is_write_action: bool = False,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Union[Dict, str, None] = None,
    ) -> Dict[str, str]:
        """
        Alias for [`build_mds_headers`] that uses the token from [`MdsApi`] client
        when `token` is not provided.
        """
        if token is None:
            # Cannot do `token = token or self.token` as token can be `False`.
            token = self.token
        return build_mds_headers(
            token=token,
            is_write_action=is_write_action,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )

    def upload_file(
        self,
        *,
        path_or_fileobj: Union[str, Path, bytes, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
    ):
        commit_message = commit_message if commit_message is not None else f"Upload {path_in_repo} with mindseed hub"
        operation = CommitOperationAdd(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
        )

        self.create_commit(
            repo_id=repo_id,
            operations=[operation],
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
        )

    def create_commit(
        self,
        repo_id,
        operations,
        commit_message,
        commit_description,
        token,
    ):
        if commit_message is None or len(commit_message) == 0:
            raise ValueError("`commit_message` can't be empty, please pass a value.")
        commit_description = commit_description if commit_description is not None else ""
        operations = list(operations)
        additions = [op for op in operations if isinstance(op, CommitOperationAdd)]
        for addition in additions:
            if addition._is_committed:
                raise ValueError(
                    f"CommitOperationAdd {addition} has "
                    f"already being committed and cannot be reused. Please create a"
                    " new CommitOperationAdd object if you want to create a new commit."
                )

        _warn_on_overwriting_operations(operations)

        self.preupload_lfs_files(
            repo_id=repo_id,
            additions=additions,
            token=token,
            num_threads=4,
            free_memory=False,
        )

        commit_payload = _prepare_commit_payload(
            operations=operations,
        )
        commit_url = f"{self.endpoint}api/v1/repos/" f"{repo_id}/contents"

        commit_resp = get_session().post(
            url=commit_url,
            json=commit_payload,
            auth=HTTPBasicAuth(
                "access_token",
                get_token_to_send(token or True),
            ),
        )

        mds_raise_for_status(commit_resp, endpoint_name="commit")

        # Mark additions as committed (cannot be reused in another commit)
        for addition in additions:
            addition._is_committed = True

        commit_data = commit_resp.json()
        if commit_resp.status_code != 201:
            return commit_data.get("message")

        commit_url = "; ".join(f["_links"]["html"] for f in commit_data["files"])

        commit_oid = "; ".join(f["sha"] for f in commit_data["files"])

        return CommitInfo(
            commit_url=commit_url,
            commit_message=commit_message,
            commit_description=commit_description,
            oid=commit_oid,
        )

    def preupload_lfs_files(
        self,
        repo_id: str,
        additions: Iterable[CommitOperationAdd],
        *,
        token: Optional[str] = None,
        num_threads: int = 5,
        free_memory: bool = True,
    ):
        # Filter out already uploaded files
        new_additions = [addition for addition in additions if not addition._is_uploaded]

        new_lfs_additions = _fetch_upload_modes(new_additions)

        # Upload new LFS files
        _upload_lfs_files(
            additions=new_lfs_additions,
            repo_id=repo_id,
            token=token or self.token,
            endpoint=self.endpoint,
            num_threads=num_threads,
        )
        for addition in new_lfs_additions:
            addition._is_uploaded = True
            if free_memory:
                addition.path_or_fileobj = b""

    def run_as_future(self, fn, *args, **kwargs):
        """
        Run a method in the background and return a Future instance.

        The main goal is to run methods without blocking the main thread (e.g. to push data during a training).
        Background jobs are queued to preserve order but are not ran in parallel. If you need to speed-up your scripts
        by parallelizing lots of call to the API, you must setup and use your own
        [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor).

        Note: Most-used methods like [`upload_file`], [`upload_folder`] and
         [`create_commit`] have a `run_as_future: bool`
        argument to directly call them in the background.
         This is equivalent to calling `api.run_as_future(...)` on them
        but less verbose.

        Args:
            fn (`Callable`):
                The method to run in the background.
            *args, **kwargs:
                Arguments with which the method will be called.

        Return:
            `Future`: a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) instance to
            get the result of the task.
        """
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._thread_pool
        return self._thread_pool.submit(fn, *args, **kwargs)

    @future_compatible
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = "",
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        run_as_future: bool = False,
    ):
        """
        Upload a local folder to the given repo. The upload is done through a HTTP requests, and doesn't require git or
        git-lfs to be installed.

        Use the `ignore_patterns` argument to specify which files to upload. These parameters
        accept either a single pattern or a list of patterns. Patterns are Standard Wildcards (globbing patterns) as
        documented [here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm).

        Any `.git/` folder present in any subdirectory will be ignored. However, please be aware that the `.gitignore`
        file is not taken into account.

        Args:
            repo_id (`str`):
                The repository to which the file will be uploaded, for example:
                `"username/custom_transformers"`
            folder_path (`str` or `Path`):
                Path to the folder to upload on the local file system
            path_in_repo (`str`, *optional*):
                Relative path of the directory in the repo, for example:
                `"checkpoints/1fec34a/results"`. Will default to the root folder of the repository.
            token (`str`, *optional*):
                Authentication token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to:
                `f"Upload {path_in_repo} with MindSeed hub"`
            commit_description (`str` *optional*):
                The description of the generated commit
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are uploaded.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not uploaded.
            run_as_future (`bool`, *optional*):
                Whether or not to run this method in the background. Background jobs are run sequentially without
                blocking the main thread. Passing `run_as_future=True`
                 will return a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
                object. Defaults to `False`.

        Returns:
            `str` or `Future[str]`: A URL to visualize the uploaded folder on the hub.
             If `run_as_future=True` is passed,
            returns a Future object which will contain the result when executed.
        """

        # Do not upload .git folder
        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        ignore_patterns += IGNORE_GIT_FOLDER_PATTERNS

        add_operations = _prepare_upload_folder_additions(
            folder_path,
            path_in_repo,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        commit_message = commit_message or "Upload folder using MindSeed hub"

        commit_inf = self.create_commit(
            repo_id=repo_id,
            operations=add_operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
        )

        return commit_inf.commit_url

    def create_repo(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        private: bool = False,
        repo_type: Optional[str] = None,
        exist_ok: bool = False,
    ) -> str:
        """Create an empty repo on Gitea.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            token (`str`, *optional*):
                An authentication token.
            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
                exist_ok (`bool`, *optional*, defaults to `False`):
                    If `True`, do not raise an error if repo already exists.

        Returns: URL to the newly created repo
        """
        org, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)
        path = f"{self.endpoint}api/v1/user/repos"

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json: Dict[str, Any] = {
            "name": name,
            "organization": org,
            "private": private,
        }
        if not repo_type:
            json["type"] = repo_type

        headers = self._build_mds_headers(token=token, is_write_action=True)
        r = get_session().post(path, headers=headers, json=json)
        if r.status_code == 201:
            d = r.json()
            return d["html_url"]
        elif r.status_code == 409 and exist_ok:
            pass
        else:
            # repo already exists, or other errors
            pass


api = MdsApi()

upload_file = api.upload_file
create_commit = api.create_commit
upload_folder = api.upload_folder
create_repo = api.create_repo
