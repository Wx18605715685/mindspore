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
"""file download"""
import os
import re
import copy
import uuid
import time
import shutil
import tempfile
import threading


from pathlib import Path
from functools import partial
from functools import lru_cache
from dataclasses import dataclass
from urllib.parse import quote, urlparse
from sysconfig import get_python_version
import contextlib
from contextlib import contextmanager
from typing import BinaryIO, Dict, Union, Optional, Any

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from filelock import FileLock

from .constant import (
    BACKEND_FACTORY_T,
    DEFAULT_REVISION,
    MINDSEED_CO_URL_TEMPLATE,
    ENDPOINT,
    # HTTP_METHOD_T,
    MDS_HUB_CACHE,
    DEFAULT_REQUEST_TIMEOUT,
    MINDSEED_HEADER_X_LINKED_ETAG,
    MINDSEED_HEADER_X_LINKED_SIZE,
    DOWNLOAD_CHUNK_SIZE,
    BIG_FILE_SIZE,
)

_CACHED_NO_EXIST = object()
_CACHED_NO_EXIST_T = Any


def _to_local_dir(
        path: str,
        local_dir: str,
        relative_filename: str,
        use_symlinks: Union[bool, str],
) -> str:
    """
    Place a file in a local dir (different than cache_dir).
    Either symlink to blob file in cache or duplicate file
    depending on `use_symlinks` and file size.
    """
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    local_dir_filepath = os.path.join(local_dir, relative_filename)
    if Path(os.path.abspath(local_dir)) not in Path(os.path.abspath(local_dir_filepath)).parents:
        raise ValueError(
            f"Cannot copy file '{relative_filename}'"
            f" to local dir '{local_dir}': file would not be in the local"
            " directory."
        )

    os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)
    real_blob_path = os.path.realpath(path)

    # If "auto" (default) copy-paste small files
    # to ease manual editing but symlink big files to save disk
    if use_symlinks == "auto":
        use_symlinks = os.stat(real_blob_path).st_size > 5 * 1024 * 1024

    if use_symlinks:
        _create_symlink(real_blob_path, local_dir_filepath, new_blob=False)
    else:
        shutil.copyfile(real_blob_path, local_dir_filepath)
    return local_dir_filepath


def _deduplicate_user_agent(user_agent: str) -> str:
    return "; ".join({key.strip(): None for key in user_agent.split(";")}.keys())


def _default_backend_factory() -> requests.Session:
    session = requests.Session()
    session.mount("http://", HTTPAdapter())
    session.mount("https://", HTTPAdapter())
    return session


_GLOBAL_BACKEND_FACTORY: BACKEND_FACTORY_T = _default_backend_factory
_are_symlinks_supported_in_dir: Dict[str, bool] = {}


def _normalize_etag(etag: Optional[str]) -> Optional[str]:
    if etag is None:
        return None
    return etag.lstrip("W/").strip('"')


def _int_or_none(value: Optional[str]) -> Optional[int]:
    try:
        return int(value)  # type: ignore
    except (TypeError, ValueError):
        return None


def repo_folder_name(*, repo_id: str, repo_type: str = "file") -> str:
    REPO_ID_SEPARATOR = "--" # pylint: disable=C0103
    # remove all `/` occurrences to correctly convert repo to directory name
    parts = [f"{repo_type}s", *repo_id.split("/")]
    return REPO_ID_SEPARATOR.join(parts)


# pylint: disable=W0613
def mds_hub_url(
        repo_id: str,
        filename: str,
        *,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        endpoint: Optional[str] = None,
) -> str:
    """get hub url"""
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"
    if revision is None:
        revision = DEFAULT_REVISION
    url = MINDSEED_CO_URL_TEMPLATE.format(repo_id=repo_id, revision=quote(revision, safe=""), filename=quote(filename))
    # Update endpoint if provided
    if endpoint is not None and url.startswith(ENDPOINT):
        url = endpoint + url.replace(ENDPOINT, "", 1)
    return url


@dataclass(frozen=True)
class MdsFileMetadata:
    commit_hash: Optional[str]
    etag: Optional[str]
    location: str
    size: Optional[int]


def reset_sessions() -> None:
    _get_session_from_cache.cache_clear()


# pylint: disable=W0613
@lru_cache(128)
def _get_session_from_cache(process_id: int, thread_id: int) -> requests.Session:
    return _GLOBAL_BACKEND_FACTORY()


def get_session() -> requests.Session:
    """get session"""
    return _get_session_from_cache(process_id=os.getpid(), thread_id=threading.get_ident())


def _request_wrapper(
        method,
        url: str,
        *,
        follow_relative_redirects: bool = False,
        **params,
) -> requests.Response:
    """request wrapper"""
    if follow_relative_redirects:
        response = _request_wrapper(
            method=method,
            url=url,
            follow_relative_redirects=False,
            **params,
        )

        if 300 <= response.status_code <= 399:
            parsed_target = urlparse(response.headers["Location"])
            if parsed_target.netloc == "":
                next_url = urlparse(url)._replace(path=parsed_target.path).geturl()
                return _request_wrapper(
                    method=method,
                    url=next_url,
                    follow_relative_redirects=True,
                    **params,
                )
        return response

    response = get_session().request(method=method, url=url, **params)

    return response


# pylint: disable=W0622, C0103
@contextlib.contextmanager
def SoftTemporaryDirectory(
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[Union[Path, str]] = None,
        **kwargs,
):
    """soft temporary directory"""
    tmpdir = tempfile.TemporaryDirectory(prefix=prefix, suffix=suffix, dir=dir, **kwargs)
    yield tmpdir.name
    import stat

    # pylint: disable=W0613
    def _set_write_permission_and_retry(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    try:
        shutil.rmtree(tmpdir.name)
    # pylint: disable=W0703
    except Exception:
        try:
            shutil.rmtree(tmpdir.name, onerror=_set_write_permission_and_retry)
        # pylint: disable=W0703
        except Exception:
            pass
    try:
        tmpdir.cleanup()
    # pylint: disable=W0703
    except Exception:
        pass


def are_symlinks_supported(cache_dir: Union[str, Path, None] = None) -> bool:
    """check symlinks supported"""
    if cache_dir is None:
        cache_dir = MDS_HUB_CACHE
    cache_dir = str(Path(cache_dir).expanduser().resolve())  # make it unique

    if cache_dir not in _are_symlinks_supported_in_dir:
        _are_symlinks_supported_in_dir[cache_dir] = True

        os.makedirs(cache_dir, exist_ok=True)
        with SoftTemporaryDirectory(dir=cache_dir) as tmpdir:
            src_path = Path(tmpdir) / "dummy_file_src"
            src_path.touch()
            dst_path = Path(tmpdir) / "dummy_file_dst"

            # Relative source path as in `_create_symlink``
            relative_src = os.path.relpath(src_path, start=os.path.dirname(dst_path))
            try:
                os.symlink(relative_src, dst_path)
            except OSError:
                _are_symlinks_supported_in_dir[cache_dir] = False

    return _are_symlinks_supported_in_dir[cache_dir]


def _create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
    """create symlink"""
    try:
        os.remove(dst)
    except OSError:
        pass

    abs_src = os.path.abspath(os.path.expanduser(src))
    abs_dst = os.path.abspath(os.path.expanduser(dst))

    # Use relative_dst in priority
    try:
        relative_src = os.path.relpath(abs_src, os.path.dirname(abs_dst))
    except ValueError:
        relative_src = None

    try:
        try:
            commonpath = os.path.commonpath([abs_src, abs_dst])
            _support_symlinks = are_symlinks_supported(os.path.dirname(commonpath)) # pylint: disable=C0103
        except ValueError:
            _support_symlinks = os.name != "nt" # pylint: disable=C0103
    except PermissionError:
        _support_symlinks = are_symlinks_supported(os.path.dirname(abs_dst)) # pylint: disable=C0103

    if _support_symlinks:
        src_rel_or_abs = relative_src or abs_src
        os.symlink(src_rel_or_abs, abs_dst)

    elif new_blob:
        shutil.move(src, dst)
    else:
        shutil.copyfile(src, dst)


def get_token_to_send(token: Optional[Union[bool, str]]) -> Optional[str]:
    """Select the token to send from either `token` or the cache."""
    # Case token is explicitly provided
    if isinstance(token, str):
        return token

    # Case token is explicitly forbidden
    if token is False:
        return None

    # Token is not provided: we get it from local cache
    # cached_token = MdsFolder().get_token()
    cached_token = ""
    # Case token is explicitly required
    if token is True:
        if cached_token is None:
            raise "Token is required (`token=True`), but no token found."
        return cached_token

    # Otherwise: we use the cached token as the user has not explicitly forbidden it
    return cached_token


def _build_ml_headers(
        *,
        token: Optional[Union[bool, str]] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Union[Dict, str, None] = None,
        is_write_action: bool = False,
) -> Dict[str, str]:
    """build ml headers"""
    # Construct user-agent string
    ua = f"{library_name}/{library_version}" if library_name else "unknown/None"
    ua += f"; python/{get_python_version()}"

    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent

    ua = _deduplicate_user_agent(ua)

    # Build headers
    headers = {"user-agent": ua}
    headers["authorization"] = f"Bearer {token}"
    return headers


def get_mds_file_metadata(
        url: str,
        token: Union[bool, str, None] = None,
        proxies: Optional[Dict] = None,
        timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
) -> MdsFileMetadata:
    """get mds file metadata"""
    headers = _build_ml_headers()
    headers["Accept-Encoding"] = "identity"
    # Retrieve metadata
    r = _request_wrapper(
        method="HEAD",
        url=url,
        headers=headers,
        allow_redirects=False,
        follow_relative_redirects=True,
        proxies=proxies,
        timeout=timeout,
    )

    return MdsFileMetadata(
        commit_hash="",
        etag=_normalize_etag(r.headers.get(MINDSEED_HEADER_X_LINKED_ETAG) or r.headers.get("ETag")),
        location=r.headers.get("Location") or r.request.url,  # type: ignore
        size=_int_or_none(r.headers.get(MINDSEED_HEADER_X_LINKED_SIZE) or r.headers.get("Content-Length")),
    )


def _get_pointer_path(storage_folder: str, revision: str, relative_filename: str) -> str:
    """get pointer path"""
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    snapshot_path = os.path.join(storage_folder, "snapshots")
    pointer_path = os.path.join(snapshot_path, revision, relative_filename)
    if Path(os.path.abspath(snapshot_path)) not in Path(os.path.abspath(pointer_path)).parents:
        raise ValueError(
            "Invalid pointer path: cannot create pointer path in snapshot folder if"
            f" `storage_folder='{storage_folder}'`, `revision='{revision}'` and"
            f" `relative_filename='{relative_filename}'`."
        )
    return pointer_path


def _cache_commit_hash_for_specific_revision(storage_folder: str, revision: str, commit_hash: str) -> None:
    if revision != commit_hash:
        ref_path = Path(storage_folder) / "refs" / revision
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        if not ref_path.exists() or commit_hash != ref_path.read_text():
            ref_path.write_text(commit_hash)


# pylint: disable=R1710
def http_get(
        url: str,
        temp_file: BinaryIO,
        *,
        proxies=None,
        resume_size: float = 0,
        headers: Optional[Dict[str, str]] = None,
        expected_size: Optional[int] = None,
        _nb_retries: int = 5, # pylint: disable=C0103
):
    """http get"""
    initial_headers = headers
    headers = copy.deepcopy(headers) or {}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)

    r = _request_wrapper(method="GET", url=url, stream=True, proxies=proxies, headers=headers, timeout=10)

    content_length = r.headers.get("Content-Length")

    total = resume_size + int(content_length) if content_length is not None else None

    displayed_name = url
    content_disposition = r.headers.get("Content-Disposition")
    if content_disposition is not None:
        HEADER_FILENAME_PATTERN = re.compile(r'filename="(?P<filename>.*?)";') # pylint: disable=C0103
        match = HEADER_FILENAME_PATTERN.search(content_disposition)
        if match is not None:
            # Means file is on CDN
            displayed_name = match.groupdict()["filename"]

    # Truncate filename if too long to display
    if len(displayed_name) > 40:
        displayed_name = f"(…){displayed_name[-40:]}"

    consistency_error_message = (
        f"Consistency check failed: file should be of size"
        f" {expected_size} but has size"
        f" {{actual_size}} ({displayed_name}).\n"
        f"We are sorry for the inconvenience."
    )

    # Stream file to buffer
    with tqdm(
            unit="B",
            unit_scale=True,
            total=total,
            initial=resume_size,
            desc=displayed_name,
    ) as progress:
        new_resume_size = resume_size
        try:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
                    new_resume_size += len(chunk)
                    _nb_retries = 5 # pylint: disable=C0103
        except (requests.ConnectionError, requests.ReadTimeout):
            if _nb_retries <= 0:
                raise
            time.sleep(1)
            reset_sessions()
            return http_get(
                url=url,
                temp_file=temp_file,
                proxies=proxies,
                headers=initial_headers,
                resume_size=new_resume_size,
                expected_size=expected_size,
                _nb_retries=_nb_retries - 1,
            )

        if expected_size is not None and expected_size != temp_file.tell():
            raise EnvironmentError(
                consistency_error_message.format(
                    actual_size=temp_file.tell(),
                )
            )


def _chmod_and_replace(src: str, dst: str) -> None:
    """chmod and replace"""
    import stat

    tmp_file = Path(dst).parent.parent / f"tmp_{uuid.uuid4()}"
    try:
        tmp_file.touch()
        cache_dir_mode = Path(tmp_file).stat().st_mode
        os.chmod(src, stat.S_IMODE(cache_dir_mode))
    finally:
        tmp_file.unlink()

    shutil.move(src, dst)


def get_gitea_hash(repo_id, file_path):
    response = requests.get(f"{ENDPOINT}api/v1/repos/{repo_id}/commits?path={file_path}")
    data = response.json()
    if isinstance(data, list):
        return data[0].get("sha", "commit_hash_not_found")
    raise "cannot find files on repo, please check repo id and file name"


def try_to_load_from_cache(
        repo_id: str,
        filename: str,
        cache_dir: Union[str, Path, None] = None,
        revision: Optional[str] = None,
) -> Union[str, _CACHED_NO_EXIST_T, None]:
    """
    Explores the cache to return the latest cached file for a given revision if found.

    This function will not raise any exception if the file in not cached.

    Args:
        cache_dir (`str` or `os.PathLike`):
            The folder where the cached files lie.
        repo_id (`str`):
            The ID of the repo.
        filename (`str`):
            The filename to look for inside `repo_id`.
        revision (`str`, *optional*):
            The specific model version to use. Will default to
            `"main"` if it's not provided and no `commit_hash` is
            provided either.

    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST`
            if the file does not exist at the given commit hash and this fact was
              cached.

    Example:

    ```python
    from mindseed_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    filepath = try_to_load_from_cache()
    if isinstance(filepath, str):
        # file exists and is cached
        ...
    elif filepath is _CACHED_NO_EXIST:
        # non-existence of file is cached
        ...
    else:
        # file is not cached
        ...
    ```
    """
    if revision is None:
        revision = "main"
    if cache_dir is None:
        cache_dir = MDS_HUB_CACHE

    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"files--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None

    refs_dir = os.path.join(repo_cache, "refs")
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    no_exist_dir = os.path.join(repo_cache, ".no_exist")

    # Resolve refs (for instance to convert main to the associated commit sha)
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()

    # Check if file is cached as "no_exist"
    if os.path.isfile(os.path.join(no_exist_dir, revision, filename)):
        return _CACHED_NO_EXIST

    # Check if revision folder exists
    if not os.path.exists(snapshots_dir):
        return None
    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    # Check if file exists in cache
    cached_file = os.path.join(snapshots_dir, revision, filename)
    return cached_file if os.path.isfile(cached_file) else None


def mds_hub_download(
        repo_id: str,
        filename: str,
        *,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Union[str, Path, None] = MDS_HUB_CACHE,
        local_dir: Union[str, Path, None] = None,
        local_dir_use_symlinks: Union[bool, str] = "auto",
        user_agent: Union[Dict, str, None] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Union[bool, str, None] = None,
        local_files_only: bool = False,
        endpoint: Optional[str] = None,
        resume_download: bool = False,
        force_filename: Optional[str] = None,
) -> str:
    """
    args：
    repo_id (`str`):
        A user or an organization name and a repo name separated by a `/`.
    filename (`str`):
        The name of the file in the repo.
    subfolder (`str`, *optional*):
        An optional value corresponding to a folder inside the model repo.
    revision (`str`, *optional*):
        An optional Git branch name
    cache_dir (`str`, `Path`, *optional*):
        Path to the folder where cached files are stored.
    local_dir (`str` or `Path`, *optional*):
        If provided, the downloaded file will be placed under this directory, either as a symlink (default) or
        a regular file (see description for more details).
    local_dir_use_symlinks (`"auto"` or `bool`, defaults to `"auto"`):
        To be used with `local_dir`. If set to "auto", the cache directory will be used and the file will be either
        duplicated or symlinked to the local directory depending on its size. It set to `True`, a symlink will be
        created, no matter the file size. If set to `False`, the file will either be duplicated from cache (if
        already exists) or downloaded from the Hub and not cached.
    user_agent (`dict`, `str`, *optional*):
        The user-agent info in the form of a dictionary or a string.
    force_download (`bool`, *optional*, defaults to `False`):
        Whether the file should be downloaded even if it already exists in
        the local cache.
    proxies (`dict`, *optional*):
        Dictionary mapping protocol to the URL of the proxy passed to
        `requests.request`.
    token (str, bool, *optional*):
        gitea token
    resume_download (`bool`, *optional*, defaults to `False`):
        If `True`, resume a previously interrupted download.
    local_files_only (`bool`, *optional*, defaults to `False`):
        If `True`, avoid downloading the file and return the path to the
        local cached file if it exists.
    Returns:
        Local path (string) of file or if networking is off, last version of
        file cached on disk.
    """
    customer_hash = get_gitea_hash(repo_id, filename)

    if revision is None:
        revision = DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(local_dir, Path):
        local_dir = str(local_dir)

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id))

    os.makedirs(storage_folder, exist_ok=True)

    # cross platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))
    if os.name == "nt":
        if relative_filename.startswith("..\\") or "\\..\\" in relative_filename:
            raise ValueError(
                f"Invalid filename: cannot handle filename '{relative_filename}'"
                f" on Windows. Please ask the repository"
                " owner to rename this file."
            )

    url = mds_hub_url(
        repo_id,
        filename,
        subfolder=subfolder,
        revision=revision,
        endpoint=endpoint,
        repo_type=repo_type,
    )
    headers = _build_ml_headers(user_agent=user_agent)

    url_to_download = url
    metadata = None
    if not local_files_only:
        metadata = get_mds_file_metadata(
            url=url,
            token=token,
            proxies=proxies,
            timeout=10,
        )

    # Commit hash must exist
    commit_hash = customer_hash
    if commit_hash is not None:
        pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)
        if os.path.exists(pointer_path):
            if local_dir is not None:
                return _to_local_dir(
                    pointer_path,
                    local_dir,
                    relative_filename,
                    use_symlinks=local_dir_use_symlinks,
                )
            return pointer_path

    if local_files_only:
        raise "cannot find files, please try local_files_only=False"
    # Etag must exist
    etag = metadata.etag

    blob_path = os.path.join(storage_folder, "blobs", etag)
    pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)

    os.makedirs(os.path.dirname(blob_path), exist_ok=True)
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
    # write commit hash to txt file
    _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)
    cache_path = os.path.join(cache_dir, filename)
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(pointer_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return pointer_path

        if resume_download:
            incomplete_path = blob_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(  # type: ignore
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        with temp_file_manager() as temp_file:
            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
            )

        if local_dir is None:
            _chmod_and_replace(temp_file.name, blob_path)
            _create_symlink(blob_path, pointer_path, new_blob=True)
        else:
            local_dir_filepath = os.path.join(local_dir, relative_filename)
            os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)
            is_big_file = os.stat(temp_file.name).st_size > BIG_FILE_SIZE
            if local_dir_use_symlinks is True or (local_dir_use_symlinks == "auto" and is_big_file):
                _chmod_and_replace(temp_file.name, blob_path)
                _create_symlink(blob_path, local_dir_filepath, new_blob=False)
            elif local_dir_use_symlinks == "auto" and not is_big_file:
                _chmod_and_replace(temp_file.name, blob_path)
                shutil.copyfile(blob_path, local_dir_filepath)
            else:
                _chmod_and_replace(temp_file.name, local_dir_filepath)
            pointer_path = local_dir_filepath

        return pointer_path
