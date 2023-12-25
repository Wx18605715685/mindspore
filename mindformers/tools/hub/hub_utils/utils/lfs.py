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


import io
import os
from contextlib import AbstractContextManager
from typing import Dict, List, BinaryIO

from ..constant import ONE_MEGABYTE
from ._http import get_session, http_backoff
from ._commit_api import CommitOperationAdd

LFS_HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}


class SliceFileObj(AbstractContextManager):
    def __init__(self, fileobj: BinaryIO, seek_from: int, read_limit: int):
        # Validate seek_from is a non-negative integer
        if not isinstance(seek_from, int) or seek_from < 0:
            raise ValueError("seek_from must be a non-negative integer")

        # Validate read_limit is a positive integer
        if not isinstance(read_limit, int) or read_limit <= 0:
            raise ValueError("read_limit must be a positive integer")

        self.fileobj = fileobj
        self.seek_from = seek_from
        self.read_limit = read_limit

    def __enter__(self):
        self._previous_position = self.fileobj.tell()
        end_of_stream = self.fileobj.seek(0, os.SEEK_END)
        self._len = min(self.read_limit, end_of_stream - self.seek_from)
        # ^^ The actual number of bytes that can be read from the slice
        self.fileobj.seek(self.seek_from, io.SEEK_SET)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fileobj.seek(self._previous_position, io.SEEK_SET)

    def read(self, n: int = -1):
        pos = self.tell()
        if pos >= self._len:
            return b""
        remaining_amount = self._len - pos
        data = self.fileobj.read(remaining_amount if n < 0 else min(n, remaining_amount))
        return data

    def tell(self) -> int:
        return self.fileobj.tell() - self.seek_from

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        start = self.seek_from
        end = start + self._len
        if whence in (os.SEEK_SET, os.SEEK_END):
            offset = start + offset if whence == os.SEEK_SET else end + offset
            offset = max(start, min(offset, end))
            whence = os.SEEK_SET
        elif whence == os.SEEK_CUR:
            cur_pos = self.fileobj.tell()
            offset = max(start - cur_pos, min(offset, end - cur_pos))
        else:
            raise ValueError(f"whence value {whence} is not supported")
        return self.fileobj.seek(offset, whence) - self.seek_from

    def __iter__(self):
        yield self.read(n=4 * ONE_MEGABYTE)


def _validate_lfs_action(lfs_action: dict):
    """validates response from the LFS batch endpoint"""
    if not (
        isinstance(lfs_action.get("href"), str)
        and (lfs_action.get("headers") is None or isinstance(lfs_action.get("headers"), dict))
    ):
        raise ValueError("lfs_action is improperly formatted")
    return lfs_action


def _validate_batch_actions(lfs_batch_actions: dict):
    """validates response from the LFS batch endpoint"""
    if not (isinstance(lfs_batch_actions.get("oid"), str) and isinstance(lfs_batch_actions.get("size"), int)):
        raise ValueError("lfs_batch_actions is improperly formatted")

    upload_action = lfs_batch_actions.get("actions", {}).get("upload")
    verify_action = lfs_batch_actions.get("actions", {}).get("verify")
    if upload_action is not None:
        _validate_lfs_action(upload_action)
    if verify_action is not None:
        _validate_lfs_action(verify_action)
    return lfs_batch_actions


def generate_etag_header(part_upload_url: List) -> Dict:
    etag_header = {"Server": "OBS", "ETag": part_upload_url[0]}
    return etag_header


def _upload_parts_iteratively(operation: "CommitOperationAdd", upload_url_dict: Dict, chunk_size: int) -> List[Dict]:
    headers = []
    with operation.as_file(with_tqdm=True) as fileobj:
        for part_idx, part_upload_url in upload_url_dict.items():
            if not part_upload_url[0]:
                with SliceFileObj(
                    fileobj,
                    seek_from=chunk_size * (part_idx - 1),
                    read_limit=chunk_size,
                ) as fileobj_slice:
                    # upload remaining file chunks to obs
                    part_upload_res = http_backoff(
                        "PUT",
                        part_upload_url[1],
                        data=fileobj_slice,
                        retry_on_status_codes=(500, 503),
                    )
                    headers.append(part_upload_res.headers)
            else:
                headers.append(generate_etag_header(part_upload_url))
    return headers  # type: ignore


def _upload_single_part(operation: "CommitOperationAdd", upload_url: str) -> None:
    """
    Uploads `fileobj` as a single PUT HTTP request (basic LFS transfer protocol)

    Args:
        upload_url (`str`):
            The URL to PUT the file to.
        fileobj:
            The file-like object holding the data to upload.

    Returns: `requests.Response`

    Raises: `requests.HTTPError` if the upload resulted in an error
    """
    with operation.as_file(with_tqdm=True) as fileobj:
        # S3 might raise a transient 500 error -> let's retry if that happens
        http_backoff("PUT", upload_url, data=fileobj, retry_on_status_codes=(500, 503))


class PayloadPartT(Dict):
    index: int
    etag: str


class CompletionPayloadT(Dict):
    """Payload that will be sent to the Hub when uploading multi-part."""

    upload_id: str
    part_ids: List[PayloadPartT]


def _get_completion_payload(response_headers: List[Dict], oid: str) -> CompletionPayloadT:
    parts: List[PayloadPartT] = []
    for part_number, header in enumerate(response_headers):
        if "ETag" in header:
            etag = header.get("ETag").replace(r'"', "")
        else:
            etag = header.get("etag").replace(r'"', "")
        if etag is None or etag == "":
            raise ValueError(f"Invalid etag (`{etag}`) returned for part {part_number + 1}")
        parts.append(
            {
                "index": part_number + 1,
                "etag": etag,
            }
        )
    return {"upload_id": oid, "part_ids": parts}


def _upload_multi_part(
    operation: "CommitOperationAdd",
    chunk_size: int,
    upload_url_dict: Dict,
    verify_inf: Dict,
) -> None:
    """
    Uploads file using obs multipart LFS transfer protocol.
    """
    response_headers = _upload_parts_iteratively(
        operation=operation, upload_url_dict=upload_url_dict, chunk_size=chunk_size
    )

    # 3. Send completion request
    data = _get_completion_payload(response_headers, operation.upload_info.sha256.hex())
    data["upload_id"] = verify_inf.get("params").get("upload_id")
    get_session().post(
        verify_inf.get("href"),
        json=data,
        params=verify_inf["params"],
        headers=verify_inf["headers"],
    )
