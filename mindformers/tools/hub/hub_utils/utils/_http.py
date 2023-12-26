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
import time
import requests
from requests import Response
from requests.adapters import HTTPAdapter

import threading
from http import HTTPStatus
from functools import lru_cache
from typing import Callable, Tuple, Type, Union


def reset_sessions() -> None:
    """Reset the cache of sessions.

    Mostly used internally when sessions are reconfigured or an SSLError is raised.
    See [`configure_http_backend`] for more details.
    """
    _get_session_from_cache.cache_clear()


def http_backoff(
    method,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (
        requests.Timeout,
        requests.ConnectionError,
    ),
    retry_on_status_codes: Union[int, Tuple[int, ...]] = HTTPStatus.SERVICE_UNAVAILABLE,
    **kwargs,
) -> Response:
    if isinstance(retry_on_exceptions, type):  # Tuple from single exception type
        retry_on_exceptions = (retry_on_exceptions,)

    if isinstance(retry_on_status_codes, int):  # Tuple from single status code
        retry_on_status_codes = (retry_on_status_codes,)

    nb_tries = 0
    sleep_time = base_wait_time

    # If `data` is used and is a file object (or any IO), it will be consumed on the
    # first HTTP request. We need to save the initial position so that the full content
    # of the file is re-sent on http backoff. See warning tip in docstring.
    io_obj_initial_pos = None
    if "data" in kwargs and isinstance(kwargs["data"], io.IOBase):
        io_obj_initial_pos = kwargs["data"].tell()

    session = get_session()
    while True:
        nb_tries += 1
        try:
            # If `data` is used and is a file object (or any IO), set back cursor to
            # initial position.
            if io_obj_initial_pos is not None:
                kwargs["data"].seek(io_obj_initial_pos)

            # Perform request and return if status_code is not in the retry list.
            response = session.request(method=method, url=url, **kwargs)
            if response.status_code not in retry_on_status_codes:
                return response

            if nb_tries > max_retries:
                response.raise_for_status()  # Will raise uncaught exception
                # We return response to avoid infinite loop in the corner case where the
                # user ask for retry on a status code that doesn't raise_for_status.
                return response

        except retry_on_exceptions as err:
            if isinstance(err, requests.ConnectionError):
                reset_sessions()  # In case of SSLError it's best to reset the shared requests.Session objects

            if nb_tries > max_retries:
                raise err

        # Sleep for X seconds
        time.sleep(sleep_time)

        # Update sleep time for next retry
        sleep_time = min(max_wait_time, sleep_time * 2)  # Exponential backoff


def _default_backend_factory() -> requests.Session:
    session = requests.Session()
    session.mount("http://", HTTPAdapter())
    session.mount("https://", HTTPAdapter())
    return session


BACKEND_FACTORY_T = Callable[[], requests.Session]
_GLOBAL_BACKEND_FACTORY: BACKEND_FACTORY_T = _default_backend_factory


@lru_cache(128)
def _get_session_from_cache(process_id: int, thread_id: int) -> requests.Session:
    """
    Create a new session per thread using global factory.
    Using LRU cache (maxsize 128) to avoid memory leaks when
    using thousands of threads.
    Cache is cleared when `configure_http_backend` is called.
    """
    return _GLOBAL_BACKEND_FACTORY()


def get_session() -> requests.Session:
    """
    Get a `requests.Session` object, using the session factory from the user.

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    create 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    Example:
    ```py
    import requests
    from hub_utils.utils._http import get_session

    def configure_http_backend(backend_factory: BACKEND_FACTORY_T = _default_backend_factory) -> None:
        global _GLOBAL_BACKEND_FACTORY
        _GLOBAL_BACKEND_FACTORY = backend_factory
        reset_sessions()

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    session = get_session()
    ```
    """
    return _get_session_from_cache(process_id=os.getpid(), thread_id=threading.get_ident())
