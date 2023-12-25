import json
from typing import Optional
from requests import HTTPError, Response

from ..constant import GatedRepo, EntryNotFound, RevisionNotFound, RepositoryNotFound


def _format_error_message(
    message: str, request_id: Optional[str], server_message: Optional[str], original_message: Optional[str]
) -> str:
    """
    Format the `MdsHubHTTPError` error message based on initial message and information
    returned by the server.
    """
    # Add message from response body
    if original_message is not None and original_message.lower() not in message.lower():
        message = f"Original Error: {original_message}\n" + message

    if server_message is not None and len(server_message) > 0 and server_message.lower() not in message.lower():
        if "\n\n" in message:
            message += "\n" + server_message
        else:
            message += "\n\n" + server_message

    # Add Request ID
    if request_id is not None and str(request_id).lower() not in message.lower():
        request_id_message = f" (Request ID: {request_id})"
        if "\n" in message:
            newline_index = message.index("\n")
            message = message[:newline_index] + request_id_message + message[newline_index:]
        else:
            message += request_id_message

    return message


class MdsHubHTTPError(HTTPError):
    """
    HTTPError to inherit from for any custom HTTP Error raised in Mds Hub.

    Any HTTPError is converted at least into a `MdsHubHTTPError`. If some information is
    sent back by the server, it will be added to the error message.

    Added details:
    - Request id from "X-Request-Id" header if exists.
    - Server error message from the header "X-Error-Message".
    - Server error message if we can found one in the response body.

    """

    request_id: Optional[str] = None
    server_message: Optional[str] = None

    def __init__(self, message: str, response: Optional[Response] = None):
        # Extract original error message if present
        original_message = str(response.content if response is not None else None)
        # Parse server information if any.
        if response is not None:
            # derived class
            self.request_id = response.headers.get("X-Request-Id")
            try:
                server_data = response.json()
            except (ValueError, json.JSONDecodeError):
                server_data = {}

            # Retrieve server error message from multiple sources
            server_message_from_body = server_data.get("error")
            server_multiple_messages_from_body = "\n".join(
                error["message"] for error in server_data.get("errors", []) if "message" in error
            )

            # Concatenate error messages
            _server_message = ""
            if server_message_from_body is not None:
                if isinstance(server_message_from_body, list):
                    server_message_from_body = "\n".join(server_message_from_body)
                if server_message_from_body not in _server_message:
                    _server_message += server_message_from_body + "\n"
            if server_multiple_messages_from_body is not None:
                if server_multiple_messages_from_body not in _server_message:
                    _server_message += server_multiple_messages_from_body + "\n"
            _server_message = _server_message.strip()

            # Set message to `HfHubHTTPError` (if any)
            if _server_message != "":
                # derived class
                self.server_message = _server_message
        # base class
        super().__init__(
            _format_error_message(
                message,
                request_id=self.request_id,
                server_message=self.server_message,
                original_message=original_message,
            ),
            response=response,  # type: ignore
            request=response.request if response is not None else None,  # type: ignore
        )


class RepositoryNotFoundError(MdsHubHTTPError):
    """
    RepositoryNotFoundError
        >>> from mindseed.hub_utils.mds_api import model_info
        >>> model_info("<non_existent_repository>")

    Request url:
        https:// {endpoint}/api/models/{repo_id}

    if repository not exist:
        merlin_hub.utils._errors.RepositoryNotFoundError: 1005 Client Error.
        Repository Not Found for url: https:// {endpoint}/api/models/<non_existent_repository>.
        Please make sure you specified the correct `repo_id` and `repo_type`.
    """


class RevisionNotFoundError(MdsHubHTTPError):
    """
    RevisionNotFoundError
        >>> from mindseed.hub_utils.mds_api import ml_hub_download
        >>> ml_hub_download(' Intel/neural-chat-7b-v3-1', 'config.json', revision='<non-existent-revision>')

    Request url:
        https:// {endpoint}/Intel/neural-chat-7b-v3-1/media/branch/{ non-existent-revision }/ config.json

    if revision not exist:
        RevisionNotFoundError: 1001 Client Error.
        Revision Not Found for url: https:// {endpoint}/Intel/neural-chat-7b-v3-1/media/branch/
        { non-existent-revision }/ config.json
        Invalid rev id: <non-existent-revision>
    """


class EntryNotFoundError(MdsHubHTTPError):
    """
    EntryNotFoundError
        >>> from mindseed.hub_utils.mds_api import ml_hub_download
        >>> ml_hub_download(' Intel/neural-chat-7b-v3-1', '<non-existent-file>')

    Request url:
        https://{endpoint}/Intel/neural-chat-7b-v3-1/media/branch/main/{ non-existent-file }

    if file do not exist:
        EntryNotFoundError: 1002 Client Error.
    """


class GatedRepoError(MdsHubHTTPError):
    """GatedRepoError
        >>> from mindseed.hub_utils.mds_api import model_info
        >>> model_info("<gated_repository>")

    Request url:
        https:// {endpoint}/api/models/{repo_id}

    if try to access a private repo but do not have the necessary permissions:
        GatedRepoError: 1003 Client Error.
        Cannot access gated repo for url https:// {endpoint}/api/models/<gated_repository>.
            Access to model <gated_repository> is restricted and you are not in the authorized list.
    """


class BadRequestError(MdsHubHTTPError, ValueError):
    """
    Raised by `mds_raise_for_status` when the server returns a HTTP 400 error.
    """


class LocalEntryNotFoundError(EntryNotFoundError, FileNotFoundError, ValueError):
    """
    Raised when trying to access a file that is not on the disk when network is
    disabled or unavailable (connection issue). The entry may exist on the Hub.
    """

    def __init__(self, message: str):
        super().__init__(message, response=None)


def mds_raise_for_status(response: Response, endpoint_name: Optional[str] = None) -> None:
    try:
        response.raise_for_status()
    except HTTPError as e:
        error_code = response.headers.get("X-Error-Code")

        if error_code == RevisionNotFound:
            message = f"{response.status_code} Client Error." + "\n\n" + f"Revision Not Found for url: {response.url}."
            raise RevisionNotFoundError(message, response) from e

        elif error_code == EntryNotFound:
            message = f"{response.status_code} Client Error." + "\n\n" + f"Entry Not Found for url: {response.url}."
            raise EntryNotFoundError(message, response) from e

        elif error_code == GatedRepo:
            message = (
                f"{response.status_code} Client Error." + "\n\n" + f"Cannot access gated repo for url {response.url}."
            )
            raise GatedRepoError(message, response) from e

        elif error_code == RepositoryNotFound:
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Repository Not Found for url: {response.url}."
                + "\nPlease make sure you specified the correct `repo_id` and"
                " `repo_type`.\nIf you are trying to access a private or gated repo,"
                " make sure you are authenticated."
            )
            raise RepositoryNotFoundError(message, response) from e

        elif response.status_code == 400:
            message = (
                f"\n\nBad request for {endpoint_name} endpoint:" if endpoint_name is not None else "\n\nBad request:"
            )
            raise BadRequestError(message, response=response) from e

        # Convert `HTTPError` into a `MdsHubHTTPError` to display request information
        # as well (request id and/or server error message)
        raise MdsHubHTTPError(str(e), response=response) from e
