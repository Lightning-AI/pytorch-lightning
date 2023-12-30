# Copyright The Lightning AI team.
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

import json
import os
import pickle
from logging import Logger
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin

import requests
import urllib3

# for backwards compatibility
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = Logger(__name__)

_CONNECTION_RETRY_TOTAL = 2880
_CONNECTION_RETRY_BACKOFF_FACTOR = 0.5
_DEFAULT_REQUEST_TIMEOUT = 30  # seconds


class _CustomRetryAdapter(HTTPAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout = kwargs.pop("timeout", _DEFAULT_REQUEST_TIMEOUT)
        super().__init__(*args, **kwargs)

    def send(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["timeout"] = kwargs.get("timeout", self.timeout)
        return super().send(request, **kwargs)


def _response(r: Any, *args: Any, **kwargs: Any) -> Any:
    return r.raise_for_status()


class _HTTPClient:
    """A wrapper class around the requests library which handles chores like logging, retries, and timeouts
    automatically."""

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        log_callback: Optional[Callable] = None,
        use_retry: bool = True,
    ) -> None:
        self.base_url = base_url
        retry_strategy = Retry(
            # wait time between retries increases exponentially according to: backoff_factor * (2 ** (retry - 1))
            # but the the maximum wait time is 120 secs. By setting a large value (2880), we'll make sure clients
            # are going to be alive for a very long time (~ 4 days) but retries every 120 seconds
            total=_CONNECTION_RETRY_TOTAL,
            backoff_factor=_CONNECTION_RETRY_BACKOFF_FACTOR,
            status_forcelist=[
                408,  # Request Timeout
                429,  # Too Many Requests
                500,  # Internal Server Error
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
            ],
        )
        adapter = _CustomRetryAdapter(max_retries=retry_strategy, timeout=_DEFAULT_REQUEST_TIMEOUT)
        self.session = requests.Session()

        self.session.hooks = {"response": _response}

        if use_retry:
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})

    def get(self, path: str) -> Any:
        url = urljoin(self.base_url, path)
        return self.session.get(url)

    def post(
        self, path: str, *, query_params: Optional[Dict] = None, data: Optional[bytes] = None, json: Any = None
    ) -> Any:
        url = urljoin(self.base_url, path)
        return self.session.post(url, data=data, params=query_params, json=json)

    def delete(self, path: str) -> Any:
        url = urljoin(self.base_url, path)
        return self.session.delete(url)


class _ImmutableDistributedMap:
    """The _ImmutableDistributedMap enables to create a distributed key value pair in the cloud.

    The first process to perform the set operation defines its value.

    """

    def __init__(self) -> None:
        token = _get_token()

        lightning_app_external_url = os.getenv("LIGHTNING_APP_EXTERNAL_URL")
        if lightning_app_external_url is None:
            raise RuntimeError("The `LIGHTNING_APP_EXTERNAL_URL` should be set.")

        self.public_client: _HTTPClient = _HTTPClient(lightning_app_external_url, auth_token=token, use_retry=False)

        lightning_app_state_url = os.getenv("LIGHTNING_APP_STATE_URL")
        if lightning_app_state_url is None:
            raise RuntimeError("The `LIGHTNING_APP_STATE_URL` should be set.")

        self.private_client: _HTTPClient = _HTTPClient(lightning_app_state_url, auth_token=token, use_retry=False)

    def set_and_get(self, key: str, value: Any) -> Any:
        payload = {"key": key, "value": pickle.dumps(value, 0).decode()}

        # Try the public address first
        try:
            resp = self.public_client.post("/broadcast", json=payload)
        except (requests.exceptions.ConnectionError, urllib3.exceptions.MaxRetryError):
            # fallback to the private one
            resp = self.private_client.post("/broadcast", json=payload)

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to broadcast the following {key=} {value=}.")
        return pickle.loads(bytes(resp.json()["value"], "utf-8"))


def broadcast_object(key: str, obj: Any) -> Any:
    """This function enables to broadcast object across machines."""
    if os.getenv("LIGHTNING_APP_EXTERNAL_URL") is not None:
        return _ImmutableDistributedMap().set_and_get(key, obj)
    return obj


def _get_token() -> Optional[str]:
    """This function tries to retrieve a temporary token."""
    if os.getenv("LIGHTNING_CLOUD_URL") is None:
        return None

    payload = {"apiKey": os.getenv("LIGHTNING_API_KEY"), "username": os.getenv("LIGHTNING_USERNAME")}
    url_login = os.getenv("LIGHTNING_CLOUD_URL", "") + "/v1/auth/login"
    res = requests.post(url_login, data=json.dumps(payload))
    if "token" not in res.json():
        raise RuntimeError(
            f"You haven't properly setup your environment variables with {url_login} and data: \n{payload}"
        )
    return res.json()["token"]
