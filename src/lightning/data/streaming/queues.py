# Copyright The Lightning AI team.
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

import multiprocessing
import os
import pickle
import queue  # needed as import instead from/import for mocking in tests
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin

import backoff
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout
from urllib3.util.retry import Retry


class BaseQueue(ABC):
    """Base Queue class that has a similar API to the Queue class in python."""

    @abstractmethod
    def __init__(self, name: str, default_timeout: float):
        self.name = name
        self.default_timeout = default_timeout

    @abstractmethod
    def put(self, item: Any) -> None:
        pass

    @abstractmethod
    def get(self, timeout: Optional[float] = None) -> Any:
        """Returns the left most element of the queue.

        Parameters
        ----------
        timeout:
            Read timeout in seconds, in case of input timeout is 0, the `self.default_timeout` is used.
            A timeout of None can be used to block indefinitely.

        """
        pass

    @property
    def is_running(self) -> bool:
        """Returns True if the queue is running, False otherwise.

        Child classes should override this property and implement custom logic as required

        """
        return True


class MultiProcessQueue(BaseQueue):
    def __init__(self, name: str, default_timeout: float) -> None:
        self.name = name
        self.default_timeout = default_timeout
        context = multiprocessing.get_context("spawn")
        self.queue = context.Queue()

    def put(self, item: Any) -> None:
        self.queue.put(item)

    def get(self, timeout: Optional[float] = None) -> Any:
        if timeout == 0:
            timeout = self.default_timeout
        return self.queue.get(timeout=timeout, block=(timeout is None))


_CONNECTION_RETRY_TOTAL = 2880
_CONNECTION_RETRY_BACKOFF_FACTOR = 0.5
_DEFAULT_REQUEST_TIMEOUT = 30  # seconds


class CustomRetryAdapter(HTTPAdapter):
    def __init__(self, *args: Any, **kwargs: Any):
        self.timeout = kwargs.pop("timeout", _DEFAULT_REQUEST_TIMEOUT)
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs: Any):
        kwargs["timeout"] = kwargs.get("timeout", self.timeout)
        return super().send(request, **kwargs)


def _http_method_logger_wrapper(func: Callable) -> Callable:
    """Returns the function decorated by a wrapper that logs the message using the `log_function` hook."""

    @wraps(func)
    def wrapped(self: "HTTPClient", *args: Any, **kwargs: Any) -> Any:
        message = f"HTTPClient: Method: {func.__name__.upper()}, Path: {args[0]}\n"
        message += f"      Base URL: {self.base_url}\n"
        params = kwargs.get("query_params", {})
        if params:
            message += f"      Params: {params}\n"
        resp: requests.Response = func(self, *args, **kwargs)
        message += f"      Response: {resp.status_code} {resp.reason}"
        self.log_function(message)
        return resp

    return wrapped


def _response(r, *args: Any, **kwargs: Any):
    return r.raise_for_status()


class HTTPClient:
    """A wrapper class around the requests library which handles chores like logging, retries, and timeouts
    automatically."""

    def __init__(
        self, base_url: str, auth_token: Optional[str] = None, log_callback: Optional[Callable] = None
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
        adapter = CustomRetryAdapter(max_retries=retry_strategy, timeout=_DEFAULT_REQUEST_TIMEOUT)
        self.session = requests.Session()

        self.session.hooks = {"response": _response}

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})

        self.log_function = log_callback or self.log_function

    @_http_method_logger_wrapper
    def get(self, path: str):
        url = urljoin(self.base_url, path)
        return self.session.get(url)

    @_http_method_logger_wrapper
    def post(self, path: str, *, query_params: Optional[Dict] = None, data: Optional[bytes] = None):
        url = urljoin(self.base_url, path)
        return self.session.post(url, data=data, params=query_params)

    @_http_method_logger_wrapper
    def delete(self, path: str):
        url = urljoin(self.base_url, path)
        return self.session.delete(url)

    def log_function(self, message: str, *args, **kwargs: Any):
        """This function is used to log the messages in the client, it can be overridden by caller to customise the
        logging logic.

        We enabled customisation here instead of just using `logger.debug` because HTTP logging can be very noisy, but
        it is crucial for finding bugs when we have them

        """
        pass


class HTTPQueue(BaseQueue):
    def __init__(self) -> None:
        self.client = HTTPClient()
        self.queue_url = ""

    @property
    def is_running(self) -> bool:
        """Pinging the http redis server to see if it is alive."""
        try:
            url = urljoin(self.queue_url, "health")
            resp = requests.get(
                url,
                timeout=1,
            )
            if resp.status_code == 200:
                return True
        except (ConnectionError, ConnectTimeout, ReadTimeout):
            return False
        return False

    def get(self, timeout: Optional[float] = None) -> Any:
        if not self.app_id:
            raise ValueError(f"App ID couldn't be extracted from the queue name: {self.name}")

        # it's a blocking call, we need to loop and call the backend to mimic this behavior
        if timeout is None:
            while True:
                try:
                    try:
                        return self._get()
                    except requests.exceptions.HTTPError:
                        pass
                except queue.Empty:
                    time.sleep(1)

        # make one request and return the result
        if timeout == 0:
            try:
                return self._get()
            except requests.exceptions.HTTPError:
                return None

        # timeout is some value - loop until the timeout is reached
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                try:
                    return self._get()
                except requests.exceptions.HTTPError:
                    if timeout > self.default_timeout:
                        return None
                    raise queue.Empty
            except queue.Empty:
                # Note: In theory, there isn't a need for a sleep as the queue shouldn't
                # block the flow if the queue is empty.
                # However, as the Http Server can saturate,
                # let's add a sleep here if a higher timeout is provided
                # than the default timeout
                if timeout > self.default_timeout:
                    time.sleep(0.05)
        return None

    def _get(self) -> Any:
        try:
            resp = self.client.post(f"v1/{self.app_id}/{self._name_suffix}", query_params={"action": "pop"})
            if resp.status_code == 204:
                raise queue.Empty
            return pickle.loads(resp.content)
        except ConnectionError:
            # Note: If the Http Queue service isn't available,
            # we consider the queue is empty to avoid failing the app.
            raise queue.Empty

    @backoff.on_exception(backoff.expo, (RuntimeError, requests.exceptions.HTTPError))
    def put(self, item: Any) -> None:
        if not self.app_id:
            raise ValueError(f"The Lightning App ID couldn't be extracted from the queue name: {self.name}")

        value = pickle.dumps(item)
        resp = self.client.post(f"v1/{self.app_id}/{self._name_suffix}", data=value, query_params={"action": "push"})
        if resp.status_code != 201:
            raise RuntimeError(f"Failed to push to queue: {self._name_suffix}")

    def length(self) -> int:
        if not self.app_id:
            raise ValueError(f"App ID couldn't be extracted from the queue name: {self.name}")

        try:
            val = self.client.get(f"/v1/{self.app_id}/{self._name_suffix}/length")
            return int(val.text)
        except requests.exceptions.HTTPError:
            return 0


class Broadcaster:
    def __init__(self):
        self.client: Optional[HTTPClient] = None

    def broadcast(self, key: str, obj: Any) -> Any:
        if os.getenv("LIGHTNING_APP_STATE_URL") is None:
            return obj
        if self.client is None:
            self.client = HTTPQueue(os.getenv("LIGHTNING_APP_STATE_URL"))
        return self._broadcast(key, obj)

    def _broadcast(self, key: str, obj: Any) -> Any:
        resp = self.client.post("/broadcast", data={"key": key, "obj": pickle.dumps(obj)})
        if resp.status_code != 201:
            raise RuntimeError("Failed to broadcast value.")
        return pickle.loads(resp.content)
