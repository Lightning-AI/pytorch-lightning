import socket
import time
from functools import wraps
from typing import Any, Callable, Optional, Dict

import lightning_cloud
import requests
import urllib3
from lightning_cloud.rest_client import create_swagger_client, GridRestClient
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout
from urllib3.util.retry import Retry
from urllib.parse import urljoin, urlencode

from core.constants import get_lightning_cloud_url, HTTP_QUEUE_URL
from lightning_app.utilities.app_helpers import Logger

logger = Logger(__name__)


def find_free_network_port() -> int:
    """Finds a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_CONNECTION_RETRY_TOTAL = 5
_CONNECTION_RETRY_BACKOFF_FACTOR = 0.5
_DEFAULT_BACKOFF_MAX = 5 * 60
_DEFAULT_REQUEST_TIMEOUT = 5


def _configure_session() -> Session:
    """Configures the session for GET and POST requests.

    It enables a generous retrial strategy that waits for the application server to connect.
    """
    retry_strategy = Retry(
        # wait time between retries increases exponentially according to: backoff_factor * (2 ** (retry - 1))
        total=_CONNECTION_RETRY_TOTAL,
        backoff_factor=_CONNECTION_RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    return http


def _check_service_url_is_ready(url: str, timeout: float = 5) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code in (200, 404)
    except (ConnectionError, ConnectTimeout, ReadTimeout):
        logger.debug(f"The url {url} is not ready.")
        return False


def _get_next_backoff_time(num_retries: int, backoff_value: float = 0.5) -> float:
    next_backoff_value = backoff_value * (2 ** (num_retries - 1))
    return min(_DEFAULT_BACKOFF_MAX, next_backoff_value)


def _retry_wrapper(func: Callable) -> Callable:
    """Returns the function decorated by a wrapper that retries the call several times if a connection error
    occurs.

    The retries follow an exponential backoff.
    """

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        consecutive_errors = 0
        while _get_next_backoff_time(consecutive_errors) != _DEFAULT_BACKOFF_MAX:
            try:
                return func(*args, **kwargs)
            except lightning_cloud.openapi.rest.ApiException as e:
                # retry if the control plane fails with all errors except 4xx but not 408 - (Request Timeout)
                if e.status == 408 or e.status == 409 or not str(e.status).startswith("4"):
                    consecutive_errors += 1
                    backoff_time = _get_next_backoff_time(consecutive_errors)
                    logger.debug(
                        f"The {func.__name__} request failed to reach the server, got a response {e.status}."
                        f" Retrying after {backoff_time} seconds."
                    )
                    time.sleep(backoff_time)
                else:
                    raise e
            except urllib3.exceptions.HTTPError as e:
                consecutive_errors += 1
                backoff_time = _get_next_backoff_time(consecutive_errors)
                logger.debug(
                    f"The {func.__name__} request failed to reach the server, got a an error {str(e)}."
                    f" Retrying after {backoff_time} seconds."
                )
                time.sleep(backoff_time)

        raise Exception(f"The default maximum backoff {_DEFAULT_BACKOFF_MAX} seconds has been reached.")

    return wrapped


class _MethodsRetryWrapperMeta(type):
    """This wrapper metaclass iterates through all methods of the type and all bases of it to wrap them into the
    :func:`_retry_wrapper`. It applies to all bound callables except the ``__init__`` method.
    """

    def __new__(mcs, name, bases, dct):
        new_class = super().__new__(mcs, name, bases, dct)
        for base in new_class.__mro__[1:-1]:
            for key, value in base.__dict__.items():
                if callable(value) and value.__name__ != "__init__":
                    setattr(new_class, key, _retry_wrapper(value))
        return new_class


class LightningClient(GridRestClient, metaclass=_MethodsRetryWrapperMeta):
    """The LightningClient is a wrapper around the GridRestClient.

    It wraps all methods to monitor connection exceptions and employs a retry strategy.
    """

    def __init__(self) -> None:
        super().__init__(api_client=create_swagger_client())


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.pop("timeout", _DEFAULT_REQUEST_TIMEOUT)
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        kwargs["timeout"] = kwargs.get("timeout", self.timeout)
        return super().send(request, **kwargs)


def _http_method_logger_wrapper(name: str, func: Callable) -> Callable:
    """Returns the function decorated by a wrapper that logs the message using the `log_function` hook
    """

    @wraps(func)
    def wrapped(self: 'HTTPClient', *args: Any, **kwargs: Any) -> Any:
        self.log_function(f"HTTP call from HTTPClient. Method: {name.upper()}, Base URL: {self.base_url}, Path: {args[0]}")
        params = kwargs.get("query_params", {})
        if params:
            self.log_function(f"Params: {params}")
        resp: requests.Response = func(self, *args, **kwargs)
        self.log_function(f"Response Status Code: {resp.status_code}")
    return wrapped


class _MethodsDebuLoggingWrapperMeta(type):
    """This metaclass wraps all the non-private methods of the child class with :func:`_http_method_logger_wrapper` so
    they log the request life cycle, depends on the debug level set
    """

    def __new__(mcs, name, bases, dct):
        new_class = super().__new__(mcs, name, bases, dct)
        for base in new_class.__mro__[1:-1]:
            for key, value in base.__dict__.items():
                if callable(value) and not key.startswith("_") and key != "log_function":
                    setattr(new_class, key, _http_method_logger_wrapper(key, value))
        return new_class


class HTTPClient(metaclass=_MethodsDebuLoggingWrapperMeta):

    def __init__(self, base_url: Optional[str] = None, log_callback: Optional[Callable] = None) -> None:
        self.base_url = base_url or HTTP_QUEUE_URL
        retry_strategy = Retry(
            # wait time between retries increases exponentially according to: backoff_factor * (2 ** (retry - 1))
            total=_CONNECTION_RETRY_TOTAL,
            backoff_factor=_CONNECTION_RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = TimeoutHTTPAdapter(max_retries=retry_strategy, timeout=_DEFAULT_REQUEST_TIMEOUT)
        sess = requests.Session()
        sess.mount("http://", adapter)  # noqa
        sess.mount("https://", adapter)
        self.client: requests.Session = sess
        self.log_function = log_callback or self.log_function

    def get(self, path: str):
        url = urljoin(self.base_url, path)
        return self.client.get(url)

    def post(self, path: str, *, query_params: Optional[Dict] = None, data: Optional[bytes] = None):
        url = urljoin(self.base_url, path) + urlencode(query_params or {})
        return self.client.post(url, data=data)

    def delete(self, path: str):
        url = urljoin(self.base_url, path)
        return self.client.delete(url)

    def log_function(self, message: str, *args, **kwargs):
        """ This function is used to log the messages in the client, it can be overridden by caller to customise
        the logging logic. We enabled customisation here instead of just using `logger.debug` because HTTP logging
        can be very noisy, but it is crucial for finding bugs when we have them
        """
        pass

    def _update_headers(self, headers: Dict):
        """ Updates the session headers with the provided headers. Note that this function doesn't replace
        the exising headers but updates them """
        self.client.headers.update(headers)
