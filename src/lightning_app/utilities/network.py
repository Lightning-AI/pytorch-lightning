import logging
import socket
import time
from functools import wraps
from typing import Any, Callable

import lightning_cloud
import requests
import urllib3
from lightning_cloud.rest_client import create_swagger_client, GridRestClient
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


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


def _check_service_url_is_ready(url: str, timeout: float = 0.1) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code in (200, 404)
    except (ConnectionError, ConnectTimeout, ReadTimeout):
        return False


def _get_next_backoff_time(num_retries: int, backoff_value: float = 0.5) -> float:
    next_backoff_value = backoff_value * (2 ** (num_retries - 1))
    return min(_DEFAULT_BACKOFF_MAX, next_backoff_value)


def _retry_wrapper(func: Callable) -> Callable:
    """Returns the function decorated by a wrapper that retries the call several times if a connection error occurs.

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


class _MethodsWrapper(type):
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


class LightningClient(GridRestClient, metaclass=_MethodsWrapper):
    """The LightningClient is a wrapper around the GridRestClient.

    It wraps all methods to monitor connection exceptions and employs a retry strategy.
    """

    def __init__(self) -> None:
        super().__init__(api_client=create_swagger_client())
