import multiprocessing
import pickle
import queue
import time
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from lightning_app.core.constants import (
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_QUEUES_READ_DEFAULT_TIMEOUT,
    WARNING_QUEUE_SIZE,
    STATE_UPDATE_TIMEOUT,
)
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.imports import _is_redis_available, requires
from core.constants import QUEUE_DEBUG_ENABLED, LIGHTNING_DIR
from utilities.network import HTTPClient

if _is_redis_available():
    import redis

logger = Logger(__name__)


READINESS_QUEUE_CONSTANT = "READINESS_QUEUE"
ERROR_QUEUE_CONSTANT = "ERROR_QUEUE"
DELTA_QUEUE_CONSTANT = "DELTA_QUEUE"
HAS_SERVER_STARTED_CONSTANT = "HAS_SERVER_STARTED_QUEUE"
CALLER_QUEUE_CONSTANT = "CALLER_QUEUE"
API_STATE_PUBLISH_QUEUE_CONSTANT = "API_STATE_PUBLISH_QUEUE"
API_DELTA_QUEUE_CONSTANT = "API_DELTA_QUEUE"
API_REFRESH_QUEUE_CONSTANT = "API_REFRESH_QUEUE"
ORCHESTRATOR_REQUEST_CONSTANT = "ORCHESTRATOR_REQUEST"
ORCHESTRATOR_RESPONSE_CONSTANT = "ORCHESTRATOR_RESPONSE"
ORCHESTRATOR_COPY_REQUEST_CONSTANT = "ORCHESTRATOR_COPY_REQUEST"
ORCHESTRATOR_COPY_RESPONSE_CONSTANT = "ORCHESTRATOR_COPY_RESPONSE"
WORK_QUEUE_CONSTANT = "WORK_QUEUE"
API_RESPONSE_QUEUE_CONSTANT = "API_RESPONSE_QUEUE"


class QueuingSystem(Enum):
    SINGLEPROCESS = "singleprocess"
    MULTIPROCESS = "multiprocess"
    REDIS = "redis"
    HTTP = "http"

    def _get_queue(self, queue_name: str) -> "BaseQueue":
        if self == QueuingSystem.MULTIPROCESS:
            return MultiProcessQueue(queue_name, default_timeout=STATE_UPDATE_TIMEOUT)
        elif self == QueuingSystem.REDIS:
            return RedisQueue(queue_name, default_timeout=REDIS_QUEUES_READ_DEFAULT_TIMEOUT)
        elif self == QueuingSystem.HTTP:
            return HTTPQueue(queue_name, default_timeout=STATE_UPDATE_TIMEOUT)
        else:
            return SingleProcessQueue(queue_name, default_timeout=STATE_UPDATE_TIMEOUT)

    def get_api_response_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{API_RESPONSE_QUEUE_CONSTANT}" if queue_id else API_RESPONSE_QUEUE_CONSTANT
        return self._get_queue(queue_name)

    def get_readiness_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{READINESS_QUEUE_CONSTANT}" if queue_id else READINESS_QUEUE_CONSTANT
        return self._get_queue(queue_name)

    def get_delta_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{DELTA_QUEUE_CONSTANT}" if queue_id else DELTA_QUEUE_CONSTANT
        return self._get_queue(queue_name)

    def get_error_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{ERROR_QUEUE_CONSTANT}" if queue_id else ERROR_QUEUE_CONSTANT
        return self._get_queue(queue_name)

    def get_has_server_started_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{HAS_SERVER_STARTED_CONSTANT}" if queue_id else HAS_SERVER_STARTED_CONSTANT
        return self._get_queue(queue_name)

    def get_caller_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{CALLER_QUEUE_CONSTANT}_{work_name}" if queue_id else f"{CALLER_QUEUE_CONSTANT}_{work_name}"
        )
        return self._get_queue(queue_name)

    def get_api_state_publish_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{API_STATE_PUBLISH_QUEUE_CONSTANT}" if queue_id else API_STATE_PUBLISH_QUEUE_CONSTANT
        return self._get_queue(queue_name)

    def get_api_delta_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{API_DELTA_QUEUE_CONSTANT}" if queue_id else API_DELTA_QUEUE_CONSTANT
        return self._get_queue(queue_name)

    def get_orchestrator_request_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{ORCHESTRATOR_REQUEST_CONSTANT}_{work_name}"
            if queue_id
            else f"{ORCHESTRATOR_REQUEST_CONSTANT}_{work_name}"
        )
        return self._get_queue(queue_name)

    def get_orchestrator_response_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{ORCHESTRATOR_RESPONSE_CONSTANT}_{work_name}"
            if queue_id
            else f"{ORCHESTRATOR_RESPONSE_CONSTANT}_{work_name}"
        )
        return self._get_queue(queue_name)

    def get_orchestrator_copy_request_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{ORCHESTRATOR_COPY_REQUEST_CONSTANT}_{work_name}"
            if queue_id
            else f"{ORCHESTRATOR_COPY_REQUEST_CONSTANT}_{work_name}"
        )
        return self._get_queue(queue_name)

    def get_orchestrator_copy_response_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{ORCHESTRATOR_COPY_RESPONSE_CONSTANT}_{work_name}"
            if queue_id
            else f"{ORCHESTRATOR_COPY_RESPONSE_CONSTANT}_{work_name}"
        )
        return self._get_queue(queue_name)

    def get_work_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{WORK_QUEUE_CONSTANT}_{work_name}" if queue_id else f"{WORK_QUEUE_CONSTANT}_{work_name}"
        )
        return self._get_queue(queue_name)


class BaseQueue(ABC):
    """Base Queue class that has a similar API to the Queue class in python."""

    @abstractmethod
    def __init__(self, name: str, default_timeout: float):
        self.name = name
        self.default_timeout = default_timeout

    @abstractmethod
    def put(self, item):
        pass

    @abstractmethod
    def get(self, timeout: int = None):
        """Returns the left most element of the queue.

        Parameters
        ----------
        timeout:
            Read timeout in seconds, in case of input timeout is 0, the `self.default_timeout` is used.
            A timeout of None can be used to block indefinitely.
        """
        pass


class SingleProcessQueue(BaseQueue):
    def __init__(self, name: str, default_timeout: float):
        self.name = name
        self.default_timeout = default_timeout
        self.queue = queue.Queue()

    def put(self, item):
        self.queue.put(item)

    def get(self, timeout: int = None):
        if timeout == 0:
            timeout = self.default_timeout
        return self.queue.get(timeout=timeout, block=(timeout is None))


class MultiProcessQueue(BaseQueue):
    def __init__(self, name: str, default_timeout: float):
        self.name = name
        self.default_timeout = default_timeout
        self.queue = multiprocessing.Queue()

    def put(self, item):
        self.queue.put(item)

    def get(self, timeout: int = None):
        if timeout == 0:
            timeout = self.default_timeout
        return self.queue.get(timeout=timeout, block=(timeout is None))


class RedisQueue(BaseQueue):
    @requires("redis")
    def __init__(
        self,
        name: str,
        default_timeout: float,
        host: str = None,
        port: int = None,
        password: str = None,
    ):
        """
        Parameters
        ----------
        name:
            The name of the list to use
        default_timeout:
            Default timeout for redis read
        host:
            The hostname of the redis server
        port:
            The port of the redis server
        password:
            Redis password
        """
        if name is None:
            raise ValueError("You must specify a name for the queue")
        host = host or REDIS_HOST
        port = port or REDIS_PORT
        password = password or REDIS_PASSWORD
        self.name = name
        self.default_timeout = default_timeout
        self.redis = redis.Redis(host=host, port=port, password=password)

    def ping(self):
        """Ping the redis server to see if it is alive."""
        try:
            return self.redis.ping()
        except redis.exceptions.ConnectionError:
            return False

    def put(self, item: Any) -> None:
        value = pickle.dumps(item)
        queue_len = self.length()
        if queue_len >= WARNING_QUEUE_SIZE:
            warnings.warn(
                f"The Redis Queue {self.name} length is larger than the "
                f"recommended length of {WARNING_QUEUE_SIZE}. "
                f"Found {queue_len}. This might cause your application to crash, "
                "please investigate this."
            )
        try:
            self.redis.rpush(self.name, value)
        except redis.exceptions.ConnectionError:
            raise ConnectionError(
                "Your app failed because it couldn't connect to Redis. "
                "Please try running your app again. "
                "If the issue persists, please contact support@lightning.ai"
            )

    def get(self, timeout: int = None):
        """Returns the left most element of the redis queue.

        Parameters
        ----------
        timeout:
            Read timeout in seconds, in case of input timeout is 0, the `self.default_timeout` is used.
            A timeout of None can be used to block indefinitely.
        """

        if timeout is None:
            # this means it's blocking in redis
            timeout = 0
        elif timeout == 0:
            timeout = self.default_timeout

        try:
            out = self.redis.blpop([self.name], timeout=timeout)
        except redis.exceptions.ConnectionError:
            raise ConnectionError(
                "Your app failed because it couldn't connect to Redis. "
                "Please try running your app again. "
                "If the issue persists, please contact support@lightning.ai"
            )

        if out is None:
            raise queue.Empty
        return pickle.loads(out[1])

    def clear(self) -> None:
        """Clear all elements in the queue."""
        self.redis.delete(self.name)

    def length(self) -> int:
        """Returns the number of elements in the queue."""
        try:
            return self.redis.llen(self.name)
        except redis.exceptions.ConnectionError:
            raise ConnectionError(
                "Your app failed because it couldn't connect to Redis. "
                "Please try running your app again. "
                "If the issue persists, please contact support@lightning.ai"
            )

# TODO - exception handling in general for HTTP and pickling


class HTTPQueue(BaseQueue):
    def __init__(self, name: str, default_timeout: float, base_url: Optional[str] = None):
        """
        Parameters
        ----------
        name:
            The name of the list to use
        default_timeout:
            Default timeout for redis read
        """
        if name is None:
            raise ValueError("You must specify a name for the queue")
        self.app_id, self.name = self._split_app_id_and_queue_name(name)
        self.default_timeout = default_timeout
        self.client = HTTPClient(base_url=base_url, log_callback=debug_log_callback)

    def get(self, timeout: int = None) -> Any:
        if timeout is None:
            # it's a blocking call, we need to loop and call the backend to mimic this behavior
            while True:
                try:
                    return self._get()
                except queue.Empty:
                    # longer sleep because we are waiting forever
                    time.sleep(1)

        if timeout == 0:
            return self._get()

        # timeout is some value - loop until the timeout is reached
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                return self._get()
            except queue.Empty:
                # shorter sleep because the wait is bounded
                time.sleep(0.1)

    def _get(self):
        resp = self.client.post(f"v1/{self.app_id}/{self.name}")
        if resp.status_code == 204:
            raise queue.Empty
        return pickle.loads(resp.content)

    def put(self, item: Any) -> None:
        value = pickle.dumps(item)
        queue_len = self.length()
        if queue_len >= WARNING_QUEUE_SIZE:
            warnings.warn(
                f"The Queue {self.name} length is larger than the recommended length of {WARNING_QUEUE_SIZE}. "
                f"Found {queue_len}. This might cause your application to crash, please investigate this."
            )
        self.client.post(f"v1/{self.app_id}/{self.name}?action=put", data=value)

    def length(self):
        val = self.client.get(f"/v1/{self.app_id}/{self.name}/length")
        if val.status_code != 200:
            pass
        return int(val.text)

    @staticmethod
    def _split_app_id_and_queue_name(queue_name):
        """ This splits the app id and the queue name into two parts. This can be brittle, as if the queue name
        creation logic changes, the response values from here wouldn't be accurate
        """
        app_id, queue_name = queue_name.split("_", 1)
        return app_id, queue_name


def debug_log_callback(message: str, *args: Any, **kwargs: Any) -> None:
    if QUEUE_DEBUG_ENABLED or "QUEUE_DEBUG_ENABLED" in os.listdir(LIGHTNING_DIR):
        logger.info(message, *args, **kwargs)