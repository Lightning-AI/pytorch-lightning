import logging
import multiprocessing
import pickle
import queue
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from lightning_app.core.constants import (
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_QUEUES_READ_DEFAULT_TIMEOUT,
    REDIS_WARNING_QUEUE_SIZE,
    STATE_UPDATE_TIMEOUT,
)
from lightning_app.utilities.imports import _is_redis_available, requires

if _is_redis_available():
    import redis

logger = logging.getLogger(__name__)


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


class QueuingSystem(Enum):
    SINGLEPROCESS = "singleprocess"
    MULTIPROCESS = "multiprocess"
    REDIS = "redis"

    def _get_queue(self, queue_name: str) -> "BaseQueue":
        if self == QueuingSystem.MULTIPROCESS:
            return MultiProcessQueue(queue_name, default_timeout=STATE_UPDATE_TIMEOUT)
        elif self == QueuingSystem.REDIS:
            return RedisQueue(queue_name, default_timeout=REDIS_QUEUES_READ_DEFAULT_TIMEOUT)
        else:
            return SingleProcessQueue(queue_name, default_timeout=STATE_UPDATE_TIMEOUT)

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

    def get_api_refresh_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{API_REFRESH_QUEUE_CONSTANT}" if queue_id else API_REFRESH_QUEUE_CONSTANT
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
        if queue_len >= REDIS_WARNING_QUEUE_SIZE:
            warnings.warn(
                f"The Redis Queue {self.name} length is larger than the "
                f"recommended length of {REDIS_WARNING_QUEUE_SIZE}. "
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
