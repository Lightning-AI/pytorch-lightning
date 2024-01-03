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

import base64
import multiprocessing
import pickle
import queue  # needed as import instead from/import for mocking in tests
import time
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urljoin

import backoff
import requests
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout

from lightning.app.core.constants import (
    BATCH_DELTA_COUNT,
    HTTP_QUEUE_REFRESH_INTERVAL,
    HTTP_QUEUE_REQUESTS_PER_SECOND,
    HTTP_QUEUE_TOKEN,
    HTTP_QUEUE_URL,
    LIGHTNING_DIR,
    QUEUE_DEBUG_ENABLED,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_QUEUES_READ_DEFAULT_TIMEOUT,
    STATE_UPDATE_TIMEOUT,
    WARNING_QUEUE_SIZE,
)
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.imports import _is_redis_available, requires
from lightning.app.utilities.network import HTTPClient

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
FLOW_TO_WORKS_DELTA_QUEUE_CONSTANT = "FLOW_TO_WORKS_DELTA_QUEUE"


class QueuingSystem(Enum):
    MULTIPROCESS = "multiprocess"
    REDIS = "redis"
    HTTP = "http"

    def get_queue(self, queue_name: str) -> "BaseQueue":
        if self == QueuingSystem.MULTIPROCESS:
            return MultiProcessQueue(queue_name, default_timeout=STATE_UPDATE_TIMEOUT)
        if self == QueuingSystem.REDIS:
            return RedisQueue(queue_name, default_timeout=REDIS_QUEUES_READ_DEFAULT_TIMEOUT)
        return RateLimitedQueue(
            HTTPQueue(queue_name, default_timeout=STATE_UPDATE_TIMEOUT), HTTP_QUEUE_REQUESTS_PER_SECOND
        )

    def get_api_response_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{API_RESPONSE_QUEUE_CONSTANT}" if queue_id else API_RESPONSE_QUEUE_CONSTANT
        return self.get_queue(queue_name)

    def get_readiness_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{READINESS_QUEUE_CONSTANT}" if queue_id else READINESS_QUEUE_CONSTANT
        return self.get_queue(queue_name)

    def get_delta_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{DELTA_QUEUE_CONSTANT}" if queue_id else DELTA_QUEUE_CONSTANT
        return self.get_queue(queue_name)

    def get_error_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{ERROR_QUEUE_CONSTANT}" if queue_id else ERROR_QUEUE_CONSTANT
        return self.get_queue(queue_name)

    def get_has_server_started_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{HAS_SERVER_STARTED_CONSTANT}" if queue_id else HAS_SERVER_STARTED_CONSTANT
        return self.get_queue(queue_name)

    def get_caller_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{CALLER_QUEUE_CONSTANT}_{work_name}" if queue_id else f"{CALLER_QUEUE_CONSTANT}_{work_name}"
        )
        return self.get_queue(queue_name)

    def get_api_state_publish_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{API_STATE_PUBLISH_QUEUE_CONSTANT}" if queue_id else API_STATE_PUBLISH_QUEUE_CONSTANT
        return self.get_queue(queue_name)

    # TODO: This is hack, so we can remove this queue entirely when fully optimized.
    def get_api_delta_queue(self, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = f"{queue_id}_{DELTA_QUEUE_CONSTANT}" if queue_id else DELTA_QUEUE_CONSTANT
        return self.get_queue(queue_name)

    def get_orchestrator_request_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{ORCHESTRATOR_REQUEST_CONSTANT}_{work_name}"
            if queue_id
            else f"{ORCHESTRATOR_REQUEST_CONSTANT}_{work_name}"
        )
        return self.get_queue(queue_name)

    def get_orchestrator_response_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{ORCHESTRATOR_RESPONSE_CONSTANT}_{work_name}"
            if queue_id
            else f"{ORCHESTRATOR_RESPONSE_CONSTANT}_{work_name}"
        )
        return self.get_queue(queue_name)

    def get_orchestrator_copy_request_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{ORCHESTRATOR_COPY_REQUEST_CONSTANT}_{work_name}"
            if queue_id
            else f"{ORCHESTRATOR_COPY_REQUEST_CONSTANT}_{work_name}"
        )
        return self.get_queue(queue_name)

    def get_orchestrator_copy_response_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{ORCHESTRATOR_COPY_RESPONSE_CONSTANT}_{work_name}"
            if queue_id
            else f"{ORCHESTRATOR_COPY_RESPONSE_CONSTANT}_{work_name}"
        )
        return self.get_queue(queue_name)

    def get_work_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{WORK_QUEUE_CONSTANT}_{work_name}" if queue_id else f"{WORK_QUEUE_CONSTANT}_{work_name}"
        )
        return self.get_queue(queue_name)

    def get_flow_to_work_delta_queue(self, work_name: str, queue_id: Optional[str] = None) -> "BaseQueue":
        queue_name = (
            f"{queue_id}_{FLOW_TO_WORKS_DELTA_QUEUE_CONSTANT}_{work_name}"
            if queue_id
            else f"{FLOW_TO_WORKS_DELTA_QUEUE_CONSTANT}_{work_name}"
        )
        return self.get_queue(queue_name)


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

    @abstractmethod
    def batch_get(self, timeout: Optional[float] = None, count: Optional[int] = None) -> List[Any]:
        """Returns the left most elements of the queue.

        Parameters
        ----------
        timeout:
            Read timeout in seconds, in case of input timeout is 0, the `self.default_timeout` is used.
            A timeout of None can be used to block indefinitely.
        count:
            The number of element to get from the queue

        """

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

    def batch_get(self, timeout: Optional[float] = None, count: Optional[int] = None) -> List[Any]:
        if timeout == 0:
            timeout = self.default_timeout
        # For multiprocessing, we can simply collect the latest upmost element
        return [self.queue.get(timeout=timeout, block=(timeout is None))]


class RedisQueue(BaseQueue):
    @requires("redis")
    def __init__(
        self,
        name: str,
        default_timeout: float,
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
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
        self.host = host or REDIS_HOST
        self.port = port or REDIS_PORT
        self.password = password or REDIS_PASSWORD
        self.name = name
        self.default_timeout = default_timeout
        self.redis = redis.Redis(host=self.host, port=self.port, password=self.password)

    def put(self, item: Any) -> None:
        from lightning.app.core.work import LightningWork

        is_work = isinstance(item, LightningWork)

        # TODO: Be careful to handle with a lock if another thread needs
        # to access the work backend one day.
        # The backend isn't picklable
        # Raises a TypeError: cannot pickle '_thread.RLock' object
        if is_work:
            backend = item._backend
            item._backend = None

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

        # The backend isn't pickable.
        if is_work:
            item._backend = backend

    def get(self, timeout: Optional[float] = None) -> Any:
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

    def batch_get(self, timeout: Optional[float] = None, count: Optional[int] = None) -> Any:
        return [self.get(timeout=timeout)]

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

    @property
    def is_running(self) -> bool:
        """Pinging the redis server to see if it is alive."""
        try:
            return self.redis.ping()
        except redis.exceptions.ConnectionError:
            return False

    def to_dict(self) -> dict:
        return {
            "type": "redis",
            "name": self.name,
            "default_timeout": self.default_timeout,
            "host": self.host,
            "port": self.port,
            "password": self.password,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "RedisQueue":
        return cls(**state)


class RateLimitedQueue(BaseQueue):
    def __init__(self, queue: BaseQueue, requests_per_second: float):
        """This is a queue wrapper that will block on get or put calls if they are made too quickly.

        Args:
            queue: The queue to wrap.
            requests_per_second: The target number of get or put requests per second.

        """
        self.name = queue.name
        self.default_timeout = queue.default_timeout

        self._queue = queue
        self._seconds_per_request = 1 / requests_per_second

        self._last_get = 0.0

    @property
    def is_running(self) -> bool:
        return self._queue.is_running

    def _wait_until_allowed(self, last_time: float) -> None:
        t = time.time()
        diff = t - last_time
        if diff < self._seconds_per_request:
            time.sleep(self._seconds_per_request - diff)

    def get(self, timeout: Optional[float] = None) -> Any:
        self._wait_until_allowed(self._last_get)
        self._last_get = time.time()
        return self._queue.get(timeout=timeout)

    def batch_get(self, timeout: Optional[float] = None, count: Optional[int] = None) -> Any:
        self._wait_until_allowed(self._last_get)
        self._last_get = time.time()
        return self._queue.batch_get(timeout=timeout)

    def put(self, item: Any) -> None:
        return self._queue.put(item)


class HTTPQueue(BaseQueue):
    def __init__(self, name: str, default_timeout: float) -> None:
        """
        Parameters
        ----------
        name:
            The name of the Queue to use. In the current implementation, we expect the name to be of the format
            `appID_queueName`. Based on this assumption, we try to fetch the app id and the queue name by splitting
            the `name` argument.
        default_timeout:
            Default timeout for redis read
        """
        if name is None:
            raise ValueError("You must specify a name for the queue")
        self.app_id, self._name_suffix = self._split_app_id_and_queue_name(name)
        self.name = name  # keeping the name for debugging
        self.default_timeout = default_timeout
        self.client = HTTPClient(base_url=HTTP_QUEUE_URL, auth_token=HTTP_QUEUE_TOKEN, log_callback=debug_log_callback)

    @property
    def is_running(self) -> bool:
        """Pinging the http redis server to see if it is alive."""
        try:
            url = urljoin(HTTP_QUEUE_URL, "health")
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {HTTP_QUEUE_TOKEN}"},
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
                    time.sleep(HTTP_QUEUE_REFRESH_INTERVAL)

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

    def batch_get(self, timeout: Optional[float] = None, count: Optional[int] = None) -> List[Any]:
        try:
            resp = self.client.post(
                f"v1/{self.app_id}/{self._name_suffix}",
                query_params={"action": "popCount", "count": str(count or BATCH_DELTA_COUNT)},
            )
            if resp.status_code == 204:
                raise queue.Empty
            return [pickle.loads(base64.b64decode(data)) for data in resp.json()]
        except ConnectionError:
            # Note: If the Http Queue service isn't available,
            # we consider the queue is empty to avoid failing the app.
            raise queue.Empty

    @backoff.on_exception(backoff.expo, (RuntimeError, requests.exceptions.HTTPError))
    def put(self, item: Any) -> None:
        if not self.app_id:
            raise ValueError(f"The Lightning App ID couldn't be extracted from the queue name: {self.name}")

        value = pickle.dumps(item)
        queue_len = self.length()
        if queue_len >= WARNING_QUEUE_SIZE:
            warnings.warn(
                f"The Queue {self._name_suffix} length is larger than the recommended length of {WARNING_QUEUE_SIZE}. "
                f"Found {queue_len}. This might cause your application to crash, please investigate this."
            )
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

    @staticmethod
    def _split_app_id_and_queue_name(queue_name: str) -> Tuple[str, str]:
        """This splits the app id and the queue name into two parts.

        This can be brittle, as if the queue name creation logic changes, the response values from here wouldn't be
        accurate. Remove this eventually and let the Queue class take app id and name of the queue as arguments

        """
        if "_" not in queue_name:
            return "", queue_name
        app_id, queue_name = queue_name.split("_", 1)
        return app_id, queue_name

    def to_dict(self) -> dict:
        return {
            "type": "http",
            "name": self.name,
            "default_timeout": self.default_timeout,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "HTTPQueue":
        return cls(**state)


def debug_log_callback(message: str, *args: Any, **kwargs: Any) -> None:
    if QUEUE_DEBUG_ENABLED or (Path(LIGHTNING_DIR) / "QUEUE_DEBUG_ENABLED").exists():
        logger.info(message, *args, **kwargs)
