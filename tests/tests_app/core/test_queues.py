import base64
import multiprocessing
import pickle
import queue
import time
from unittest import mock

import pytest
import requests_mock
from lightning.app import LightningFlow
from lightning.app.core import queues
from lightning.app.core.constants import HTTP_QUEUE_URL, STATE_UPDATE_TIMEOUT
from lightning.app.core.queues import (
    READINESS_QUEUE_CONSTANT,
    BaseQueue,
    HTTPQueue,
    QueuingSystem,
    RateLimitedQueue,
    RedisQueue,
)
from lightning.app.utilities.imports import _is_redis_available
from lightning.app.utilities.redis import check_if_redis_running


@pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running")
@pytest.mark.parametrize("queue_type", [QueuingSystem.REDIS, QueuingSystem.MULTIPROCESS])
def test_queue_api(queue_type, monkeypatch):
    """Test the Queue API.

    This test run all the Queue implementation but we monkeypatch the Redis Queues to avoid external interaction

    """
    import redis

    blpop_out = (b"entry-id", pickle.dumps("test_entry"))

    monkeypatch.setattr(redis.Redis, "blpop", lambda *args, **kwargs: blpop_out)
    monkeypatch.setattr(redis.Redis, "rpush", lambda *args, **kwargs: None)
    monkeypatch.setattr(redis.Redis, "set", lambda *args, **kwargs: None)
    monkeypatch.setattr(redis.Redis, "get", lambda *args, **kwargs: None)

    test_queue = queue_type.get_readiness_queue()
    assert test_queue.name == READINESS_QUEUE_CONSTANT
    assert isinstance(test_queue, BaseQueue)
    test_queue.put("test_entry")
    assert test_queue.get() == "test_entry"


@pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running")
def test_redis_queue():
    queue_id = int(time.time())
    queue1 = QueuingSystem.REDIS.get_readiness_queue(queue_id=str(queue_id))
    queue2 = QueuingSystem.REDIS.get_readiness_queue(queue_id=str(queue_id + 1))
    queue1.put("test_entry1")
    queue2.put("test_entry2")
    assert queue1.get() == "test_entry1"
    assert queue2.get() == "test_entry2"
    with pytest.raises(queue.Empty):
        queue2.get(timeout=1)
    queue1.put("test_entry1")
    assert queue1.length() == 1
    queue1.clear()
    with pytest.raises(queue.Empty):
        queue1.get(timeout=1)


@pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running")
def test_redis_health_check_success():
    redis_queue = QueuingSystem.REDIS.get_readiness_queue()
    assert redis_queue.is_running

    redis_queue = RedisQueue(name="test_queue", default_timeout=1)
    assert redis_queue.is_running


@pytest.mark.skipif(not _is_redis_available(), reason="redis is required for this test.")
@pytest.mark.skipif(check_if_redis_running(), reason="This is testing the failure case when redis is not running")
def test_redis_health_check_failure():
    redis_queue = RedisQueue(name="test_queue", default_timeout=1)
    assert not redis_queue.is_running


@pytest.mark.skipif(not _is_redis_available(), reason="redis isn't installed.")
def test_redis_credential(monkeypatch):
    monkeypatch.setattr(queues, "REDIS_HOST", "test-host")
    monkeypatch.setattr(queues, "REDIS_PORT", "test-port")
    monkeypatch.setattr(queues, "REDIS_PASSWORD", "test-password")
    redis_queue = QueuingSystem.REDIS.get_readiness_queue()
    assert redis_queue.redis.connection_pool.connection_kwargs["host"] == "test-host"
    assert redis_queue.redis.connection_pool.connection_kwargs["port"] == "test-port"
    assert redis_queue.redis.connection_pool.connection_kwargs["password"] == "test-password"


@pytest.mark.skipif(not _is_redis_available(), reason="redis isn't installed.")
@mock.patch("lightning.app.core.queues.redis.Redis")
def test_redis_queue_read_timeout(redis_mock):
    redis_mock.return_value.blpop.return_value = (b"READINESS_QUEUE", pickle.dumps("test_entry"))
    redis_queue = QueuingSystem.REDIS.get_readiness_queue()

    # default timeout
    assert redis_queue.get(timeout=0) == "test_entry"
    assert redis_mock.return_value.blpop.call_args_list[0] == mock.call(["READINESS_QUEUE"], timeout=0.005)

    # custom timeout
    assert redis_queue.get(timeout=2) == "test_entry"
    assert redis_mock.return_value.blpop.call_args_list[1] == mock.call(["READINESS_QUEUE"], timeout=2)

    # blocking timeout
    assert redis_queue.get() == "test_entry"
    assert redis_mock.return_value.blpop.call_args_list[2] == mock.call(["READINESS_QUEUE"], timeout=0)


@pytest.mark.parametrize(
    ("queue_type", "queue_process_mock"),
    [(QueuingSystem.MULTIPROCESS, multiprocessing)],
)
def test_process_queue_read_timeout(queue_type, queue_process_mock, monkeypatch):
    context = mock.MagicMock()
    queue_mocked = mock.MagicMock()
    context.Queue = queue_mocked
    monkeypatch.setattr(queue_process_mock, "get_context", mock.MagicMock(return_value=context))
    my_queue = queue_type.get_readiness_queue()

    # default timeout
    my_queue.get(timeout=0)
    assert queue_mocked.return_value.get.call_args_list[0] == mock.call(timeout=0.001, block=False)

    # custom timeout
    my_queue.get(timeout=2)
    assert queue_mocked.return_value.get.call_args_list[1] == mock.call(timeout=2, block=False)

    # blocking timeout
    my_queue.get()
    assert queue_mocked.return_value.get.call_args_list[2] == mock.call(timeout=None, block=True)


@pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running")
@mock.patch("lightning.app.core.queues.WARNING_QUEUE_SIZE", 2)
def test_redis_queue_warning():
    my_queue = QueuingSystem.REDIS.get_api_delta_queue(queue_id="test_redis_queue_warning")
    my_queue.clear()
    with pytest.warns(UserWarning, match="is larger than the"):
        my_queue.put(None)
        my_queue.put(None)
        my_queue.put(None)


@pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running")
@mock.patch("lightning.app.core.queues.redis.Redis")
def test_redis_raises_error_if_failing(redis_mock):
    import redis

    my_queue = QueuingSystem.REDIS.get_api_delta_queue(queue_id="test_redis_queue_warning")
    redis_mock.return_value.rpush.side_effect = redis.exceptions.ConnectionError("EROOOR")
    redis_mock.return_value.llen.side_effect = redis.exceptions.ConnectionError("EROOOR")

    with pytest.raises(ConnectionError, match="Your app failed because it couldn't connect to Redis."):
        redis_mock.return_value.blpop.side_effect = redis.exceptions.ConnectionError("EROOOR")
        my_queue.get()

    with pytest.raises(ConnectionError, match="Your app failed because it couldn't connect to Redis."):
        redis_mock.return_value.rpush.side_effect = redis.exceptions.ConnectionError("EROOOR")
        redis_mock.return_value.llen.return_value = 1
        my_queue.put(1)

    with pytest.raises(ConnectionError, match="Your app failed because it couldn't connect to Redis."):
        redis_mock.return_value.llen.side_effect = redis.exceptions.ConnectionError("EROOOR")
        my_queue.length()


class TestHTTPQueue:
    def test_http_queue_failure_on_queue_name(self):
        test_queue = HTTPQueue("test", STATE_UPDATE_TIMEOUT)
        with pytest.raises(ValueError, match="App ID couldn't be extracted"):
            test_queue.put("test")

        with pytest.raises(ValueError, match="App ID couldn't be extracted"):
            test_queue.get()

        with pytest.raises(ValueError, match="App ID couldn't be extracted"):
            test_queue.length()

    def test_http_queue_put(self, monkeypatch):
        monkeypatch.setattr(queues, "HTTP_QUEUE_TOKEN", "test-token")
        test_queue = HTTPQueue("test_http_queue", STATE_UPDATE_TIMEOUT)
        test_obj = LightningFlow()

        # mocking requests and responses
        adapter = requests_mock.Adapter()
        test_queue.client.session.mount("http://", adapter)
        adapter.register_uri(
            "GET",
            f"{HTTP_QUEUE_URL}/v1/test/http_queue/length",
            request_headers={"Authorization": "Bearer test-token"},
            status_code=200,
            content=b"1",
        )
        adapter.register_uri(
            "POST",
            f"{HTTP_QUEUE_URL}/v1/test/http_queue?action=push",
            status_code=201,
            additional_matcher=lambda req: pickle.dumps(test_obj) == req._request.body,
            request_headers={"Authorization": "Bearer test-token"},
            content=b"data pushed",
        )

        test_queue.put(test_obj)

    def test_http_queue_get(self, monkeypatch):
        monkeypatch.setattr(queues, "HTTP_QUEUE_TOKEN", "test-token")
        test_queue = HTTPQueue("test_http_queue", STATE_UPDATE_TIMEOUT)
        adapter = requests_mock.Adapter()
        test_queue.client.session.mount("http://", adapter)

        adapter.register_uri(
            "POST",
            f"{HTTP_QUEUE_URL}/v1/test/http_queue?action=pop",
            request_headers={"Authorization": "Bearer test-token"},
            status_code=200,
            content=pickle.dumps("test"),
        )
        assert test_queue.get() == "test"

    def test_http_queue_batch_get(self, monkeypatch):
        monkeypatch.setattr(queues, "HTTP_QUEUE_TOKEN", "test-token")
        test_queue = HTTPQueue("test_http_queue", STATE_UPDATE_TIMEOUT)
        adapter = requests_mock.Adapter()
        test_queue.client.session.mount("http://", adapter)

        adapter.register_uri(
            "POST",
            f"{HTTP_QUEUE_URL}/v1/test/http_queue?action=popCount",
            request_headers={"Authorization": "Bearer test-token"},
            status_code=200,
            json=[
                base64.b64encode(pickle.dumps("test")).decode("utf-8"),
                base64.b64encode(pickle.dumps("test2")).decode("utf-8"),
            ],
        )
        assert test_queue.batch_get() == ["test", "test2"]


def test_unreachable_queue(monkeypatch):
    monkeypatch.setattr(queues, "HTTP_QUEUE_TOKEN", "test-token")

    test_queue = HTTPQueue("test_http_queue", STATE_UPDATE_TIMEOUT)

    resp1 = mock.MagicMock()
    resp1.status_code = 204

    resp2 = mock.MagicMock()
    resp2.status_code = 201

    test_queue.client = mock.MagicMock()
    test_queue.client.post = mock.Mock(side_effect=[resp1, resp1, resp2])

    with pytest.raises(queue.Empty):
        test_queue._get()

    # Test backoff on queue.put
    test_queue.put("foo")
    assert test_queue.client.post.call_count == 3


@mock.patch("lightning.app.core.queues.time.sleep")
def test_rate_limited_queue(mock_sleep):
    sleeps = []
    mock_sleep.side_effect = lambda sleep_time: sleeps.append(sleep_time)

    mock_queue = mock.MagicMock()

    mock_queue.name = "inner_queue"
    mock_queue.default_timeout = 10.0

    rate_limited_queue = RateLimitedQueue(mock_queue, requests_per_second=1)

    assert rate_limited_queue.name == "inner_queue"
    assert rate_limited_queue.default_timeout == 10.0

    timeout = time.perf_counter() + 1
    while time.perf_counter() + sum(sleeps) < timeout:
        rate_limited_queue.get()

    assert mock_queue.get.call_count == 2
