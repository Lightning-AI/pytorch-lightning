import pickle
import queue
import time
from unittest import mock

import pytest

from lightning_app.core import queues
from lightning_app.core.queues import QueuingSystem, READINESS_QUEUE_CONSTANT, RedisQueue
from lightning_app.utilities.imports import _is_redis_available
from lightning_app.utilities.redis import check_if_redis_running


@pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running")
@pytest.mark.parametrize("queue_type", list(QueuingSystem.__members__.values()))
def test_queue_api(queue_type, monkeypatch):
    """Test the Queue API.

    This test run all the Queue implementation but we monkeypatch the Redis Queues to avoid external interaction
    """

    blpop_out = (b"entry-id", pickle.dumps("test_entry"))

    monkeypatch.setattr(queues.redis.Redis, "blpop", lambda *args, **kwargs: blpop_out)
    monkeypatch.setattr(queues.redis.Redis, "rpush", lambda *args, **kwargs: None)
    monkeypatch.setattr(queues.redis.Redis, "set", lambda *args, **kwargs: None)
    monkeypatch.setattr(queues.redis.Redis, "get", lambda *args, **kwargs: None)
    queue = queue_type.get_readiness_queue()
    assert queue.name == READINESS_QUEUE_CONSTANT
    assert isinstance(queue, queues.BaseQueue)
    queue.put("test_entry")
    assert queue.get() == "test_entry"


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
def test_redis_ping_success():
    redis_queue = QueuingSystem.REDIS.get_readiness_queue()
    assert redis_queue.ping()

    redis_queue = RedisQueue(name="test_queue", default_timeout=1)
    assert redis_queue.ping()


@pytest.mark.skipif(not _is_redis_available(), reason="redis is required for this test.")
@pytest.mark.skipif(check_if_redis_running(), reason="This is testing the failure case when redis is not running")
def test_redis_ping_failure():
    redis_queue = RedisQueue(name="test_queue", default_timeout=1)
    assert not redis_queue.ping()


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
@mock.patch("lightning_app.core.queues.redis.Redis")
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
    "queue_type, queue_process_mock",
    [(QueuingSystem.SINGLEPROCESS, queues.queue), (QueuingSystem.MULTIPROCESS, queues.multiprocessing)],
)
def test_process_queue_read_timeout(queue_type, queue_process_mock, monkeypatch):

    queue_mocked = mock.MagicMock()
    monkeypatch.setattr(queue_process_mock, "Queue", queue_mocked)
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
@mock.patch("lightning_app.core.queues.REDIS_WARNING_QUEUE_SIZE", 2)
def test_redis_queue_warning():
    my_queue = QueuingSystem.REDIS.get_api_delta_queue(queue_id="test_redis_queue_warning")
    my_queue.clear()
    with pytest.warns(UserWarning, match="is larger than the"):
        my_queue.put(None)
        my_queue.put(None)
        my_queue.put(None)


@pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running")
@mock.patch("lightning_app.core.queues.redis.Redis")
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
