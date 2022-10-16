from typing import Optional

from lightning_app.core.constants import REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from lightning_app.utilities.imports import _is_redis_available


def check_if_redis_running(
    host: Optional[str] = "", port: Optional[int] = 6379, password: Optional[str] = None
) -> bool:
    if not _is_redis_available():
        return False
    import redis

    try:
        host = host or REDIS_HOST
        port = port or REDIS_PORT
        password = password or REDIS_PASSWORD
        return redis.Redis(host=host, port=port, password=password).ping()
    except redis.exceptions.ConnectionError:
        return False
