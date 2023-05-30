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

from typing import Optional

from lightning.app.core.constants import REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from lightning.app.utilities.imports import _is_redis_available


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
