# Copyright The Lightning AI team.
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

import os
import pickle
from typing import Any

from lightning.app.utilities.network import HTTPClient


class ImmutableDistributedMap:
    """The ImmutableDistributedMap enables to create a distributed key value pair in the cloud.

    The first process to perform the set operation defines its value.

    """

    def __init__(self):
        lightning_app_state_url = os.getenv("LIGHTNING_APP_STATE_URL")
        if lightning_app_state_url is None:
            raise RuntimeError("The `LIGHTNING_APP_STATE_URL` should be set.")

        self.client: HTTPClient = HTTPClient(lightning_app_state_url)

    def set(self, key: str, value: Any) -> Any:
        resp = self.client.post("/broadcast", json={"key": key, "value": pickle.dumps(value, 0).decode()})
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to broadcast the following {key=} {value=}.")
        return pickle.loads(bytes(resp.json()["value"], "utf-8"))


def broadcast_object(key: str, obj: Any) -> Any:
    """This function enables to broadcast object across machines."""
    if os.getenv("LIGHTNING_APP_STATE_URL") is not None:
        return ImmutableDistributedMap().set(key, obj)
    return obj
