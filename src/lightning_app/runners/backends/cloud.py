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

from typing import List, Optional, TYPE_CHECKING

from lightning_app.core.queues import QueuingSystem
from lightning_app.runners.backends import Backend
from lightning_app.utilities.network import LightningClient

if TYPE_CHECKING:
    import lightning_app


class CloudBackend(Backend):
    def __init__(self, entrypoint_file, queue_id: Optional[str] = None, status_update_interval: int = None):
        super().__init__(entrypoint_file, queues=QueuingSystem.MULTIPROCESS, queue_id=queue_id)
        self.client = LightningClient()

    def create_work(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork") -> None:
        raise NotImplementedError

    def update_work_statuses(self, works: List["lightning_app.LightningWork"]) -> None:
        raise NotImplementedError

    def stop_all_works(self, works: List["lightning_app.LightningWork"]) -> None:
        raise NotImplementedError

    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        raise NotImplementedError

    def stop_work(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork") -> None:
        raise NotImplementedError
