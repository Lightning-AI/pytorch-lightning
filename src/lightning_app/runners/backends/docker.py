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

from time import time
from typing import List, Optional

import lightning_app
from lightning_app.core.queues import QueuingSystem
from lightning_app.runners.backends.backend import Backend


class DockerBackend(Backend):
    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        pass

    def stop_work(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork") -> None:
        pass

    def __init__(self, entrypoint_file: str):
        super().__init__(entrypoint_file=entrypoint_file, queues=QueuingSystem.REDIS, queue_id=str(int(time())))

    def create_work(self, app, work):
        pass

    def update_work_statuses(self, works) -> None:
        pass

    def stop_all_works(self, works: List["lightning_app.LightningWork"]) -> None:
        pass
