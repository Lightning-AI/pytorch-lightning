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

import threading
from datetime import datetime
from typing import Optional

from croniter import croniter
from deepdiff import Delta

from lightning.app.utilities.proxies import ComponentDelta


class SchedulerThread(threading.Thread):
    # TODO (tchaton) Abstract this logic to a generic scheduling service.

    def __init__(self, app) -> None:
        super().__init__(daemon=True)
        self._exit_event = threading.Event()
        self._sleep_time = 1.0
        self._app = app

    def run(self) -> None:
        try:
            while not self._exit_event.is_set():
                self._exit_event.wait(self._sleep_time)
                self.run_once()
        except Exception as e:
            raise e

    def run_once(self):
        for call_hash in list(self._app._schedules.keys()):
            metadata = self._app._schedules[call_hash]
            start_time = datetime.fromisoformat(metadata["start_time"])
            current_date = datetime.now()
            next_event = croniter(metadata["cron_pattern"], start_time).get_next(datetime)
            # When the event is reached, send a delta to activate scheduling.
            if current_date > next_event:
                component_delta = ComponentDelta(
                    id=metadata["name"],
                    delta=Delta(
                        {
                            "values_changed": {
                                f"root['calls']['scheduling']['{call_hash}']['running']": {"new_value": True}
                            }
                        }
                    ),
                )
                self._app.delta_queue.put(component_delta)
                metadata["start_time"] = next_event.isoformat()

    def join(self, timeout: Optional[float] = None) -> None:
        self._exit_event.set()
        super().join(timeout)
