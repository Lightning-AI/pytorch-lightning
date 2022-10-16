import threading
from copy import deepcopy
from datetime import datetime
from typing import Optional

from croniter import croniter
from deepdiff import DeepDiff, Delta

from lightning_app.utilities.proxies import ComponentDelta


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
                flow = self._app.get_component_by_name(metadata["name"])
                previous_state = deepcopy(flow.state)
                flow._enable_schedule(call_hash)
                component_delta = ComponentDelta(
                    id=flow.name, delta=Delta(DeepDiff(previous_state, flow.state, verbose_level=2))
                )
                self._app.delta_queue.put(component_delta)
                metadata["start_time"] = next_event.isoformat()

    def join(self, timeout: Optional[float] = None) -> None:
        self._exit_event.set()
        super().join(timeout)
