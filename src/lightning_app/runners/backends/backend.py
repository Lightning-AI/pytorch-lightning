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

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from lightning_app.core.queues import QueuingSystem
from lightning_app.utilities.proxies import ProxyWorkRun, unwrap

if TYPE_CHECKING:
    import lightning_app


class Backend(ABC):
    """The Backend provides and interface for the framework to communicate with resources in the cloud."""

    def __init__(self, entrypoint_file: str, queues: QueuingSystem, queue_id: str) -> None:
        self.queues: QueuingSystem = queues
        self.queue_id = queue_id
        self.entrypoint_file = entrypoint_file

    @abstractmethod
    def create_work(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork") -> None:
        pass

    @abstractmethod
    def update_work_statuses(self, works: List["lightning_app.LightningWork"]) -> None:
        pass

    @abstractmethod
    def stop_all_works(self, works: List["lightning_app.LightningWork"]) -> None:
        pass

    @abstractmethod
    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def stop_work(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork") -> None:
        pass

    def _dynamic_run_wrapper(
        self,
        *args: Any,
        app: "lightning_app.LightningApp",
        work: "lightning_app.LightningWork",
        work_run: Callable,
        **kwargs: Any,
    ) -> None:
        if not work.name:
            # the name is empty, which means this work was never assigned to a parent flow
            raise AttributeError(
                f"Failed to create process for {work.__class__.__name__}."
                f" Make sure to set this work as an attribute of a `LightningFlow` before calling the run method."
            )

        # 1. Create and register the queues associated the work
        self._register_queues(app, work)

        work.run = work_run

        # 2. Create the work
        self.create_work(app, work)

        # 3. Attach backend
        work._backend = self

        # 4. Create the work proxy to manipulate the work
        work.run = ProxyWorkRun(
            work_run=work_run,
            work_name=work.name,
            work=work,
            caller_queue=app.caller_queues[work.name],
        )

        # 5. Run the work proxy
        return work.run(*args, **kwargs)

    def _wrap_run_method(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork"):
        if work.run.__name__ == "_dynamic_run_wrapper":
            return

        work.run = partial(self._dynamic_run_wrapper, app=app, work=work, work_run=unwrap(work.run))

    def _prepare_queues(self, app: "lightning_app.LightningApp"):
        kw = dict(queue_id=self.queue_id)
        app.delta_queue = self.queues.get_delta_queue(**kw)
        app.readiness_queue = self.queues.get_readiness_queue(**kw)
        app.api_response_queue = self.queues.get_api_response_queue(**kw)
        app.error_queue = self.queues.get_error_queue(**kw)
        app.api_publish_state_queue = self.queues.get_api_state_publish_queue(**kw)
        app.api_delta_queue = app.delta_queue
        app.request_queues = {}
        app.response_queues = {}
        app.copy_request_queues = {}
        app.copy_response_queues = {}
        app.caller_queues = {}
        app.work_queues = {}
        app.flow_to_work_delta_queues = {}

    def _register_queues(self, app, work):
        kw = dict(queue_id=self.queue_id, work_name=work.name)
        app.request_queues.update({work.name: self.queues.get_orchestrator_request_queue(**kw)})
        app.response_queues.update({work.name: self.queues.get_orchestrator_response_queue(**kw)})
        app.copy_request_queues.update({work.name: self.queues.get_orchestrator_copy_request_queue(**kw)})
        app.copy_response_queues.update({work.name: self.queues.get_orchestrator_copy_response_queue(**kw)})
        app.caller_queues.update({work.name: self.queues.get_caller_queue(**kw)})
        app.flow_to_work_delta_queues.update({work.name: self.queues.get_flow_to_work_delta_queue(**kw)})


class WorkManager(ABC):
    """The work manager is an interface for the backend, runtime to control the LightningWork."""

    def __init__(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork"):
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def kill(self) -> None:
        pass

    @abstractmethod
    def restart(self) -> None:
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        pass
