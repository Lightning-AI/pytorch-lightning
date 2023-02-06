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

import multiprocessing
from typing import List, Optional

import lightning_app
from lightning_app.core import constants
from lightning_app.core.queues import QueuingSystem
from lightning_app.runners.backends.backend import Backend, WorkManager
from lightning_app.utilities.enum import WorkStageStatus
from lightning_app.utilities.network import _check_service_url_is_ready, find_free_network_port
from lightning_app.utilities.port import disable_port, enable_port
from lightning_app.utilities.proxies import ProxyWorkRun, WorkRunner


class MultiProcessWorkManager(WorkManager):
    def __init__(self, app, work):
        self.app = app
        self.work = work
        self._process = None

    def start(self):
        self._work_runner = WorkRunner(
            work=self.work,
            work_name=self.work.name,
            caller_queue=self.app.caller_queues[self.work.name],
            delta_queue=self.app.delta_queue,
            readiness_queue=self.app.readiness_queue,
            error_queue=self.app.error_queue,
            request_queue=self.app.request_queues[self.work.name],
            response_queue=self.app.response_queues[self.work.name],
            copy_request_queue=self.app.copy_request_queues[self.work.name],
            copy_response_queue=self.app.copy_response_queues[self.work.name],
            flow_to_work_delta_queue=self.app.flow_to_work_delta_queues[self.work.name],
            run_executor_cls=self.work._run_executor_cls,
        )

        start_method = self.work._start_method
        context = multiprocessing.get_context(start_method)
        self._process = context.Process(target=self._work_runner)
        self._process.start()

    def kill(self):
        self._process.terminate()

    def restart(self):
        assert not self.is_alive()
        work = self._work_runner.work
        # un-wrap ProxyRun.
        is_proxy = isinstance(work.run, ProxyWorkRun)
        if is_proxy:
            work_run = work.run
            work.run = work_run.work_run
        work._restarting = True
        self.start()
        if is_proxy:
            work.run = work_run

    def is_alive(self) -> bool:
        return self._process.is_alive()


class MultiProcessingBackend(Backend):
    def __init__(self, entrypoint_file: str):
        super().__init__(entrypoint_file=entrypoint_file, queues=QueuingSystem.MULTIPROCESS, queue_id="0")

    def create_work(self, app, work) -> None:
        if constants.LIGHTNING_CLOUDSPACE_HOST is not None:
            # Override the port if set by the user
            work._port = find_free_network_port()
            work._host = "0.0.0.0"
            work._future_url = f"https://{work.port}-{constants.LIGHTNING_CLOUDSPACE_HOST}"

        app.processes[work.name] = MultiProcessWorkManager(app, work)
        app.processes[work.name].start()
        self.resolve_url(app)
        app._update_layout()

    def update_work_statuses(self, works) -> None:
        pass

    def stop_all_works(self, works: List["lightning_app.LightningWork"]) -> None:
        pass

    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        for work in app.works:
            if (
                work.status.stage in (WorkStageStatus.RUNNING, WorkStageStatus.SUCCEEDED)
                and work._url == ""
                and work._port
            ):
                url = work._future_url if work._future_url else f"http://{work._host}:{work._port}"
                if _check_service_url_is_ready(url, metadata=f"Checking {work.name}"):
                    work._url = url

    def stop_work(self, app, work: "lightning_app.LightningWork") -> None:
        work_manager: MultiProcessWorkManager = app.processes[work.name]
        work_manager.kill()

    def delete_work(self, app, work: "lightning_app.LightningWork") -> None:
        self.stop_work(app, work)


class CloudMultiProcessingBackend(MultiProcessingBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Note: Track the open ports to close them on termination.
        self.ports = []

    def create_work(self, app, work) -> None:
        work._host = "0.0.0.0"
        nc = enable_port()
        self.ports.append(nc.port)
        work._port = nc.port
        work._future_url = f"https://{nc.host}"
        return super().create_work(app, work)

    def stop_work(self, app, work: "lightning_app.LightningWork") -> None:
        disable_port(work._port)
        self.ports = [port for port in self.ports if port != work._port]
        return super().stop_work(app, work)

    def delete_work(self, app, work: "lightning_app.LightningWork") -> None:
        self.stop_work(app, work)
