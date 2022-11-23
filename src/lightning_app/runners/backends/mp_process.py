import multiprocessing
from typing import List, Optional

import lightning_app
from lightning_app.core.queues import QueuingSystem
from lightning_app.runners.backends.backend import Backend, WorkManager
from lightning_app.utilities.enum import WorkStageStatus
from lightning_app.utilities.network import _check_service_url_is_ready
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
        self._process = multiprocessing.Process(target=self._work_runner)
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
