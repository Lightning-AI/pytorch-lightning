import logging
import multiprocessing
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Type, Union

import lightning_app
from lightning_app import LightningApp
from lightning_app.core.constants import APP_SERVER_HOST, APP_SERVER_PORT
from lightning_app.runners.backends import Backend, BackendType
from lightning_app.utilities.enum import AppStage, make_status, WorkStageStatus
from lightning_app.utilities.load_app import load_app_from_file
from lightning_app.utilities.proxies import WorkRunner

logger = logging.getLogger(__name__)


def dispatch(
    entrypoint_file: Path,
    runtime_type: "lightning_app.runners.runtime_type.RuntimeType",
    start_server: bool = True,
    no_cache: bool = False,
    host: str = APP_SERVER_HOST,
    port: int = APP_SERVER_PORT,
    blocking: bool = True,
    on_before_run: Optional[Callable] = None,
    name: str = "",
    env_vars: Dict[str, str] = {},
) -> Optional[Any]:
    """Bootstrap and dispatch the application to the target.

    Arguments:
        entrypoint_file: Filepath to the current script
        runtime_type: The runtime to be used for launching the app.
        start_server: Whether to run the app REST API.
        no_cache: Whether to use the dependency cache for the app.
        host: Server host address
        port: Server port
        blocking: Whether for the wait for the UI to start running.
        on_before_run: Callable to be executed before run.
        name: Name of app execution
        env_vars: Dict of env variables to be set on the app
    """
    from lightning_app.runners.runtime_type import RuntimeType
    from lightning_app.utilities.component import _set_flow_context

    _set_flow_context()

    runtime_type = RuntimeType(runtime_type)
    runtime_cls: Type[Runtime] = runtime_type.get_runtime()
    app = load_app_from_file(str(entrypoint_file))

    if blocking:
        app.stage = AppStage.BLOCKING

    runtime = runtime_cls(
        app=app, entrypoint_file=entrypoint_file, start_server=start_server, host=host, port=port, env_vars=env_vars
    )
    # a cloud dispatcher will return the result while local
    # dispatchers will be running the app in the main process
    return runtime.dispatch(on_before_run=on_before_run, name=name, no_cache=no_cache)


@dataclass
class Runtime:
    app: LightningApp
    entrypoint_file: Optional[Path] = None
    start_server: bool = True
    host: str = APP_SERVER_HOST
    port: int = APP_SERVER_PORT
    processes: Dict[str, multiprocessing.Process] = field(default_factory=dict)
    threads: List[Thread] = field(default_factory=list)
    work_runners: Dict[str, WorkRunner] = field(default_factory=dict)
    done: bool = False
    backend: Optional[Union[str, Backend]] = "multiprocessing"
    env_vars: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.backend, str):
            self.backend = BackendType(self.backend).get_backend(self.entrypoint_file)

        lightning_app.LightningFlow._attach_backend(self.app.root, self.backend)

    def terminate(self) -> None:
        """This method is used to terminate all the objects (threads, processes, etc..) created by the app."""
        logger.info("Your Lightning App is being stopped. This won't take long.")
        self.done = False
        has_messaged = False
        while not self.done:
            try:
                for work in self.app.works:
                    if not hasattr(work, "_has_called_on_exit"):
                        work.on_exit()
                        work._has_called_on_exit = True

                if self.app.backend is not None:
                    self.app.backend.stop_all_works(self.app.works)

                if self.app.api_publish_state_queue:
                    for work in self.app.works:
                        self._add_stopped_status_to_work(work)

                    # Publish the updated state and wait for the frontend to update.
                    self.app.api_publish_state_queue.put(self.app.state)

                for thread in self.threads + self.app.threads:
                    thread.join(timeout=0)

                for frontend in self.app.frontends.values():
                    frontend.stop_server()

                for proc in list(self.processes.values()) + list(self.app.processes.values()):
                    if proc.is_alive():
                        proc.kill()

                self.done = True

            except KeyboardInterrupt:
                if not has_messaged:
                    logger.info("Your Lightning App is being stopped. This won't take long.")
                has_messaged = True

            if self.done:
                logger.info("Your Lightning App has been stopped successfully!")

        # Inform the application failed.
        if self.app.stage == AppStage.FAILED:
            sys.exit(1)

    def dispatch(self, *args, **kwargs):
        raise NotImplementedError

    def _add_stopped_status_to_work(self, work: "lightning_app.LightningWork") -> None:
        if work.status.stage == WorkStageStatus.STOPPED:
            return
        latest_hash = work._calls["latest_call_hash"]
        if latest_hash is None:
            return
        work._calls[latest_hash]["statuses"].append(make_status(WorkStageStatus.STOPPED))
