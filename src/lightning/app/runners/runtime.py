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
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union

from lightning.app import LightningApp, LightningFlow
from lightning.app.core.constants import APP_SERVER_HOST, APP_SERVER_PORT
from lightning.app.runners.backends import Backend, BackendType
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.enum import AppStage, CacheCallsKeys, make_status, WorkStageStatus
from lightning.app.utilities.load_app import load_app_from_file
from lightning.app.utilities.proxies import WorkRunner

logger = Logger(__name__)

if TYPE_CHECKING:
    import lightning.app


def dispatch(
    entrypoint_file: Path,
    runtime_type: "lightning.app.runners.runtime_type.RuntimeType",
    start_server: bool = True,
    no_cache: bool = False,
    host: str = APP_SERVER_HOST,
    port: int = APP_SERVER_PORT,
    blocking: bool = True,
    open_ui: bool = True,
    name: str = "",
    env_vars: Dict[str, str] = None,
    secrets: Dict[str, str] = None,
    cluster_id: str = None,
    run_app_comment_commands: bool = False,
    enable_basic_auth: str = "",
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
        open_ui: Whether to open the UI in the browser.
        name: Name of app execution
        env_vars: Dict of env variables to be set on the app
        secrets: Dict of secrets to be passed as environment variables to the app
        cluster_id: the Lightning AI cluster to run the app on. Defaults to managed Lightning AI cloud
        run_app_comment_commands: whether to parse commands from the entrypoint file and execute them before app startup
        enable_basic_auth: whether to enable basic authentication for the app
                           (use credentials in the format username:password as an argument)
    """
    from lightning.app.runners.runtime_type import RuntimeType
    from lightning.app.utilities.component import _set_flow_context

    _set_flow_context()

    runtime_type = RuntimeType(runtime_type)
    runtime_cls: Type[Runtime] = runtime_type.get_runtime()
    app = runtime_cls.load_app_from_file(str(entrypoint_file))

    env_vars = {} if env_vars is None else env_vars
    secrets = {} if secrets is None else secrets

    if blocking:
        app.stage = AppStage.BLOCKING

    runtime = runtime_cls(
        app=app,
        entrypoint=entrypoint_file,
        start_server=start_server,
        host=host,
        port=port,
        env_vars=env_vars,
        secrets=secrets,
        run_app_comment_commands=run_app_comment_commands,
        enable_basic_auth=enable_basic_auth,
    )
    # Used to indicate Lightning has been dispatched
    os.environ["LIGHTNING_DISPATCHED"] = "1"
    # a cloud dispatcher will return the result while local
    # dispatchers will be running the app in the main process
    return runtime.dispatch(open_ui=open_ui, name=name, no_cache=no_cache, cluster_id=cluster_id)


@dataclass
class Runtime:
    app: Optional[LightningApp] = None
    entrypoint: Optional[Path] = None
    start_server: bool = True
    host: str = APP_SERVER_HOST
    port: int = APP_SERVER_PORT
    processes: Dict[str, multiprocessing.Process] = field(default_factory=dict)
    threads: List[Thread] = field(default_factory=list)
    work_runners: Dict[str, WorkRunner] = field(default_factory=dict)
    done: bool = False
    backend: Optional[Union[str, Backend]] = "multiprocessing"
    env_vars: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    run_app_comment_commands: bool = False
    enable_basic_auth: str = ""

    def __post_init__(self):
        if isinstance(self.backend, str):
            self.backend = BackendType(self.backend).get_backend(self.entrypoint)

        if self.app is not None:
            LightningFlow._attach_backend(self.app.root, self.backend)

    def terminate(self) -> None:
        """This method is used to terminate all the objects (threads, processes, etc..) created by the app."""
        logger.info("Your Lightning App is being stopped. This won't take long.")
        self.done = False
        has_messaged = False
        while not self.done:
            try:
                if self.app.backend is not None:
                    self.app.backend.stop_all_works(self.app.works)

                if self.app.api_publish_state_queue:
                    for work in self.app.works:
                        self._add_stopped_status_to_work(work)

                    # Publish the updated state and wait for the frontend to update.
                    self.app.api_publish_state_queue.put((self.app.state, self.app.status))

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

    def _add_stopped_status_to_work(self, work: "lightning.app.LightningWork") -> None:
        if work.status.stage == WorkStageStatus.STOPPED:
            return

        latest_call_hash = work._calls[CacheCallsKeys.LATEST_CALL_HASH]
        if latest_call_hash in work._calls:
            work._calls[latest_call_hash]["statuses"].append(make_status(WorkStageStatus.STOPPED))

    @classmethod
    def load_app_from_file(cls, filepath: str) -> "LightningApp":
        return load_app_from_file(filepath)
