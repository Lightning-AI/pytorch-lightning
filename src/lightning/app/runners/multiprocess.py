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
from dataclasses import dataclass
from typing import Any, Union

import click

from lightning.app.api.http_methods import _add_tags_to_api, _validate_api
from lightning.app.core import constants
from lightning.app.core.api import start_server
from lightning.app.runners.backends import Backend
from lightning.app.runners.runtime import Runtime
from lightning.app.storage.orchestrator import StorageOrchestrator
from lightning.app.utilities.app_helpers import _is_headless, is_overridden
from lightning.app.utilities.commands.base import _commands_to_api, _prepare_commands
from lightning.app.utilities.component import _set_flow_context, _set_frontend_context
from lightning.app.utilities.load_app import extract_metadata_from_app
from lightning.app.utilities.network import find_free_network_port
from lightning.app.utilities.port import disable_port


@dataclass
class MultiProcessRuntime(Runtime):
    """Runtime to launch the LightningApp into multiple processes.

    The MultiProcessRuntime will generate 1 process for each :class:`~lightning.app.core.work.LightningWork` and attach
    queues to enable communication between the different processes.
    """

    backend: Union[str, Backend] = "multiprocessing"
    _has_triggered_termination: bool = False

    def dispatch(self, *args: Any, open_ui: bool = True, **kwargs: Any):
        """Method to dispatch and run the LightningApp."""
        try:
            _set_flow_context()

            # Note: In case the runtime is used in the cloud.
            in_cloudspace = constants.LIGHTNING_CLOUDSPACE_HOST is not None
            self.host = "0.0.0.0" if constants.APP_SERVER_IN_CLOUD or in_cloudspace else self.host

            self.app.backend = self.backend
            self.backend._prepare_queues(self.app)
            self.backend.resolve_url(self.app, "http://127.0.0.1")
            self.app._update_index_file()

            # set env variables
            os.environ.update(self.env_vars)

            # refresh the layout with the populated urls.
            self.app._update_layout()

            _set_frontend_context()
            for frontend in self.app.frontends.values():
                host = "localhost"
                port = find_free_network_port()
                frontend.start_server(host="localhost", port=port)
                frontend.flow._layout["target"] = f"http://{host}:{port}/{frontend.flow.name}"

            _set_flow_context()

            storage_orchestrator = StorageOrchestrator(
                self.app,
                self.app.request_queues,
                self.app.response_queues,
                self.app.copy_request_queues,
                self.app.copy_response_queues,
            )
            self.threads.append(storage_orchestrator)
            storage_orchestrator.setDaemon(True)
            storage_orchestrator.start()

            if self.start_server:
                self.app.should_publish_changes_to_api = True
                has_started_queue = self.backend.queues.get_has_server_started_queue()

                apis = []
                if is_overridden("configure_api", self.app.root):
                    apis = self.app.root.configure_api()
                    _validate_api(apis)
                    _add_tags_to_api(apis, ["app_api"])

                if is_overridden("configure_commands", self.app.root):
                    commands = _prepare_commands(self.app)
                    apis += _commands_to_api(commands, info=self.app.info)

                kwargs = dict(
                    apis=apis,
                    host=self.host,
                    port=self.port,
                    api_response_queue=self.app.api_response_queue,
                    api_publish_state_queue=self.app.api_publish_state_queue,
                    api_delta_queue=self.app.api_delta_queue,
                    has_started_queue=has_started_queue,
                    spec=extract_metadata_from_app(self.app),
                    root_path=self.app.root_path,
                )
                server_proc = multiprocessing.Process(target=start_server, kwargs=kwargs)
                self.processes["server"] = server_proc
                server_proc.start()
                # requires to wait for the UI to be clicked on.

                # wait for server to be ready
                has_started_queue.get()

            if all(
                [
                    open_ui,
                    "PYTEST_CURRENT_TEST" not in os.environ,
                    not _is_headless(self.app),
                    constants.LIGHTNING_CLOUDSPACE_HOST is None,
                ]
            ):
                click.launch(self._get_app_url())

            # Connect the runtime to the application.
            self.app.connect(self)

            # Once the bootstrapping is done, running the rank 0
            # app with all the components inactive
            self.app._run()
        except KeyboardInterrupt:
            self.terminate()
            self._has_triggered_termination = True
            raise
        finally:
            if not self._has_triggered_termination:
                self.terminate()

    def terminate(self):
        if constants.APP_SERVER_IN_CLOUD:
            # Close all the ports open for the App within the App.
            ports = [self.port] + getattr(self.backend, "ports", [])
            for port in ports:
                disable_port(port)
        super().terminate()

    @staticmethod
    def _get_app_url() -> str:
        return os.getenv("APP_SERVER_HOST", "http://127.0.0.1:7501/view")
