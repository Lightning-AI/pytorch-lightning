# Copyright The Lightning team.
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
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.component import _set_flow_context
from lightning_app.utilities.enum import AppStage
from lightning_app.utilities.network import LightningClient

logger = Logger(__name__)


class Plugin:
    """A ``Plugin`` is a single-file Python class that can be executed within a cloudspace to perform actions."""

    def __init__(self) -> None:
        self.app_url = None

    def run(self, name: str, entrypoint: str) -> None:
        """Override with the logic to execute on the client side."""

    def run_app_command(self, command_name: str, config: Optional[BaseModel] = None) -> Dict[str, Any]:
        """Run a command on the app associated with this plugin.

        Args:
            command_name: The name of the command to run.
            config: The command config or ``None`` if the command doesn't require configuration.
        """
        if self.app_url is None:
            raise RuntimeError("The plugin must be set up before `run_app_command` can be called.")

        command = command_name.replace(" ", "_")
        resp = requests.post(self.app_url + f"/command/{command}", data=config.json() if config else None)
        if resp.status_code != 200:
            try:
                detail = str(resp.json())
            except Exception:
                detail = "Internal Server Error"
            raise RuntimeError(f"Failed with status code {resp.status_code}. Detail: {detail}")

        return resp.json()

    def _setup(self, app_id: str) -> None:
        client = LightningClient()
        project_id = _get_project(client).project_id
        response = client.lightningapp_instance_service_list_lightningapp_instances(
            project_id=project_id, app_id=app_id
        )
        if len(response.lightningapps) > 1:
            raise RuntimeError(f"Found multiple apps with ID: {app_id}")
        if len(response.lightningapps) == 0:
            raise RuntimeError(f"Found no apps with ID: {app_id}")
        self.app_url = response.lightningapps[0].status.url


class _Run(BaseModel):
    plugin_name: str
    project_id: str
    cloudspace_id: str
    name: str
    entrypoint: str
    cluster_id: Optional[str] = None
    app_id: Optional[str] = None


def _run_plugin(run: _Run) -> None:
    """Create a run with the given name and entrypoint under the cloudspace with the given ID."""
    if run.app_id is None and run.plugin_name == "app":
        from lightning_app.runners.cloud import CloudRuntime

        # TODO: App dispatch should be a plugin
        # Dispatch the run
        _set_flow_context()

        entrypoint_file = Path("/content") / run.entrypoint

        app = CloudRuntime.load_app_from_file(str(entrypoint_file.resolve().absolute()))

        app.stage = AppStage.BLOCKING

        runtime = CloudRuntime(
            app=app,
            entrypoint=entrypoint_file,
            start_server=True,
            env_vars={},
            secrets={},
            run_app_comment_commands=True,
        )
        # Used to indicate Lightning has been dispatched
        os.environ["LIGHTNING_DISPATCHED"] = "1"

        try:
            runtime.cloudspace_dispatch(
                project_id=run.project_id,
                cloudspace_id=run.cloudspace_id,
                name=run.name,
                cluster_id=run.cluster_id,
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    elif run.app_id is not None:
        from lightning_app.utilities.cli_helpers import _LightningAppOpenAPIRetriever
        from lightning_app.utilities.commands.base import _download_command

        retriever = _LightningAppOpenAPIRetriever(run.app_id)

        metadata = retriever.api_commands[run.plugin_name]  # type: ignore

        with tempfile.TemporaryDirectory() as tmpdir:

            target_file = os.path.join(tmpdir, f"{run.plugin_name}.py")
            plugin = _download_command(
                run.plugin_name,
                metadata["cls_path"],
                metadata["cls_name"],
                run.app_id,
                target_file=target_file,
            )

            if isinstance(plugin, Plugin):
                plugin._setup(app_id=run.app_id)
                plugin.run(run.name, run.entrypoint)
            else:
                # This should never be possible but we check just in case
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"The plugin {run.plugin_name} is an incorrect type.",
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="App ID must be specified unless `plugin_name='app'`."
        )


def _start_plugin_server(host: str, port: int) -> None:
    """Start the plugin server which can be used to dispatch apps or run plugins."""
    fastapi_service = FastAPI()

    fastapi_service.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_service.post("/v1/runs")(_run_plugin)

    uvicorn.run(app=fastapi_service, host=host, port=port, log_level="error")
