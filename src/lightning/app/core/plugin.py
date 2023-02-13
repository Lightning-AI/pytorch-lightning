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

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.component import _set_flow_context
from lightning.app.utilities.enum import AppStage
from lightning.app.utilities.load_app import _load_plugin_from_file

logger = Logger(__name__)


class LightningPlugin:
    """A ``LightningPlugin`` is a single-file Python class that can be executed within a cloudspace to perform
    actions."""

    def __init__(self) -> None:
        self.project_id = None
        self.cloudspace_id = None
        self.cluster_id = None

    def run(self, name: str, entrypoint: str) -> None:
        """Override with the logic to execute on the client side."""

    def run_job(self, name: str, app_entrypoint: str) -> None:
        """Run a job in the cloudspace associated with this plugin.

        Args:
            name: The name of the job.
            app_entrypoint: The path of the file containing the app to run.
        """
        from lightning.app.runners.cloud import CloudRuntime

        # Dispatch the job
        _set_flow_context()

        entrypoint_file = Path("/content") / app_entrypoint

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
                project_id=self.project_id,
                cloudspace_id=self.cloudspace_id,
                name=name,
                cluster_id=self.cluster_id,
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    def _setup(
        self,
        project_id: str,
        cloudspace_id: str,
        cluster_id: str,
    ) -> None:
        self.project_id = project_id
        self.cloudspace_id = cloudspace_id
        self.cluster_id = cluster_id


class _Run(BaseModel):
    plugin_entrypoint: str
    source_code_url: str
    project_id: str
    cloudspace_id: str
    cluster_id: str
    name: str
    entrypoint: str


def _run_plugin(run: _Run) -> None:
    """Create a run with the given name and entrypoint under the cloudspace with the given ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download the tarball
        os.system(f"curl '{run.source_code_url}' | tar -xz --no-overwrite-dir -m - {tmpdir}")

        # Import the plugin
        plugin = _load_plugin_from_file(os.path.join(tmpdir, run.plugin_entrypoint))

        plugin._setup(
            project_id=run.project_id,
            cloudspace_id=run.cloudspace_id,
            cluster_id=run.cluster_id,
        )
        plugin.run(run.name, run.entrypoint)


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
