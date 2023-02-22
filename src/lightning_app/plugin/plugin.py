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
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lightning_app.core import constants
from lightning_app.plugin.actions import _Action
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.component import _set_flow_context
from lightning_app.utilities.enum import AppStage
from lightning_app.utilities.load_app import _load_plugin_from_file

logger = Logger(__name__)


class LightningPlugin:
    """A ``LightningPlugin`` is a single-file Python class that can be executed within a cloudspace to perform
    actions."""

    def __init__(self) -> None:
        self.project_id = ""
        self.cloudspace_id = ""
        self.cluster_id = ""

    def run(self, *args: str, **kwargs: str) -> Optional[List[_Action]]:
        """Override with the logic to execute on the cloudspace."""
        raise NotImplementedError

    def run_job(self, name: str, app_entrypoint: str, env_vars: Optional[Dict[str, str]] = None) -> str:
        """Run a job in the cloudspace associated with this plugin.

        Args:
            name: The name of the job.
            app_entrypoint: The path of the file containing the app to run.
            env_vars: Additional env vars to set when running the app.

        Returns:
            The relative URL of the created job.
        """
        from lightning_app.runners.cloud import CloudRuntime

        # Dispatch the job
        _set_flow_context()

        entrypoint_file = Path(app_entrypoint)

        app = CloudRuntime.load_app_from_file(str(entrypoint_file.resolve().absolute()))

        app.stage = AppStage.BLOCKING

        runtime = CloudRuntime(
            app=app,
            entrypoint=entrypoint_file,
            start_server=True,
            env_vars=env_vars if env_vars is not None else {},
            secrets={},
            run_app_comment_commands=True,
        )
        # Used to indicate Lightning has been dispatched
        os.environ["LIGHTNING_DISPATCHED"] = "1"

        url = runtime.cloudspace_dispatch(
            project_id=self.project_id,
            cloudspace_id=self.cloudspace_id,
            name=name,
            cluster_id=self.cluster_id,
        )
        # Return a relative URL so it can be used with the NavigateTo action.
        return url.replace(constants.get_lightning_cloud_url(), "")

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
    plugin_arguments: Dict[str, str]


def _run_plugin(run: _Run) -> Dict[str, Any]:
    """Create a run with the given name and entrypoint under the cloudspace with the given ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        download_path = os.path.join(tmpdir, "source.tar.gz")
        source_path = os.path.join(tmpdir, "source")
        os.makedirs(source_path)

        # Download the tarball
        try:
            # Sometimes the URL gets encoded, so we parse it here
            source_code_url = urlparse(run.source_code_url).geturl()

            response = requests.get(source_code_url)

            # TODO: Backoff retry a few times in case the URL is flaky
            response.raise_for_status()

            with open(download_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error downloading plugin source: {str(e)}.",
            )

        # Extract
        try:
            with tarfile.open(download_path, "r:gz") as tf:
                tf.extractall(source_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error extracting plugin source: {str(e)}.",
            )

        # Import the plugin
        try:
            plugin = _load_plugin_from_file(os.path.join(source_path, run.plugin_entrypoint))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error loading plugin: {str(e)}."
            )

        # Ensure that apps are dispatched from the temp directory
        cwd = os.getcwd()
        os.chdir(source_path)

        # Setup and run the plugin
        try:
            plugin._setup(
                project_id=run.project_id,
                cloudspace_id=run.cloudspace_id,
                cluster_id=run.cluster_id,
            )
            actions = plugin.run(**run.plugin_arguments) or []
            return {"actions": [action.to_spec().to_dict() for action in actions]}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error running plugin: {str(e)}."
            )
        finally:
            os.chdir(cwd)


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
