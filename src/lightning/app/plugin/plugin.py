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
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from lightning_cloud.openapi import Externalv1LightningappInstance
from pydantic import BaseModel

from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.component import _set_flow_context
from lightning.app.utilities.enum import AppStage
from lightning.app.utilities.load_app import _load_plugin_from_file

logger = Logger(__name__)

_PLUGIN_MAX_CLIENT_TRIES: int = 3
_PLUGIN_INTERNAL_DIR_PATH: str = f"{os.environ.get('HOME', '')}/internal"


class LightningPlugin:
    """A ``LightningPlugin`` is a single-file Python class that can be executed within a cloudspace to perform
    actions."""

    def __init__(self) -> None:
        self.project_id = ""
        self.cloudspace_id = ""
        self.cluster_id = ""
        self.source_app = ""
        self.keep_machines_after_stop = False

    def run(self, *args: str, **kwargs: str) -> Externalv1LightningappInstance:
        """Override with the logic to execute on the cloudspace."""
        raise NotImplementedError

    def run_job(self, name: str, app_entrypoint: str, env_vars: Dict[str, str] = {}) -> Externalv1LightningappInstance:
        """Run a job in the cloudspace associated with this plugin.

        Args:
            name: The name of the job.
            app_entrypoint: The path of the file containing the app to run.
            env_vars: Additional env vars to set when running the app.

        Returns:
            The spec of the created LightningappInstance.

        """
        from lightning.app.runners.backends.cloud import CloudBackend
        from lightning.app.runners.cloud import CloudRuntime

        logger.info(f"Processing job run request. name: {name}, app_entrypoint: {app_entrypoint}, env_vars: {env_vars}")

        # Dispatch the job
        _set_flow_context()

        entrypoint_file = Path(app_entrypoint)

        app = CloudRuntime.load_app_from_file(str(entrypoint_file.resolve().absolute()), env_vars=env_vars)

        app.stage = AppStage.BLOCKING

        runtime = CloudRuntime(
            app=app,
            entrypoint=entrypoint_file,
            start_server=True,
            env_vars=env_vars,
            secrets={},
            run_app_comment_commands=True,
            backend=CloudBackend(entrypoint_file, client_max_tries=_PLUGIN_MAX_CLIENT_TRIES),
        )
        # Used to indicate Lightning has been dispatched
        os.environ["LIGHTNING_DISPATCHED"] = "1"

        return runtime.cloudspace_dispatch(
            project_id=self.project_id,
            cloudspace_id=self.cloudspace_id,
            name=name,
            cluster_id=self.cluster_id,
            source_app=self.source_app,
            keep_machines_after_stop=self.keep_machines_after_stop,
        )

    def _setup(
        self,
        project_id: str,
        cloudspace_id: str,
        cluster_id: str,
        source_app: str,
        keep_machines_after_stop: bool,
    ) -> None:
        self.source_app = source_app
        self.project_id = project_id
        self.cloudspace_id = cloudspace_id
        self.cluster_id = cluster_id
        self.keep_machines_after_stop = keep_machines_after_stop


class _Run(BaseModel):
    plugin_entrypoint: str
    source_code_url: str
    project_id: str
    cloudspace_id: str
    cluster_id: str
    plugin_arguments: Dict[str, str]
    source_app: str
    keep_machines_after_stop: bool


def _run_plugin(run: _Run) -> Dict[str, Any]:
    from lightning.app.runners.cloud import _to_clean_dict

    """Create a run with the given name and entrypoint under the cloudspace with the given ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        download_path = os.path.join(tmpdir, "source.tar.gz")
        source_path = os.path.join(tmpdir, "source")
        os.makedirs(source_path)

        # Download the tarball
        try:
            logger.info(f"Downloading plugin source: {run.source_code_url}")

            # Sometimes the URL gets encoded, so we parse it here
            source_code_url = urlparse(run.source_code_url).geturl()

            response = requests.get(source_code_url)

            # TODO: Backoff retry a few times in case the URL is flaky
            response.raise_for_status()

            with open(download_path, "wb") as f:
                f.write(response.content)
        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error downloading plugin source: {str(ex)}.",
            )

        # Extract
        try:
            logger.info("Extracting plugin source.")

            with tarfile.open(download_path, "r:gz") as tf:
                tf.extractall(source_path)  # noqa: S202
        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error extracting plugin source: {str(ex)}.",
            )

        # Import the plugin
        try:
            logger.info(f"Importing plugin: {run.plugin_entrypoint}")

            plugin = _load_plugin_from_file(os.path.join(source_path, run.plugin_entrypoint))
        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error loading plugin: {str(ex)}."
            )

        # Allow devs to add files to the app source
        if os.path.isdir(_PLUGIN_INTERNAL_DIR_PATH):
            shutil.copytree(_PLUGIN_INTERNAL_DIR_PATH, source_path, dirs_exist_ok=True)

        # Ensure that apps are dispatched from the temp directory
        cwd = os.getcwd()
        os.chdir(source_path)

        # Setup and run the plugin
        try:
            logger.info(
                "Running plugin. "
                f"project_id: {run.project_id}, cloudspace_id: {run.cloudspace_id}, cluster_id: {run.cluster_id}."
            )

            plugin._setup(
                project_id=run.project_id,
                cloudspace_id=run.cloudspace_id,
                cluster_id=run.cluster_id,
                source_app=run.source_app,
                keep_machines_after_stop=run.keep_machines_after_stop,
            )
            app_instance = plugin.run(**run.plugin_arguments)
            return _to_clean_dict(app_instance, True)
        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error running plugin: {str(ex)}."
            )
        finally:
            os.chdir(cwd)


async def _healthz() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def _start_plugin_server(port: int) -> None:
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
    fastapi_service.get("/healthz", status_code=200)(_healthz)

    uvicorn.run(
        app=fastapi_service,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
