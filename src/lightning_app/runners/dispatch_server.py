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
import glob
import os
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lightning_app.runners import CloudRuntime
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.component import _set_flow_context
from lightning_app.utilities.enum import AppStage

logger = Logger(__name__)

fastapi_service = FastAPI()

fastapi_service.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Run(BaseModel):
    project_id: str
    cloudspace_id: str
    name: str
    entrypoint: str
    cluster_id: Optional[str] = None


@fastapi_service.post("/v1/runs")
def _create_run(run: Run) -> None:
    """Create a run with the given name and entrypoint under the cloudspace with the given ID."""
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


@fastapi_service.get("/v1/files")
def _list_files() -> List[str]:
    """List all files under `/content`."""
    # TODO: Return more information here
    paths = glob.iglob("/content/**/*", recursive=True)
    paths = [Path(path) for path in paths]
    relative_files = [str(path.absolute().relative_to("/content")) for path in paths if path.is_file()]
    return relative_files


def _start_dispatch_server(
    host="0.0.0.0",
    port=8888,
):
    host = host.split("//")[-1] if "//" in host else host
    uvicorn.run(app=fastapi_service, host=host, port=port, log_level="error")
