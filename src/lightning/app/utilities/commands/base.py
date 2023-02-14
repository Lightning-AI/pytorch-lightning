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

import errno
import inspect
import os
import os.path as osp
import shutil
import sys
import traceback
from dataclasses import asdict
from getpass import getuser
from importlib.util import module_from_spec, spec_from_file_location
from tempfile import gettempdir
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from fastapi import HTTPException
from pydantic import BaseModel

from lightning.app.api.http_methods import Post
from lightning.app.api.request_types import _APIRequest, _CommandRequest, _RequestResponse
from lightning.app.utilities import frontend
from lightning.app.utilities.app_helpers import is_overridden, Logger
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient
from lightning.app.utilities.state import AppState

logger = Logger(__name__)


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


class ClientCommand:
    description: str = ""
    requirements: List[str] = []

    def __init__(self, method: Callable):
        self.method = method
        if not self.description:
            self.description = self.method.__doc__ or ""
        flow = getattr(self.method, "__self__", None)
        self.owner = flow.name if flow else None
        self.models: Optional[Dict[str, BaseModel]] = None
        self.app_url = None
        self._state = None

    def _setup(self, command_name: str, app_url: str) -> None:
        self.command_name = command_name
        self.app_url = app_url

    @property
    def state(self):
        if self._state is None:
            assert self.app_url
            # TODO: Resolve this hack
            os.environ["LIGHTNING_APP_STATE_URL"] = "1"
            self._state = AppState(host=self.app_url)
            self._state._request_state()
            os.environ.pop("LIGHTNING_APP_STATE_URL")
        return self._state

    def run(self, **cli_kwargs) -> None:
        """Overrides with the logic to execute on the client side."""

    def invoke_handler(self, config: Optional[BaseModel] = None) -> Dict[str, Any]:
        command = self.command_name.replace(" ", "_")
        resp = requests.post(self.app_url + f"/command/{command}", data=config.json() if config else None)
        if resp.status_code != 200:
            try:
                detail = str(resp.json())
            except Exception:
                detail = "Internal Server Error"
            print(f"Failed with status code {resp.status_code}. Detail: {detail}")
            sys.exit(0)

        return resp.json()

    def _to_dict(self):
        return {"owner": self.owner, "requirements": self.requirements}

    def __call__(self, **kwargs):
        return self.method(**kwargs)


def _download_command(
    command_name: str,
    cls_path: str,
    cls_name: str,
    app_id: Optional[str] = None,
    debug_mode: bool = False,
    target_file: Optional[str] = None,
) -> ClientCommand:
    # TODO: This is a skateboard implementation and the final version will rely on versioned
    # immutable commands for security concerns
    command_name = command_name.replace(" ", "_")
    tmpdir = None
    if not target_file:
        tmpdir = osp.join(gettempdir(), f"{getuser()}_commands")
        makedirs(tmpdir)
        target_file = osp.join(tmpdir, f"{command_name}.py")

    if not debug_mode:
        if app_id:
            if not os.path.exists(target_file):
                client = LightningClient(retry=False)
                project_id = _get_project(client).project_id
                response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(
                    project_id=project_id, id=app_id
                )
                for artifact in response.artifacts:
                    if f"commands/{command_name}.py" == artifact.filename:
                        resp = requests.get(artifact.url, allow_redirects=True)

                        with open(target_file, "wb") as f:
                            f.write(resp.content)
        else:
            shutil.copy(cls_path, target_file)

    spec = spec_from_file_location(cls_name, target_file)
    mod = module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    command_type = getattr(mod, cls_name)
    if issubclass(command_type, ClientCommand):
        command = command_type(method=None)
    else:
        raise ValueError(f"Expected class {cls_name} for command {command_name} to be a `ClientCommand`.")
    if tmpdir and os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    return command


def _to_annotation(anno: str) -> str:
    anno = anno.split("'")[1]
    if "." in anno:
        return anno.split(".")[-1]
    return anno


def _validate_client_command(command: ClientCommand):
    """Extract method and its metadata from a ClientCommand."""
    params = inspect.signature(command.method).parameters
    command_metadata = {
        "cls_path": inspect.getfile(command.__class__),
        "cls_name": command.__class__.__name__,
        "params": {p.name: _to_annotation(str(p.annotation)) for p in params.values()},
        **command._to_dict(),
    }
    method = command.method
    command.models = {}
    for k, v in command_metadata["params"].items():
        if v == "_empty":
            raise Exception(
                f"Please, annotate your method {method} with pydantic BaseModel. Refer to the documentation."
            )
        config = getattr(sys.modules[command.__module__], v, None)
        if config is None:
            config = getattr(sys.modules[method.__module__], v, None)
            if config:
                raise Exception(
                    f"The provided annotation for the argument {k} should in the file "
                    f"{inspect.getfile(command.__class__)}, not {inspect.getfile(command.method)}."
                )
        if config is None or not issubclass(config, BaseModel):
            raise Exception(
                f"The provided annotation for the argument {k} shouldn't an instance of pydantic BaseModel."
            )


def _upload(name: str, prefix: str, obj: Any) -> Optional[str]:
    from lightning.app.storage.path import _filesystem, _is_s3fs_available, _shared_storage_path

    name = name.replace(" ", "_")
    filepath = f"{prefix}/{name}.py"
    fs = _filesystem()

    if _is_s3fs_available():
        from s3fs import S3FileSystem

        if not isinstance(fs, S3FileSystem):
            return

        source_file = str(inspect.getfile(obj.__class__))
        remote_url = str(_shared_storage_path() / "artifacts" / filepath)
        fs.put(source_file, remote_url)
        return filepath


def _prepare_commands(app) -> List:
    if not is_overridden("configure_commands", app.root):
        return []

    # 1: Upload the command to s3.
    commands = app.root.configure_commands()
    for command_mapping in commands:
        for command_name, command in command_mapping.items():
            if isinstance(command, ClientCommand):
                _upload(command_name, "commands", command)

    # 2: Cache the commands on the app.
    app.commands = commands
    return commands


def _process_api_request(app, request: _APIRequest):
    flow = app.get_component_by_name(request.name)
    method = getattr(flow, request.method_name)
    try:
        response = _RequestResponse(content=method(*request.args, **request.kwargs), status_code=200)
    except HTTPException as e:
        logger.error(repr(e))
        response = _RequestResponse(status_code=e.status_code, content=e.detail)
    except Exception:
        logger.error(traceback.print_exc())
        response = _RequestResponse(status_code=500)
    return {"response": response, "id": request.id}


def _process_command_requests(app, request: _CommandRequest):
    for command in app.commands:
        for command_name, method in command.items():
            command_name = command_name.replace(" ", "_")
            if request.method_name == command_name:
                # 2.1: Evaluate the method associated to a specific command.
                # Validation is done on the CLI side.
                try:
                    response = _RequestResponse(content=method(*request.args, **request.kwargs), status_code=200)
                except HTTPException as e:
                    logger.error(repr(e))
                    response = _RequestResponse(status_code=e.status_code, content=e.detail)
                except Exception:
                    logger.error(traceback.print_exc())
                    response = _RequestResponse(status_code=500)
                return {"response": response, "id": request.id}


def _process_requests(app, requests: List[Union[_APIRequest, _CommandRequest]]) -> None:
    """Convert user commands to API endpoint."""
    responses = []
    for request in requests:
        if isinstance(request, _APIRequest):
            response = _process_api_request(app, request)
        else:
            response = _process_command_requests(app, request)

        if response:
            responses.append(response)

    app.api_response_queue.put(responses)


def _collect_open_api_extras(command, info) -> Dict:
    if not isinstance(command, ClientCommand):
        if command.__doc__ is not None:
            return {"description": command.__doc__}
        return {}

    extras = {
        "cls_path": inspect.getfile(command.__class__),
        "cls_name": command.__class__.__name__,
        "description": command.description,
    }
    if command.requirements:
        extras.update({"requirements": command.requirements})
    if info:
        extras.update({"app_info": asdict(info)})
    return extras


def _commands_to_api(
    commands: List[Dict[str, Union[Callable, ClientCommand]]], info: Optional[frontend.AppInfo] = None
) -> List:
    """Convert user commands to API endpoint."""
    api = []
    for command in commands:
        for k, v in command.items():
            k = k.replace(" ", "_")
            api.append(
                Post(
                    f"/command/{k}",
                    v.method if isinstance(v, ClientCommand) else v,
                    method_name=k,
                    tags=["app_client_command"] if isinstance(v, ClientCommand) else ["app_command"],
                    openapi_extra=_collect_open_api_extras(v, info),
                )
            )
    return api
