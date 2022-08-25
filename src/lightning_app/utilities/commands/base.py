import errno
import inspect
import os
import os.path as osp
import shutil
import sys
from getpass import getuser
from importlib.util import module_from_spec, spec_from_file_location
from tempfile import gettempdir
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from pydantic import BaseModel

from lightning_app.api.http_methods import Post
from lightning_app.api.request_types import APIRequest, CommandRequest
from lightning_app.utilities.app_helpers import is_overridden
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient
from lightning_app.utilities.state import AppState


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


class ClientCommand:
    def __init__(self, method: Callable, requirements: Optional[List[str]] = None) -> None:
        self.method = method
        flow = getattr(method, "__self__", None)
        self.owner = flow.name if flow else None
        self.requirements = requirements
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

    def invoke_handler(self, config: BaseModel) -> Dict[str, Any]:
        resp = requests.post(self.app_url + f"/command/{self.command_name}", data=config.json())
        assert resp.status_code == 200, resp.json()
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
) -> ClientCommand:
    # TODO: This is a skateboard implementation and the final version will rely on versioned
    # immutable commands for security concerns
    tmpdir = osp.join(gettempdir(), f"{getuser()}_commands")
    makedirs(tmpdir)
    target_file = osp.join(tmpdir, f"{command_name}.py")
    if app_id:
        client = LightningClient()
        project_id = _get_project(client).project_id
        response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(project_id, app_id)
        for artifact in response.artifacts:
            if f"commands/{command_name}.py" == artifact.filename:
                r = requests.get(artifact.url, allow_redirects=True)
                with open(target_file, "wb") as f:
                    f.write(r.content)
    else:
        if not debug_mode:
            shutil.copy(cls_path, target_file)

    spec = spec_from_file_location(cls_name, cls_path if debug_mode else target_file)
    mod = module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    command = getattr(mod, cls_name)(method=None, requirements=[])
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


def _upload_command(command_name: str, command: ClientCommand) -> Optional[str]:
    from lightning_app.storage.path import _is_s3fs_available, filesystem, shared_storage_path

    filepath = f"commands/{command_name}.py"
    remote_url = str(shared_storage_path() / "artifacts" / filepath)
    fs = filesystem()

    if _is_s3fs_available():
        from s3fs import S3FileSystem

        if not isinstance(fs, S3FileSystem):
            return
        source_file = str(inspect.getfile(command.__class__))
        remote_url = str(shared_storage_path() / "artifacts" / filepath)
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
                _upload_command(command_name, command)

    # 2: Cache the commands on the app.
    app.commands = commands
    return commands


def _process_api_request(app, request: APIRequest) -> None:
    flow = app.get_component_by_name(request.name)
    method = getattr(flow, request.method_name)
    response = method(*request.args, **request.kwargs)
    app.api_response_queue.put({"response": response, "id": request.id})


def _process_command_requests(app, request: CommandRequest) -> None:
    for command in app.commands:
        for command_name, method in command.items():
            if request.method_name == command_name:
                # 2.1: Evaluate the method associated to a specific command.
                # Validation is done on the CLI side.
                response = method(*request.args, **request.kwargs)
                app.api_response_queue.put({"response": response, "id": request.id})


def _process_requests(app, request: Union[APIRequest, CommandRequest]) -> None:
    """Convert user commands to API endpoint."""
    if isinstance(request, APIRequest):
        _process_api_request(app, request)
    else:
        _process_command_requests(app, request)


def _collect_open_api_extras(command) -> Dict:
    if not isinstance(command, ClientCommand):
        return {}
    return {
        "cls_path": inspect.getfile(command.__class__),
        "cls_name": command.__class__.__name__,
    }


def _commands_to_api(commands: List[Dict[str, Union[Callable, ClientCommand]]]) -> List:
    """Convert user commands to API endpoint."""
    api = []
    for command in commands:
        for k, v in command.items():
            api.append(
                Post(
                    f"/command/{k}",
                    v.method if isinstance(v, ClientCommand) else v,
                    method_name=k,
                    tags=["app_client_command"] if isinstance(v, ClientCommand) else ["app_command"],
                    openapi_extra=_collect_open_api_extras(v),
                )
            )
    return api
