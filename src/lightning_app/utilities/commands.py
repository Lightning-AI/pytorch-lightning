import errno
import os
import os.path as osp
import shutil
import sys
from getpass import getuser
from importlib.util import module_from_spec, spec_from_file_location
from tempfile import gettempdir
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel

from lightning.app.utilities.state import headers_for


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


class _Config(BaseModel):
    command: str
    affiliation: str
    params: Dict[str, str]
    is_command: bool
    cls_path: str
    cls_name: str
    owner: str
    requirements: Optional[List[str]]


class ClientCommand:
    def __init__(self, method: Callable, requirements: Optional[List[str]] = None) -> None:
        self.method = method
        flow = getattr(method, "__self__", None)
        self.owner = flow.name if flow else None
        self.requirements = requirements
        self.metadata = None
        self.models = Optional[Dict[str, BaseModel]]
        self.url = None

    def _setup(self, metadata: Dict[str, Any], models: Dict[str, BaseModel], url: str) -> None:
        self.metadata = metadata
        self.models = models
        self.url = url

    def run(self):
        """Overrides with the logic to execute on the client side."""

    def invoke_handler(self, **kwargs: Any) -> Dict[str, Any]:
        assert kwargs.keys() == self.models.keys()
        for k, v in kwargs.items():
            assert isinstance(v, self.models[k])
        json = {
            "command_name": self.metadata["command"],
            "command_arguments": {k: v.json() for k, v in kwargs.items()},
            "affiliation": self.metadata["affiliation"],
        }
        resp = requests.post(self.url + "/api/v1/commands", json=json, headers=headers_for({}))
        assert resp.status_code == 200, resp.json()
        return resp.json()

    def _to_dict(self):
        return {"owner": self.owner, "requirements": self.requirements}

    def __call__(self, **kwargs: Any) -> Any:
        assert self.models
        kwargs = {k: self.models[k].parse_raw(v) for k, v in kwargs.items()}
        return self.method(**kwargs)


def _download_command(command_metadata: Dict[str, Any]) -> Tuple[ClientCommand, Dict[str, BaseModel]]:
    config = _Config(**command_metadata)
    print(config)
    if config.cls_path.startswith("s3://"):
        raise NotImplementedError()
    else:
        tmpdir = osp.join(gettempdir(), f"{getuser()}_commands")
        makedirs(tmpdir)
        cls_name = config.cls_name
        target_file = osp.join(tmpdir, f"{config.command}.py")
        shutil.copy(config.cls_path, target_file)
        spec = spec_from_file_location(config.cls_name, target_file)
        mod = module_from_spec(spec)
        sys.modules[cls_name] = mod
        spec.loader.exec_module(mod)
        command = getattr(mod, cls_name)(method=None, requirements=config.requirements)
        models = {k: getattr(mod, v) for k, v in config.params.items()}
        shutil.rmtree(tmpdir)
        return command, models
