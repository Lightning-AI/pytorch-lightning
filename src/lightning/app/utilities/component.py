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

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, TYPE_CHECKING

from deepdiff.helper import NotPresent
from lightning_utilities.core.apply_func import apply_to_collection

from lightning.app.utilities.app_helpers import is_overridden
from lightning.app.utilities.enum import ComponentContext
from lightning.app.utilities.packaging.cloud_compute import CloudCompute
from lightning.app.utilities.tree import breadth_first

if TYPE_CHECKING:
    from lightning.app import LightningFlow

COMPONENT_CONTEXT: Optional[ComponentContext] = None


def _convert_paths_after_init(root: "LightningFlow"):
    """Converts the path attributes on a component to a dictionary.

    This is necessary because at the time of instantiating the component, its full affiliation is not known and Paths
    that get passed to other componenets during ``__init__`` are otherwise not able to reference their origin or
    consumer.
    """
    from lightning.app import LightningFlow, LightningWork
    from lightning.app.storage import Path

    for component in breadth_first(root, types=(LightningFlow, LightningWork)):
        for attr in list(component.__dict__.keys()):
            value = getattr(component, attr)
            if isinstance(value, Path):
                delattr(component, attr)
                component._paths[attr] = value.to_dict()


def _sanitize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Utility function to sanitize the state of a component.

    Sanitization enables the state to be deep-copied and hashed.
    """
    from lightning.app.storage import Drive, Path
    from lightning.app.storage.payload import _BasePayload

    def sanitize_path(path: Path) -> Path:
        path_copy = Path(path)
        path_copy._sanitize()
        return path_copy

    def sanitize_payload(payload: _BasePayload):
        return type(payload).from_dict(content=payload.to_dict())

    def sanitize_drive(drive: Drive) -> Dict:
        return drive.to_dict()

    def sanitize_cloud_compute(cloud_compute: CloudCompute) -> Dict:
        return cloud_compute.to_dict()

    state = apply_to_collection(state, dtype=Path, function=sanitize_path)
    state = apply_to_collection(state, dtype=_BasePayload, function=sanitize_payload)
    state = apply_to_collection(state, dtype=Drive, function=sanitize_drive)
    state = apply_to_collection(state, dtype=CloudCompute, function=sanitize_cloud_compute)
    return state


def _state_to_json(state: Dict[str, Any]) -> Dict[str, Any]:
    """Utility function to make sure that state dict is json serializable."""
    from lightning.app.storage import Path
    from lightning.app.storage.payload import _BasePayload

    state_paths_cleaned = apply_to_collection(state, dtype=(Path, _BasePayload), function=lambda x: x.to_dict())
    state_diff_cleaned = apply_to_collection(state_paths_cleaned, dtype=type(NotPresent), function=lambda x: None)
    return state_diff_cleaned


def _set_context(name: Optional[str]) -> None:
    global COMPONENT_CONTEXT
    COMPONENT_CONTEXT = os.getenv("COMPONENT_CONTEXT") if name is None else ComponentContext(name)


def _get_context() -> Optional[ComponentContext]:
    global COMPONENT_CONTEXT
    return COMPONENT_CONTEXT


def _set_flow_context() -> None:
    global COMPONENT_CONTEXT
    COMPONENT_CONTEXT = ComponentContext.FLOW


def _set_work_context() -> None:
    global COMPONENT_CONTEXT
    COMPONENT_CONTEXT = ComponentContext.WORK


def _set_frontend_context() -> None:
    global COMPONENT_CONTEXT
    COMPONENT_CONTEXT = ComponentContext.FRONTEND


def _is_flow_context() -> bool:
    global COMPONENT_CONTEXT
    return COMPONENT_CONTEXT == ComponentContext.FLOW


def _is_work_context() -> bool:
    global COMPONENT_CONTEXT
    return COMPONENT_CONTEXT == ComponentContext.WORK


def _is_frontend_context() -> bool:
    global COMPONENT_CONTEXT
    return COMPONENT_CONTEXT == ComponentContext.FRONTEND


@contextmanager
def _context(ctx: str) -> Generator[None, None, None]:
    """Set the global component context for the block below this context manager.

    The context is used to determine whether the current process is running for a LightningFlow or for a LightningWork.
    See also :func:`_get_context`, :func:`_set_context`. For internal use only.
    """
    prev = _get_context()
    _set_context(ctx)
    yield
    _set_context(prev)


def _validate_root_flow(flow: "LightningFlow") -> None:
    from lightning.app.core.flow import LightningFlow

    if not is_overridden("run", instance=flow, parent=LightningFlow):
        raise TypeError(
            "The root flow passed to `LightningApp` does not override the `run()` method. This is required. Please"
            f" implement `run()` in your `{flow.__class__.__name__}` class."
        )
