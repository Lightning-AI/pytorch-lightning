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

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

from lightning_app.core.constants import enable_interruptible_works, ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER
from lightning_app.storage.mount import Mount

__CLOUD_COMPUTE_IDENTIFIER__ = "__cloud_compute__"


@dataclass
class _CloudComputeStore:
    id: str
    component_names: List[str]

    def add_component_name(self, new_component_name: str) -> None:
        found_index = None
        # When the work is being named by the flow, pop its previous names
        for index, component_name in enumerate(self.component_names):
            if new_component_name.endswith(component_name.replace("root.", "")):
                found_index = index

        if found_index is not None:
            self.component_names[found_index] = new_component_name
        else:
            if (
                len(self.component_names) == 1
                and not ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER
                and self.id != "default"
            ):
                raise Exception(
                    f"A Cloud Compute can be assigned only to a single Work. Attached to {self.component_names[0]}"
                )
            self.component_names.append(new_component_name)

    def remove(self, new_component_name: str) -> None:
        found_index = None
        for index, component_name in enumerate(self.component_names):
            if new_component_name == component_name:
                found_index = index

        if found_index is not None:
            del self.component_names[found_index]


_CLOUD_COMPUTE_STORE = {}


@dataclass
class CloudCompute:
    """Configure the cloud runtime for a lightning work or flow.

    Arguments:
        name: The name of the hardware to use. A full list of supported options can be found in
            :doc:`/core_api/lightning_work/compute`. If you have a request for more hardware options, please contact
            `onprem@lightning.ai <mailto:onprem@lightning.ai>`_.

        disk_size: The disk size in Gigabytes.
            The value you set here will be allocated to the /home folder.

        idle_timeout: The number of seconds to wait before pausing the compute when the work is running and idle.
            This timeout starts whenever your run() method succeeds (or fails).
            If the timeout is reached, the instance pauses until the next run() call happens.

        shm_size: Shared memory size in MiB, backed by RAM. min 512, max 8192, it will auto update in steps of 512.
            For example 1100 will become 1024. If set to zero (the default) will get the default 64MiB inside docker.

        mounts: External data sources which should be mounted into a work as a filesystem at runtime.

        interruptible: Whether to run on a interruptible machine e.g the machine can be stopped
            at any time by the providers. This is also known as spot or preemptible machines.
            Compared to on-demand machines, they tend to be cheaper.
    """

    name: str = "default"
    disk_size: int = 0
    idle_timeout: Optional[int] = None
    shm_size: Optional[int] = None
    mounts: Optional[Union[Mount, List[Mount]]] = None
    interruptible: bool = False
    _internal_id: Optional[str] = None

    def __post_init__(self) -> None:
        _verify_mount_root_dirs_are_unique(self.mounts)

        self.name = self.name.lower()

        if self.shm_size is None:
            if "gpu" in self.name:
                self.shm_size = 1024
            else:
                self.shm_size = 0

        if self.interruptible:
            if not enable_interruptible_works():
                raise ValueError("CloudCompute with `interruptible=True` isn't supported yet.")
            if "gpu" not in self.name:
                raise ValueError("CloudCompute `interruptible=True` is supported only with GPU.")

        # TODO: Remove from the platform first.
        self.preemptible = self.interruptible

        # All `default` CloudCompute are identified in the same way.
        if self._internal_id is None:
            self._internal_id = self._generate_id()

    def to_dict(self) -> dict:
        _verify_mount_root_dirs_are_unique(self.mounts)
        return {"type": __CLOUD_COMPUTE_IDENTIFIER__, **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "CloudCompute":
        assert d.pop("type") == __CLOUD_COMPUTE_IDENTIFIER__
        mounts = d.pop("mounts", None)
        if mounts is None:
            pass
        elif isinstance(mounts, dict):
            d["mounts"] = Mount(**mounts)
        elif isinstance(mounts, (list)):
            d["mounts"] = []
            for mount in mounts:
                d["mounts"].append(Mount(**mount))
        else:
            raise TypeError(
                f"mounts argument must be one of [None, Mount, List[Mount]], "
                f"received {mounts} of type {type(mounts)}"
            )
        _verify_mount_root_dirs_are_unique(d.get("mounts", None))
        return cls(**d)

    @property
    def id(self) -> Optional[str]:
        return self._internal_id

    def is_default(self) -> bool:
        return self.name == "default"

    def _generate_id(self):
        return "default" if self.name == "default" else uuid4().hex[:7]

    def clone(self):
        new_dict = self.to_dict()
        new_dict["_internal_id"] = self._generate_id()
        return self.from_dict(new_dict)


def _verify_mount_root_dirs_are_unique(mounts: Union[None, Mount, List[Mount], Tuple[Mount]]) -> None:
    if isinstance(mounts, (list, tuple, set)):
        mount_paths = [mount.mount_path for mount in mounts]
        if len(set(mount_paths)) != len(mount_paths):
            raise ValueError("Every Mount attached to a work must have a unique 'mount_path' argument.")


def _maybe_create_cloud_compute(state: Dict) -> Union[CloudCompute, Dict]:
    if state and __CLOUD_COMPUTE_IDENTIFIER__ == state.get("type", None):
        cloud_compute = CloudCompute.from_dict(state)
        return cloud_compute
    return state
