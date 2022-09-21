from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union
from uuid import uuid4

__CLOUD_COMPUTE_IDENTIFIER__ = "__cloud_compute__"


@dataclass
class CloudCompute:
    """
    Arguments:
        name: The name of the hardware to use. A full list of supported options can be found in
            :doc:`/core_api/lightning_work/compute`. If you have a request for more hardware options, please contact
            `onprem@lightning.ai <mailto:onprem@lightning.ai>`_.

        disk_size: The disk size in Gigabytes.
            The value you set here will be allocated to the /home folder.

        clusters: Name of the cluster or a list of cluster names.
            The cluster(s) must already exist.
            If multiple clusters are provided, we try one by one until we can allocate the
            resources we need in the order they were provided.
            Cluster default to the Grid Default Cluster.

        preemptible: Whether to use a preemptible / spot instance.
            If none are available at the moment, we will wait forever or up to the specified timeout
            (see wait_timeout argument).
            Default: False (on-demand instance)

        wait_timeout: The number of seconds to wait before giving up on the getting the requested compute.
            If used in combination with spot instance (spot preemptible=True) and the timeout is reached,
            falls back to regular instance type and waits again for this amount.

        idle_timeout: The number of seconds to wait before pausing the compute when the work is running and idle.
            This timeout starts whenever your run() method succeeds (or fails).
            If the timeout is reached, the instance pauses until the next run() call happens.

        shm_size: Shared memory size in MiB, backed by RAM. min 512, max 8192, it will auto update in steps of 512.
            For example 1100 will become 1024. If set to zero (the default) will get the default 64MiB inside docker.
    """

    name: str = "default"
    disk_size: int = 0
    clusters: Optional[Union[str, List[str]]] = None
    preemptible: bool = False
    wait_timeout: Optional[int] = None
    idle_timeout: Optional[int] = None
    shm_size: Optional[int] = 0

    def __post_init__(self):
        if self.clusters:
            raise ValueError("Clusters are't supported yet. Coming soon.")
        if self.wait_timeout:
            raise ValueError("Setting a wait timeout isn't supported yet. Coming soon.")

        self.name = self.name.lower()

        # All `default` CloudCompute are identified in the same way.
        self._internal_id = "default" if self.name == "default" else uuid4().hex

    def to_dict(self):
        return {"type": __CLOUD_COMPUTE_IDENTIFIER__, **asdict(self), "_internal_id": self._internal_id}

    @classmethod
    def from_dict(cls, d):
        assert d.pop("type") == __CLOUD_COMPUTE_IDENTIFIER__
        internal_id = d.pop("_internal_id")
        cloud_compute = cls(**d)
        cloud_compute._internal_id = internal_id
        return cloud_compute


def _maybe_create_cloud_compute(state: Dict) -> Union[CloudCompute, Dict]:
    if __CLOUD_COMPUTE_IDENTIFIER__ == state.get("type", None):
        cloud_compute = CloudCompute.from_dict(state)
        return cloud_compute
    return state
