from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class CloudCompute:
    """
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
    """

    name: str = "default"
    disk_size: int = 0
    idle_timeout: Optional[int] = None
    shm_size: Optional[int] = 0

    def __post_init__(self):
        self.name = self.name.lower()

    def to_dict(self):
        return {"__cloud_compute__": asdict(self)}

    @classmethod
    def from_dict(cls, d):
        return cls(**d["__cloud_compute__"])
