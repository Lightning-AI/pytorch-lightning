from dataclasses import dataclass
from typing import List

__MOUNT_IDENTIFIER__: str = "__mount__"
__MOUNT_PROTOCOLS__: List[str] = ["s3://"]


@dataclass
class Mount:
    """
    Arguments:
        source: The location which contains the external data which should be mounted in the
            running work. At the moment, only AWS S3 mounts are supported. This must be a full
            `s3` style identifier pointing to a bucket and (optionally) prefix to mount. For
            example: `s3://foo/bar/`.

        root_dir: A fully qualified directory path in the work where external data source should
            be mounted as a filesystem.
    """

    source: str = ""
    root_dir: str = ""

    def __post_init__(self):

        for protocol in __MOUNT_PROTOCOLS__:
            if self.source.startswith(protocol):
                protocol = protocol
                break
        else:  # N.B. for-else loop
            raise ValueError(
                f"Unknown protocol for the mount 'source' argument '{self.source}`. The 'source' "
                f"string must start with one of the following prefixes: {__MOUNT_PROTOCOLS__}"
            )

        if protocol == "s3://" and not self.source.endswith("/"):
            raise ValueError(
                "S3 mounts must end in a trailing slash (`/`) to indicate a folder is being mounted. "
                f"Received: '{self.source}'. Mounting a single file is not currently supported."
            )

        if self.root_dir == "":
            raise ValueError(
                f"The mount for `source` `{self.source}` does not set the required `root_dir` argument. "
                f"Please set this value to indicate the directory where the external data source should "
                f"be mounted in the Work filesystem at runtime."
            )

    @property
    def protocol(self) -> str:
        """The backing storage protocol indicated by this drive source."""
        for protocol in __MOUNT_PROTOCOLS__:
            if self.source.startswith(protocol):
                return protocol
        return ""
