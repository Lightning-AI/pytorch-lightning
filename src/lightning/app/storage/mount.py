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
from dataclasses import dataclass
from pathlib import Path
from typing import List

__MOUNT_IDENTIFIER__: str = "__mount__"
__MOUNT_PROTOCOLS__: List[str] = ["s3://"]


@dataclass
class Mount:
    """Allows you to mount the contents of an AWS S3 bucket on disk when running an app on the cloud.

    Arguments:
        source: The location which contains the external data which should be mounted in the
            running work. At the moment, only AWS S3 mounts are supported. This must be a full
            `s3` style identifier pointing to a bucket and (optionally) prefix to mount. For
            example: `s3://foo/bar/`.

        mount_path: An absolute directory path in the work where external data source should
            be mounted as a filesystem. This path should not already exist in your codebase.
            If not included, then the root_dir will be set to `/data/<last folder name in the bucket>`
    """

    source: str = ""
    mount_path: str = ""

    def __post_init__(self) -> None:
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

        if self.mount_path == "":
            self.mount_path = f"/data/{Path(self.source).stem}"

        if not os.path.isabs(self.mount_path):
            raise ValueError(
                f"mount_path argument must be an absolute path to a "
                f"location; received relative path {self.mount_path}"
            )

    @property
    def protocol(self) -> str:
        """The backing storage protocol indicated by this drive source."""
        for protocol in __MOUNT_PROTOCOLS__:
            if self.source.startswith(protocol):
                return protocol
        return ""
