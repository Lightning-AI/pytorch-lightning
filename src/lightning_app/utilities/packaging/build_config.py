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

import inspect
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from typing_extensions import Self

import lightning_app as L
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.packaging.cloud_compute import CloudCompute

logger = Logger(__name__)


def load_requirements(
    path_dir: str, file_name: str = "base.txt", comment_char: str = "#", unfreeze: bool = True
) -> List[str]:
    """Load requirements from a file.

    >>> from lightning_app import _PROJECT_ROOT
    >>> path_req = os.path.join(_PROJECT_ROOT, "requirements")
    >>> load_requirements(path_req, "docs.txt")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE +SKIP
    ['sphinx>=4.0', ...]  # TODO: remove SKIP, fails on python 3.7
    """
    path = os.path.join(path_dir, file_name)
    if not os.path.isfile(path):
        return []

    with open(path) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        comment = ""
        if comment_char in ln:
            comment = ln[ln.index(comment_char) :]
            ln = ln[: ln.index(comment_char)]
        req = ln.strip()
        # skip directly installed dependencies
        if not req or req.startswith("http") or "@http" in req:
            continue
        # remove version restrictions unless they are strict
        if unfreeze and "<" in req and "strict" not in comment:
            req = re.sub(r",? *<=? *[\d\.\*]+", "", req).strip()
        reqs.append(req)
    return reqs


@dataclass
class _Dockerfile:
    path: str
    data: List[str]


@dataclass
class BuildConfig:
    """The Build Configuration describes how the environment a LightningWork runs in should be set up.

    Arguments:
        requirements: List of requirements or list of paths to requirement files. If not passed, they will be
            automatically extracted from a `requirements.txt` if it exists.
        dockerfile: The path to a dockerfile to be used to build your container.
            You need to add those lines to ensure your container works in the cloud.

            .. warning:: This feature isn't supported yet, but coming soon.

            Example::

                WORKDIR /gridai/project
                COPY . .
        image: The base image that the work runs on. This should be a publicly accessible image from a registry that
            doesn't enforce rate limits (such as DockerHub) to pull this image, otherwise your application will not
            start.
    """

    requirements: List[str] = field(default_factory=list)
    dockerfile: Optional[Union[str, Path, _Dockerfile]] = None
    image: Optional[str] = None

    def __post_init__(self) -> None:
        current_frame = inspect.currentframe()
        co_filename = current_frame.f_back.f_back.f_code.co_filename  # type: ignore[union-attr]
        self._call_dir = os.path.dirname(co_filename)
        self._prepare_requirements()
        self._prepare_dockerfile()

    def build_commands(self) -> List[str]:
        """Override to run some commands before your requirements are installed.

        .. note:: If you provide your own dockerfile, this would be ignored.

        Example:

            from dataclasses import dataclass
            from lightning_app import BuildConfig

            @dataclass
            class MyOwnBuildConfig(BuildConfig):

                def build_commands(self):
                    return ["apt-get install libsparsehash-dev"]

            BuildConfig(requirements=["git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0"])
        """
        return []

    def on_work_init(self, work: "L.LightningWork", cloud_compute: Optional["CloudCompute"] = None) -> None:
        """Override with your own logic to load the requirements or dockerfile."""
        found_requirements = self._find_requirements(work)
        if self.requirements:
            if found_requirements and self.requirements != found_requirements:
                # notify the user of this silent behaviour
                logger.info(
                    f"A 'requirements.txt' exists with {found_requirements} but {self.requirements} was passed to"
                    f" the `{type(self).__name__}` in {work.name!r}. The `requirements.txt` file will be ignored."
                )
        else:
            self.requirements = found_requirements
        self._prepare_requirements()

        found_dockerfile = self._find_dockerfile(work)
        if self.dockerfile:
            if found_dockerfile and self.dockerfile != found_dockerfile:
                # notify the user of this silent behaviour
                logger.info(
                    f"A Dockerfile exists at {found_dockerfile!r} but {self.dockerfile!r} was passed to"
                    f" the `{type(self).__name__}` in {work.name!r}. {found_dockerfile!r}` will be ignored."
                )
        else:
            self.dockerfile = found_dockerfile
        self._prepare_dockerfile()

    def _find_requirements(self, work: "L.LightningWork", filename: str = "requirements.txt") -> List[str]:
        # 1. Get work file
        file = _get_work_file(work)
        if file is None:
            return []
        # 2. Try to find a requirement file associated the file.
        dirname = os.path.dirname(file)
        try:
            requirements = load_requirements(dirname, filename)
        except NotADirectoryError:
            return []
        return [r for r in requirements if r != "lightning"]

    def _find_dockerfile(self, work: "L.LightningWork", filename: str = "Dockerfile") -> Optional[str]:
        # 1. Get work file
        file = _get_work_file(work)
        if file is None:
            return None
        # 2. Check for Dockerfile.
        dirname = os.path.dirname(file)
        dockerfile = os.path.join(dirname, filename)
        if os.path.isfile(dockerfile):
            return dockerfile

    def _prepare_requirements(self) -> None:
        requirements = []
        for req in self.requirements:
            # 1. Check for relative path
            path = os.path.join(self._call_dir, req)
            if os.path.isfile(path):
                try:
                    new_requirements = load_requirements(self._call_dir, req)
                except NotADirectoryError:
                    continue
                requirements.extend(new_requirements)
            else:
                requirements.append(req)
        self.requirements = requirements

    def _prepare_dockerfile(self) -> None:
        if isinstance(self.dockerfile, (str, Path)):
            path = os.path.join(self._call_dir, self.dockerfile)
            if os.path.exists(path):
                with open(path) as f:
                    self.dockerfile = _Dockerfile(path, f.readlines())

    def to_dict(self) -> Dict:
        return {"__build_config__": asdict(self)}

    @classmethod
    def from_dict(cls, d: Dict) -> Self:  # type: ignore[valid-type]
        return cls(**d["__build_config__"])


def _get_work_file(work: "L.LightningWork") -> Optional[str]:
    cls = work.__class__
    try:
        return inspect.getfile(cls)
    except TypeError:
        logger.debug(f"The {cls.__name__} file couldn't be found.")
        return None
