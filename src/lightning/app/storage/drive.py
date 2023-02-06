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
import pathlib
import shutil
import sys
from copy import deepcopy
from time import sleep, time
from typing import Dict, List, Optional, Union

from lightning.app.storage.path import _filesystem, _shared_storage_path, LocalFileSystem
from lightning.app.utilities.component import _is_flow_context


class Drive:
    __IDENTIFIER__ = "__drive__"
    __PROTOCOLS__ = ["lit://"]

    def __init__(
        self,
        id: str,
        allow_duplicates: bool = False,
        component_name: Optional[str] = None,
        root_folder: Optional[str] = None,
    ):
        """The Drive object provides a shared space to write and read files from.

        When the drive object is passed from one component to another, a copy is made and ownership
        is transferred to the new component.

        Arguments:
            id: Unique identifier for this Drive.
            allow_duplicates: Whether to enable files duplication between components.
            component_name: The component name which owns this drive.
                When not provided, it is automatically inferred by Lightning.
            root_folder: This is the folder from where the Drive perceives the data (e.g this acts as a mount dir).
        """
        if id.startswith("s3://"):
            raise ValueError(
                "Using S3 buckets in a Drive is no longer supported. Please pass an S3 `Mount` to "
                "a Work's CloudCompute config in order to mount an s3 bucket as a filesystem in a work.\n"
                f"`CloudCompute(mount=Mount({id}), ...)`"
            )

        self.id = None
        self.protocol = None
        for protocol in self.__PROTOCOLS__:
            if id.startswith(protocol):
                self.protocol = protocol
                self.id = id.replace(protocol, "")
                break
        else:  # N.B. for-else loop
            raise ValueError(
                f"Unknown protocol for the drive 'id' argument '{id}`. The 'id' string "
                f"must start with one of the following prefixes {self.__PROTOCOLS__}"
            )

        if not self.id:
            raise Exception(f"The Drive id needs to start with one of the following protocols: {self.__PROTOCOLS__}")

        if "/" in self.id:
            raise Exception(f"The id should be unique to identify your drive. Found `{self.id}`.")

        self.root_folder = pathlib.Path(root_folder).resolve() if root_folder else pathlib.Path(os.getcwd())
        if self.protocol != "s3://" and not os.path.isdir(self.root_folder):
            raise Exception(f"The provided root_folder isn't a directory: {root_folder}")
        self.component_name = component_name
        self.allow_duplicates = allow_duplicates
        self.fs = _filesystem()

    @property
    def root(self) -> pathlib.Path:
        root_path = self.drive_root / self.component_name
        if isinstance(self.fs, LocalFileSystem):
            self.fs.makedirs(root_path, exist_ok=True)
        return root_path

    @property
    def drive_root(self) -> pathlib.Path:
        drive_root = _shared_storage_path() / "artifacts" / "drive" / self.id
        return drive_root

    def put(self, path: str) -> None:
        """This method enables to put a file to the Drive in a blocking fashion.

        Arguments:
            path: The relative path to your files to be added to the Drive.
        """
        if not self.component_name:
            raise Exception("The component name needs to be known to put a path to the Drive.")
        if _is_flow_context():
            raise Exception("The flow isn't allowed to put files into a Drive.")

        self._validate_path(path)

        if not self.allow_duplicates:
            self._check_for_allow_duplicates(path)

        from lightning.app.storage.copier import _copy_files

        src = pathlib.Path(os.path.join(self.root_folder, path)).resolve()
        dst = self._to_shared_path(path, component_name=self.component_name)

        _copy_files(src, dst)

    def list(self, path: Optional[str] = ".", component_name: Optional[str] = None) -> List[str]:
        """This method enables to list files under the provided path from the Drive in a blocking fashion.

        Arguments:
            path: The relative path you want to list files from the Drive.
            component_name: By default, the Drive lists files across all components.
                If you provide a component name, the listing is specific to this component.
        """
        if _is_flow_context():
            raise Exception("The flow isn't allowed to list files from a Drive.")

        if component_name:
            paths = [
                self._to_shared_path(
                    path,
                    component_name=component_name,
                )
            ]
        else:
            paths = [
                self._to_shared_path(
                    path,
                    component_name=component_name,
                )
                for component_name in self._collect_component_names()
            ]

        files = []
        sep = "\\" if sys.platform == "win32" else "/"
        prefix_len = len(str(self.root).split(sep))
        for p in paths:
            if self.fs.exists(p):
                for f in self.fs.ls(p):
                    files.append(str(pathlib.Path(*pathlib.Path(f).parts[prefix_len:])))
        return files

    def get(
        self,
        path: str,
        component_name: Optional[str] = None,
        timeout: Optional[float] = None,
        overwrite: bool = False,
    ) -> None:
        """This method enables to get files under the provided path from the Drive in a blocking fashion.

        Arguments:
            path: The relative path you want to list files from the Drive.
            component_name: By default, the Drive get the matching files across all components.
                If you provide a component name, the matching is specific to this component.
            timeout: Whether to wait for the files to be available if not created yet.
            overwrite: Whether to override the provided path if it exists.
        """
        if _is_flow_context():
            raise Exception("The flow isn't allowed to get files from a Drive.")

        if component_name:
            shared_path = self._to_shared_path(
                path,
                component_name=component_name,
            )
            if timeout:
                start_time = time()
                while not self.fs.exists(shared_path):
                    sleep(1)
                    if (time() - start_time) > timeout:
                        raise Exception(f"The following {path} wasn't found in {timeout} seconds")
                    break

            self._get(
                self.fs,
                shared_path,
                pathlib.Path(os.path.join(self.root_folder, path)).resolve(),
                overwrite=overwrite,
            )
        else:
            if timeout:
                start_time = time()
                while True:
                    if (time() - start_time) > timeout:
                        raise Exception(f"The following {path} wasn't found in {timeout} seconds.")
                    match = self._find_match(path)
                    if match is None:
                        sleep(1)
                        continue
                    break
            else:
                match = self._find_match(path)
                if not match:
                    raise Exception(f"We didn't find any match for the associated {path}.")

            self._get(self.fs, match, pathlib.Path(os.path.join(self.root_folder, path)).resolve(), overwrite=overwrite)

    def delete(self, path: str) -> None:
        """This method enables to delete files under the provided path from the Drive in a blocking fashion. Only
        the component which added a file can delete them.

        Arguments:
            path: The relative path you want to delete files from the Drive.
        """
        if not self.component_name:
            raise Exception("The component name needs to be known to delete a path to the Drive.")

        shared_path = self._to_shared_path(
            path,
            component_name=self.component_name,
        )
        if self.fs.exists(str(shared_path)):
            self.fs.rm(str(shared_path))
        else:
            raise Exception(f"The file {path} doesn't exists in the component_name space {self.component_name}.")

    def to_dict(self):
        return {
            "type": self.__IDENTIFIER__,
            "id": self.id,
            "protocol": self.protocol,
            "allow_duplicates": self.allow_duplicates,
            "component_name": self.component_name,
            "root_folder": str(self.root_folder),
        }

    @classmethod
    def from_dict(cls, dict: Dict) -> "Drive":
        assert dict["type"] == cls.__IDENTIFIER__
        drive = cls(
            dict["protocol"] + dict["id"],
            allow_duplicates=dict["allow_duplicates"],
            root_folder=dict["root_folder"],
        )
        drive.component_name = dict["component_name"]
        return drive

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def _collect_component_names(self) -> List[str]:
        sep = "/"
        if self.fs.exists(self.drive_root):
            # Invalidate cache before running ls in case new directories have been added
            # TODO: Re-evaluate this - may lead to performance issues
            self.fs.invalidate_cache()
            return [str(p.split(sep)[-1]) for p in self.fs.ls(self.drive_root)]
        return []

    def _to_shared_path(self, path: str, component_name: Optional[str] = None) -> pathlib.Path:
        shared_path = self.drive_root
        if component_name:
            shared_path /= component_name
        shared_path /= path
        return shared_path

    def _get(self, fs, src: pathlib.Path, dst: pathlib.Path, overwrite: bool):
        if fs.isdir(src):
            if isinstance(fs, LocalFileSystem):
                dst = dst.resolve()
                if fs.exists(dst):
                    if overwrite:
                        fs.rm(str(dst), recursive=True)
                    else:
                        raise FileExistsError(f"The file {dst} was found. Add get(..., overwrite=True) to replace it.")

                shutil.copytree(src, dst)
            else:
                glob = f"{str(src)}/**"
                fs.get(glob, str(dst.absolute()), recursive=False)
        else:
            fs.get(str(src), str(dst.absolute()), recursive=False)

    def _find_match(self, path: str) -> Optional[pathlib.Path]:
        matches = []
        for component_name in self._collect_component_names():
            possible_path = self._to_shared_path(path, component_name=component_name)
            if self.fs.exists(possible_path):
                matches.append(possible_path)

        if not matches:
            return None

        if len(matches) > 1:
            sep = "\\" if sys.platform == "win32" else "/"
            prefix_len = len(str(self.root).split(sep))
            matches = [str(pathlib.Path(*pathlib.Path(p).parts[prefix_len:])) for p in matches]
            raise Exception(f"We found several matching files created by multiples components: {matches}.")

        return matches[0]

    def _check_for_allow_duplicates(self, path):
        possible_paths = [
            self._to_shared_path(
                path,
                component_name=component_name,
            )
            for component_name in self._collect_component_names()
            if component_name != self.component_name
        ]
        matches = [self.fs.exists(p) for p in possible_paths]

        if sum(matches):
            raise Exception(f"The file {path} can't be added as already found in the Drive.")

    def _validate_path(self, path: str) -> None:
        if not os.path.exists(os.path.join(self.root_folder, path)):
            raise FileExistsError(f"The provided path {path} doesn't exists")

    def __str__(self) -> str:
        assert self.id
        return self.id


def _maybe_create_drive(component_name: str, state: Dict) -> Union[Dict, Drive]:
    if Drive.__IDENTIFIER__ == state.get("type", None):
        drive = Drive.from_dict(state)
        drive.component_name = component_name
        return drive
    return state
