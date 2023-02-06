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
from pathlib import Path
from shutil import rmtree
from typing import List, Optional

from lightning_app.core.constants import DOT_IGNORE_FILENAME
from lightning_app.source_code.copytree import _copytree, _IGNORE_FUNCTION
from lightning_app.source_code.hashing import _get_hash
from lightning_app.source_code.tar import _tar_path
from lightning_app.source_code.uploader import FileUploader


class LocalSourceCodeDir:
    """Represents the source code directory and provide the utilities to manage it."""

    cache_location: Path = Path.home() / ".lightning" / "cache" / "repositories"

    def __init__(self, path: Path, ignore_functions: Optional[List[_IGNORE_FUNCTION]] = None) -> None:
        self.path = path
        self.ignore_functions = ignore_functions

        # cache checksum version
        self._version: Optional[str] = None
        self._non_ignored_files: Optional[List[str]] = None

        # create global cache location if it doesn't exist
        if not self.cache_location.exists():
            self.cache_location.mkdir(parents=True, exist_ok=True)

        # Create a default dotignore if it doesn't exist
        if not (path / DOT_IGNORE_FILENAME).is_file():
            with open(path / DOT_IGNORE_FILENAME, "w") as f:
                f.write("venv/\n")
                if (path / "bin" / "activate").is_file() or (path / "pyvenv.cfg").is_file():
                    # the user is developing inside venv
                    f.write("bin/\ninclude/\nlib/\npyvenv.cfg\n")

        # clean old cache entries
        self._prune_cache()

    @property
    def files(self) -> List[str]:
        """Returns a set of files that are not ignored by .lightningignore."""
        if self._non_ignored_files is None:
            self._non_ignored_files = _copytree(self.path, "", ignore_functions=self.ignore_functions, dry_run=True)
        return self._non_ignored_files

    @property
    def version(self):
        """Calculates the checksum of a local path."""
        # cache value to prevent doing this over again
        if self._version is not None:
            return self._version

        # stores both version and a set with the files used to generate the checksum
        self._version = _get_hash(files=self.files, algorithm="blake2")
        return self._version

    @property
    def package_path(self):
        """Location to tarball in local cache."""
        filename = f"{self.version}.tar.gz"
        return self.cache_location / filename

    @contextmanager
    def packaging_session(self) -> Path:
        """Creates a local directory with source code that is used for creating a local source-code package."""
        session_path = self.cache_location / "packaging_sessions" / self.version
        try:
            rmtree(session_path, ignore_errors=True)
            _copytree(self.path, session_path, ignore_functions=self.ignore_functions)
            yield session_path
        finally:
            rmtree(session_path, ignore_errors=True)

    def _prune_cache(self) -> None:
        """Prunes cache; only keeps the 10 most recent items."""
        packages = sorted(self.cache_location.iterdir(), key=os.path.getmtime)
        for package in packages[10:]:
            if package.is_file():
                package.unlink()

    def package(self) -> Path:
        """Packages local path using tar."""
        if self.package_path.exists():
            return self.package_path
        # create a packaging session if not available
        with self.packaging_session() as session_path:
            _tar_path(source_path=session_path, target_file=str(self.package_path), compression=True)
        return self.package_path

    def upload(self, url: str) -> None:
        """Uploads package to URL, usually pre-signed URL.

        Notes
        -----
        Since we do not use multipart uploads here, we cannot upload any
        packaged repository files which have a size > 2GB.

        This limitation should be removed during the datastore upload redesign
        """
        if self.package_path.stat().st_size > 2e9:
            raise OSError(
                "cannot upload directory code whose total fize size is greater than 2GB (2e9 bytes)"
            ) from None

        uploader = FileUploader(
            presigned_url=url,
            source_file=str(self.package_path),
            name=self.package_path.name,
            total_size=self.package_path.stat().st_size,
        )
        uploader.upload()
