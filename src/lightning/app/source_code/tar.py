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

import math
import os
import subprocess
import tarfile
from dataclasses import dataclass
from typing import Optional, Tuple

import click

MAX_SPLIT_COUNT = 999


def _get_dir_size_and_count(source_dir: str, prefix: Optional[str] = None) -> Tuple[int, int]:
    """Get size and file count of a directory.

    Parameters
    ----------
    source_dir: str
        Directory path

    Returns
    -------
    Tuple[int, int]
        Size in megabytes and file count
    """
    size = 0
    count = 0
    for root, _, files in os.walk(source_dir, topdown=True):
        for f in files:
            if prefix and not f.startswith(prefix):
                continue

            full_path = os.path.join(root, f)
            size += os.path.getsize(full_path)
            count += 1

    return (size, count)


@dataclass
class _TarResults:
    """This class holds the results of running tar_path.

    Attributes
    ----------
    before_size: int
        The total size of the original directory files in bytes
    after_size: int
        The total size of the compressed and tarred split files in bytes
    """

    before_size: int
    after_size: int


def _get_split_size(
    total_size: int, minimum_split_size: int = 1024 * 1000 * 20, max_split_count: int = MAX_SPLIT_COUNT
) -> int:
    """Calculate the split size we should use to split the multipart upload of an object to a bucket.  We are
    limited to 1000 max parts as the way we are using ListMultipartUploads. More info
    https://github.com/gridai/grid/pull/5267
    https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html#mpu-process
    https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListMultipartUploads.html
    https://github.com/psf/requests/issues/2717#issuecomment-724725392 Python or requests has a limit of 2**31
    bytes for a single file upload.

    Parameters
    ----------
    minimum_split_size: int
        The minimum split size to use
    max_split_count: int
        The maximum split count
    total_size: int
        Total size of the file to split

    Returns
    -------
    int
        Split size
    """
    max_size = max_split_count * (1 << 31)  # max size per part limited by Requests or urllib as shown in ref above
    if total_size > max_size:
        raise click.ClickException(
            f"The size of the datastore to be uploaded is bigger than our {max_size/(1 << 40):.2f} TBytes limit"
        )

    split_size = minimum_split_size
    split_count = math.ceil(total_size / split_size)
    if split_count > max_split_count:
        # Adjust the split size based on max split count
        split_size = math.ceil(total_size / max_split_count)

    return split_size


def _tar_path(source_path: str, target_file: str, compression: bool = False) -> _TarResults:
    """Create tar from directory using `tar`

    Parameters
    ----------
    source_path: str
        Source directory or file
    target_file
        Target tar file
    compression: bool, default False
        Enable compression, which is disabled by default.

    Returns
    -------
    TarResults
        Results that holds file counts and sizes
    """
    if os.path.isdir(source_path):
        before_size, _ = _get_dir_size_and_count(source_path)
    else:
        before_size = os.path.getsize(source_path)

    try:
        _tar_path_subprocess(source_path, target_file, compression)
    except subprocess.CalledProcessError:
        _tar_path_python(source_path, target_file, compression)

    after_size = os.stat(target_file).st_size
    return _TarResults(before_size=before_size, after_size=after_size)


def _tar_path_python(source_path: str, target_file: str, compression: bool = False) -> None:
    """Create tar from directory using `python`

    Parameters
    ----------
    source_path: str
        Source directory or file
    target_file
        Target tar file
    compression: bool, default False
        Enable compression, which is disabled by default.
    """
    file_mode = "w:gz" if compression else "w:"

    with tarfile.open(target_file, file_mode) as tar:
        if os.path.isdir(source_path):
            tar.add(str(source_path), arcname=".")
        elif os.path.isfile(source_path):
            file_info = tarfile.TarInfo(os.path.basename(str(source_path)))
            tar.addfile(file_info, open(source_path))


def _tar_path_subprocess(source_path: str, target_file: str, compression: bool = False) -> None:
    """Create tar from directory using `tar`

    Parameters
    ----------
    source_path: str
        Source directory or file
    target_file
        Target tar file
    compression: bool, default False
        Enable compression, which is disabled by default.
    """
    # Only add compression when users explicitly request it.
    # We do this because it takes too long to compress
    # large datastores.
    tar_flags = "-cvf"
    if compression:
        tar_flags = "-zcvf"
    if os.path.isdir(source_path):
        command = f"tar -C {source_path} {tar_flags} {target_file} ./"
    else:
        abs_path = os.path.abspath(source_path)
        parent_dir = os.path.dirname(abs_path)
        base_name = os.path.basename(abs_path)
        command = f"tar -C {parent_dir} {tar_flags} {target_file} {base_name}"

    subprocess.check_call(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
        env={"GZIP": "-9", "COPYFILE_DISABLE": "1"},
    )
