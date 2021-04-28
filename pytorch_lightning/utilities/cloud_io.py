# Copyright The PyTorch Lightning team.
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

import io
from pathlib import Path
from typing import IO, Union

import fsspec
import torch
from fsspec.implementations.local import LocalFileSystem
from packaging.version import Version


def load(path_or_url: Union[str, IO, Path], map_location=None):
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similiar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)


def get_filesystem(path: Union[str, Path]):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0])
    else:
        # use local filesystem
        return LocalFileSystem()


def atomic_save(checkpoint, filepath: str):
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
    """

    bytesbuffer = io.BytesIO()
    # Can't use the new zipfile serialization for 1.6.0 because there's a bug in
    # torch.hub.load_state_dict_from_url() that prevents it from loading the new files.
    # More details can be found here: https://github.com/pytorch/pytorch/issues/42239
    if Version(torch.__version__).release[:3] == (1, 6, 0):
        torch.save(checkpoint, bytesbuffer, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, bytesbuffer)
    with fsspec.open(filepath, "wb") as f:
        f.write(bytesbuffer.getvalue())
