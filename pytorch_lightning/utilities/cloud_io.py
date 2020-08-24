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
import platform
import os
from distutils.version import LooseVersion
from typing import Union
from pathlib import Path
from urllib.parse import urlparse
import torch

import tensorboard
from packaging import version
from pytorch_lightning import _logger as log

# we want this for tf.io.gfile, which if tf is installed gives full tf,
# otherwise gives a pruned down version which works for some file backends but
# not all
from tensorboard.compat import tf

gfile = tf.io.gfile

pathlike = Union[Path, str]

# older version of tensorboard had buggy gfile compatibility layers
# only support remote cloud paths if newer


def load(path_or_url: str, map_location=None):
    if urlparse(path_or_url).scheme == '' or Path(path_or_url).drive:  # no scheme or with a drive letter
        return torch.load(path_or_url, map_location=map_location)
    return torch.hub.load_state_dict_from_url(path_or_url, map_location=map_location)


def is_remote_path(path: pathlike):
    """Determine if a path is a local path or a remote path like s3://bucket/path

    This should catch paths like s3:// hdfs:// and gcs://
    """
    return "://" in str(path)


def modern_gfile():
    """Check the version number of tensorboard.

    Cheking to see if it has the gfile compatibility layers needed for remote
    file operations
    """
    tb_version = version.parse(tensorboard.version.VERSION)
    modern_gfile = tb_version >= version.parse("2.0")
    return modern_gfile


def cloud_open(path: pathlike, mode: str, newline: str = None):
    if platform.system() == "Windows":
        log.debug(
            "gfile does not handle newlines correctly on windows so remote files are not"
            " supported falling back to normal local file open."
        )
        return open(path, mode, newline=newline)
    if not modern_gfile():
        log.debug(
            "tenosrboard.compat gfile does not work on older versions "
            "of tensorboard for remote files, using normal local file open."
        )
        return open(path, mode, newline=newline)
    try:
        return gfile.GFile(path, mode)
    except NotImplementedError as e:
        # minimal dependencies are installed and only local files will work
        return open(path, mode, newline=newline)


def makedirs(path: pathlike):
    if hasattr(gfile, "makedirs") and modern_gfile():
        if not gfile.exists(str(path)):
            return gfile.makedirs(str(path))
    # otherwise minimal dependencies are installed and only local files will work
    return os.makedirs(path, exist_ok=True)


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
    if LooseVersion(torch.__version__).version[:3] == [1, 6, 0]:
        torch.save(checkpoint, bytesbuffer, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, bytesbuffer)
    with cloud_open(filepath, 'wb') as f:
        f.write(bytesbuffer.getvalue())
