import sys
import os
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


def modern_gfile():
    """Check the version number of tensorboard.

    Cheking to see if it has the gfile compatibility layers needed for remote
    file operations
    """
    tb_version = version.parse(tensorboard.version.VERSION)
    modern_gfile = tb_version >= version.parse('2.0')


def cloud_open(path: pathlike, mode: str, newline: str = None):
    if sys.platform == "win32":
        log.debug(
            "gfile does not handle newlines correctly on windows so remote files are not"
            "supported falling back to normal local file open."
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
        return gfile.makedirs(str(path))
    # otherwise minimal dependencies are installed and only local files will work
    return os.makedirs(path, exist_ok=True)
