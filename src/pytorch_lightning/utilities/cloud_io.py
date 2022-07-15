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
"""Utilities related to data saving/loading."""

import io
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional, Union

import fsspec
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem

from pytorch_lightning.utilities.types import _PATH

_state_observer_lock = threading.Lock()


class ThreadQueue(threading.Thread):
    def __init__(self, func: Callable, q: queue.Queue, interval: int = 5) -> None:
        super().__init__(daemon=True)
        self.func = func
        self.q = q
        self._close_thread = False
        self._interval = interval

    def run(self) -> None:
        with _state_observer_lock:
            while not (self._close_thread and self.q.empty()):
                time.sleep(self._interval)
                args = self.q.get()
                self.func(*args)
                self.q.task_done()

    def join(self, timeout: Optional[float] = None) -> None:
        self._close_thread = True
        self.q.join()
        super().join(timeout)


def load(
    path_or_url: Union[IO, _PATH],
    map_location: Optional[
        Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]]
    ] = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
    """
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)


def get_filesystem(path: _PATH, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


def atomic_save(
    checkpoint: Dict[str, Any], filepath: Union[str, Path], threads: List[ThreadQueue] = None, queue: queue.Queue = None
) -> None:
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
    """
    if threads is not None:
        assert queue is not None
        queue.put((checkpoint, filepath))
    else:
        _atomic_save(checkpoint, filepath)


def _atomic_save(checkpoint: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
    """
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer)
    with fsspec.open(filepath, "wb") as f:
        f.write(bytesbuffer.getvalue())
