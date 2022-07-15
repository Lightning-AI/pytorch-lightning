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
import logging
import os
import queue
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.utilities.cloud_io import _atomic_save, atomic_save, get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.cloud_io import ThreadQueue
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import _PATH

log = logging.getLogger(__name__)


class TorchCheckpointIO(CheckpointIO):
    """CheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save and load checkpoints
    respectively, common for most use cases.

    Args:
        save_async: whether to save the checkpoint asynchronously or not.
        num_threads: Number of threads to use for asynchronous checkpointing.
    """

    def __init__(self, save_async: bool = False, num_threads: Optional[int] = None):

        if save_async and not (isinstance(num_threads, int) and (num_threads >= 0)):
            raise MisconfigurationException(
                f"Asynchronous checkpoint is not possible with `num_threds={num_threads!r}`."
            )

        self.queue = None
        self.threads = None

        if save_async:
            self.queue = queue.Queue()
            assert isinstance(num_threads, int)
            assert self.queue is not None
            self.threads = [ThreadQueue(func=_atomic_save, q=self.queue) for _ in range(num_threads)]
            for thread in self.threads:
                thread.start()

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``TorchCheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # write the checkpoint dictionary on the file
            atomic_save(checkpoint, path, self.threads, self.queue)
        except AttributeError as err:
            # todo (sean): is this try catch necessary still?
            # https://github.com/Lightning-AI/lightning/pull/431
            key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
            checkpoint.pop(key, None)
            rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            atomic_save(checkpoint, path, self.threads, self.queue)

    def load_checkpoint(
        self, path: _PATH, map_location: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        """Loads checkpoint using :func:`torch.load`, with additional handling for ``fsspec`` remote loading of
        files.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations.

        Returns: The loaded checkpoint.

        Raises:
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem
        """

        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint at {path} not found. Aborting training.")

        return pl_load(path, map_location=map_location)

    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint
        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug(f"Removed checkpoint: {path}")

    def on_train_end(self) -> None:
        if self.threads is None:
            return

        for thread in self.threads:
            thread.join()
