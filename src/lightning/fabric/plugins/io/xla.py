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
import logging
from typing import Any, Optional

from typing_extensions import override

from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)


class XLACheckpointIO(TorchCheckpointIO):
    """CheckpointIO that utilizes ``xm.save`` to save checkpoints for TPU training strategies.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.io.xla import XLACheckpointIO as EnterpriseXLACheckpointIO

        self.xla_impl = EnterpriseXLACheckpointIO(*args, **kwargs)

    @override
    def save_checkpoint(self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``XLACheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in

        """
        return self.xla_impl.save_checkpoint(checkpoint, path, storage_options)
