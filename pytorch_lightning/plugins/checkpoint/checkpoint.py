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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union


class CheckpointIOPlugin(ABC):
    """
    Interface to save/load checkpoints as they are saved through the ``TrainingTypePlugin``.

    Typically most plugins either use the Torch based IO Plugin; ``TorchCheckpointIOPlugin`` but may
    require particular handling depending the plugin.

    In addition, you can pass a custom ``CheckpointIOPlugin`` by extending this class and passing it
    to the Trainer, i.e ``Trainer(plugins=[MyCustomCheckpointIOPlugin()])``.

    .. note::

        For some plugins it is not possible to use a custom checkpoint plugin as checkpointing logic is not
        modifiable.

    """

    @abstractmethod
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: Optional parameters when saving the model/training states.
        """

    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path], storage_options: Optional[Any] = None) -> Dict[str, Any]:
        """
        Load checkpoint from a path when resuming or loading ckpt for test/validate/predict stages.

        Args:
            path: Path to checkpoint
            storage_options: Optional parameters when loading the model/training states.

        Returns: The loaded checkpoint.
        """
