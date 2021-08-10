from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union


class CheckpointIOPlugin(ABC):
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
