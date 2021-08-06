from abc import ABC
from pathlib import Path
from typing import Any, Dict, Union


class CheckpointPlugin(ABC):
    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: str) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """

    def load_checkpoint_file(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load checkpoint from a path when resuming or loading ckpt for test/validate/predict stages.
        Args:
            checkpoint_path: Path to checkpoint

        Returns: The loaded checkpoint.
        """
