from abc import ABC
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from torch.nn import Module

from pytorch_lightning import LightningModule


class CheckpointPlugin(ABC):

    def __init__(self):
        self._training_type_plugin = None

    @property
    def training_type_plugin(self) -> 'TrainingTypePlugin':
        return self._training_type_plugin

    @training_type_plugin.setter
    def training_type_plugin(self, plugin) -> None:
        self._training_type_plugin = plugin

    @property
    def lightning_module(self) -> LightningModule:
        return self.training_type_plugin.lightning_module

    @property
    def model(self) -> Module:
        return self.training_type_plugin.model

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
        pass

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """
        Given the loaded checkpoint file, loads the state dict into the model.
        Args:
            checkpoint: The loaded checkpoint file.
        """

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """
        Given the loaded checkpoint file, loads the optimizer state dicts into optimizers.
        Args:
            checkpoint: The loaded checkpoint file.
        """
