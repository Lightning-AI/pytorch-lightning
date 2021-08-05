from pathlib import Path
from typing import Any, Dict, Mapping, Union

from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.plugins.checkpoint.checkpoint import CheckpointPlugin
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.cloud_io import load as pl_load


class TorchCheckpointPlugin(CheckpointPlugin):
    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: str) -> None:
        # dump states as a checkpoint dictionary object
        try:
            # write the checkpoint dictionary on the file
            atomic_save(checkpoint, filepath)
        except AttributeError as err:
            key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
            checkpoint.pop(key, None)
            rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            atomic_save(checkpoint, filepath)

    def load_checkpoint_file(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        return pl_load(checkpoint_path, map_location=(lambda storage, loc: storage))

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        self.lightning_module.load_state_dict(checkpoint["state_dict"])

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        optimizer_states = checkpoint["optimizer_states"]
        for optimizer, opt_state in zip(self.lightning_module.trainer.accelerator.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)

    def lightning_module_state_dict(self) -> Dict[str, Union[Any, Tensor]]:
        """Returns model state."""
        model = self.lightning_module
        return model.state_dict()
