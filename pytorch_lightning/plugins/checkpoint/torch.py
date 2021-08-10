from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.plugins.checkpoint.checkpoint import CheckpointIOPlugin
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.cloud_io import load as pl_load


class TorchCheckpointIOPlugin(CheckpointIOPlugin):
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Mapping] = None
    ) -> None:
        try:
            # write the checkpoint dictionary on the file
            atomic_save(checkpoint, path)
        except AttributeError as err:
            key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
            checkpoint.pop(key, None)
            rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            atomic_save(checkpoint, path)

    def load_checkpoint(
        self, path: Union[str, Path], map_location: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        return pl_load(path, map_location=map_location)
