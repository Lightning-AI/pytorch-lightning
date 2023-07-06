from lightning.fabric.utilities.spike import SpikeDetection as FabricSpikeDetection
from lightning.pytorch.callbacks.callback import Callback

import lightning as L
from typing import Any, Literal, Mapping, Optional, Protocol, Union, runtime_checkable

class SpikeDetection(FabricSpikeDetection, Callback):
    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        batch: Any,
        batch_idx: int,
    ) -> None:
        if isinstance(outputs, torch.Tensor):
            loss = outputs.detach()
        elif isinstance(outputs, Mapping):
            loss = outputs["loss"].detach()
        else:
            raise TypeError(
                f"outputs have to be of type torch.Tensor or Mapping, got {type(outputs).__qualname__}"
            )

        if self.exclude_batches_path is None:
            self.exclude_batches_path = os.path.join(
                trainer.default_root_dir, "skip_batches.json"
            )

        return FabricSpikeDetection.on_train_batch_end(self, trainer, loss, batch, batch_idx)
