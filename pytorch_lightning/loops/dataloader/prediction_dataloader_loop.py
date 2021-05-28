from typing import Any, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.loops.dataloader.dataloader_loop import DataLoaderLoop
from pytorch_lightning.loops.prediction_loop import PredictionLoop
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT


class PredictionDataLoaderLoop(DataLoaderLoop):

    def __init__(self):
        super().__init__()
        self.prediction_loop = PredictionLoop()
        self._return_predictions = False
        self.predictions = None
        self.epoch_batch_indices = None
        self._dataloaders = None
        self._max_batches = None

    @property
    def return_predictions(self) -> bool:
        return self._return_predictions

    @return_predictions.setter
    def return_predictions(self, return_predictions: Optional[bool] = None) -> None:
        # ``DDPSpawnPlugin`` plugins and derivate don't support return predictions.
        is_ddp_spawn = isinstance(self.trainer.training_type_plugin, DDPSpawnPlugin)
        if return_predictions and is_ddp_spawn:
            raise MisconfigurationException(
                "`return_predictions` should be set to `False` when using the `DDPSpawnPlugin` or children class. "
                f"Found {return_predictions} with training_type_plugin {type(self.trainer.training_type_plugin)}."
            )
        # For non ``DDPSpawnPlugin`` plugin, the `return_predictions` is True by default unless user decide otherwise.
        self._return_predictions = not is_ddp_spawn if return_predictions is None else return_predictions

    @property
    def num_dataloaders(self) -> int:
        return self._get_num_dataloaders(self.dataloaders)

    @property
    def dataloaders(self) -> Sequence[DataLoader]:
        return self._dataloaders

    @property
    def done(self) -> bool:
        return (self.current_dataloader_idx >= len(self.dataloaders)) or self.should_skip_predict(self._max_batches)

    def connect(self, trainer, *args, **kwargs) -> None:
        super().connect(trainer, *args, **kwargs)
        self.prediction_loop.connect(trainer, *args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._dataloaders, self._max_batches = self.get_predict_dataloaders()

        # convert max_batches to list
        if isinstance(self._max_batches, int):
            self._max_batches = [self._max_batches] * len(self.dataloaders)

        self.predictions = []
        self.epoch_batch_indices = []

    def on_run_start(self) -> None:
        self.on_predict_start()

    def advance(self, *args, **kwargs) -> None:
        dataloader = self.trainer.accelerator.process_dataloader(self.current_dataloader)
        dataloader_iter = enumerate(dataloader)
        dl_max_batches = self._max_batches[self.current_dataloader_idx]

        dl_predictions, dl_batch_indices = self.prediction_loop.run(
            dataloader_iter, self.current_dataloader_idx, dl_max_batches, self.num_dataloaders, self.return_predictions
        )
        self.predictions.append(dl_predictions)
        self.epoch_batch_indices.append(dl_batch_indices)

    def on_run_end(self):
        results = self.on_predict_epoch_end()
        self.on_predict_end()
        return results


# ------------------------------------------------------------------------------------------------------------
# HELPER --- TO BE CLEANED UP
# ------------------------------------------------------------------------------------------------------------

    def get_predict_dataloaders(self):
        self.trainer.reset_predict_dataloader(self.trainer.lightning_module)

        dataloaders = self.trainer.predict_dataloaders
        max_batches = self.trainer.num_predict_batches

        return dataloaders, max_batches

    def should_skip_predict(self, max_batches):
        return sum(max_batches) == 0

    def on_predict_start(self) -> None:
        # enable eval mode + no grads
        self.on_predict_model_eval()
        self.trainer.lightning_module.zero_grad()
        self._previous_grad_status = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        # hook
        self.trainer.call_hook("on_predict_start")
        self.trainer.call_hook("on_predict_epoch_start")

    def on_predict_epoch_end(self) -> Optional[_PREDICT_OUTPUT]:
        self.trainer.profiler.describe()

        results = self.predictions

        self.trainer.call_hook("on_predict_epoch_end", results)

        if self.return_predictions:
            return results[0] if self.num_dataloaders == 1 else results

    def on_predict_end(self):
        # clear memory. the predictions are extracted in `on_predict_epoch_end`.
        self.predictions = []
        self.epoch_batch_indices = []

        # reset grad to its previous status.
        torch.set_grad_enabled(self._previous_grad_status)

        # hook
        self.trainer.call_hook("on_predict_end")

    def on_predict_model_eval(self):
        model_ref = self.trainer.lightning_module
        model_ref.on_predict_model_eval()

    def _get_num_dataloaders(self, dataloaders: List[DataLoader]) -> int:
        # case where user does:
        # return dl1, dl2
        length = len(dataloaders)
        if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
            length = len(dataloaders[0])
        return length
