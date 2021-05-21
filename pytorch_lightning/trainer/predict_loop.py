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
from collections import OrderedDict
from typing import Any, List, Optional

import torch
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache


class PredictLoop(object):

    def __init__(self, trainer):
        self.trainer = trainer
        self.max_batches = None
        self.num_dataloaders = None
        self.warning_cache = WarningCache()
        self.batch_indices: Optional[List[int]] = None
        self.epoch_batch_indices: Optional[List[List[int]]] = None
        self.predictions: Optional[List[List[Any]]] = None
        # `DDPSpawnPlugin` plugins and derivate don't support return predictions.
        self._return_predictions: Optional[bool] = None
        self._previous_grad_status: Optional[bool] = None

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
    def should_store_predictions(self) -> bool:
        any_pred = any(cb.interval.on_epoch for cb in self.trainer.prediction_writer_callbacks)
        return self.return_predictions or any_pred

    def on_trainer_init(self):
        self.trainer.num_predict_batches = []
        self.trainer.predicted_ckpt_path = None

    def get_predict_dataloaders(self):
        self.trainer.reset_predict_dataloader(self.trainer.lightning_module)

        dataloaders = self.trainer.predict_dataloaders
        max_batches = self.trainer.num_predict_batches

        return dataloaders, max_batches

    def should_skip_predict(self, max_batches):
        return sum(max_batches) == 0

    def on_predict_model_eval(self):
        model_ref = self.trainer.lightning_module
        model_ref.on_predict_model_eval()

    def setup(self, max_batches, dataloaders):
        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        self.max_batches = max_batches
        self.num_dataloaders = self._get_num_dataloaders(dataloaders)
        self.predictions = [[] for _ in range(self.num_dataloaders)]
        self.epoch_batch_indices = [[] for _ in range(self.num_dataloaders)]

    def _get_num_dataloaders(self, dataloaders: List[DataLoader]) -> int:
        # case where user does:
        # return dl1, dl2
        length = len(dataloaders)
        if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
            length = len(dataloaders[0])
        return length

    def _build_kwargs(self, batch, batch_idx, dataloader_idx):
        step_kwargs = OrderedDict([('batch', batch), ('batch_idx', batch_idx)])
        if self.num_dataloaders:
            step_kwargs['dataloader_idx'] = dataloader_idx
        return step_kwargs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        # extract batch_indices and store them
        self._store_batch_indices(dataloader_idx)

        model_ref = self.trainer.lightning_module

        self.trainer.call_hook("on_predict_batch_start", batch, batch_idx, dataloader_idx)

        model_ref._current_fx_name = "predict_step"
        predictions = self.trainer.accelerator.predict_step(step_kwargs)

        if predictions is None:
            self.warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        self.trainer.call_hook("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)

        if self.should_store_predictions:
            self.predictions[dataloader_idx].append(predictions)

    def _store_batch_indices(self, dataloader_idx: int) -> None:
        batch_sampler = self.trainer.predict_dataloaders[dataloader_idx].batch_sampler
        if isinstance(batch_sampler, IndexBatchSamplerWrapper):
            self.batch_indices = batch_sampler.batch_indices
            if self.should_store_predictions:
                self.epoch_batch_indices[dataloader_idx].append(batch_sampler.batch_indices)

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
        self.predictions = None
        self.batch_indices = None

        # reset grad to its previous status.
        torch.set_grad_enabled(self._previous_grad_status)

        # hook
        self.trainer.call_hook("on_predict_end")
