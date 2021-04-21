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
import torch

from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.warnings import WarningCache


class PredictLoop(object):

    def __init__(self, trainer):
        self.trainer = trainer
        self.max_batches = None
        self.num_dataloaders = None
        self.warning_cache = WarningCache()

    def on_trainer_init(self):
        self.trainer.num_predict_batches = []

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

    def setup(self, model, max_batches, dataloaders):

        # copy properties for forward overrides
        self.trainer.model_connector.copy_trainer_model_properties(model)

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        self.max_batches = max_batches
        self.num_dataloaders = self._get_num_dataloaders(dataloaders)
        self._predictions = [[] for _ in range(self.num_dataloaders)]

        if self.trainer._progress_bar_callback is not None:
            self.trainer._progress_bar_callback.on_predict_start(self.trainer, self.trainer.lightning_module)

    def _get_num_dataloaders(self, dataloaders):
        # case where user does:
        # return dl1, dl2
        length = len(dataloaders)
        if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
            length = len(dataloaders[0])
        return length

    def predict_step(self, batch, batch_idx, dataloader_idx):
        # configure args
        args = [batch, batch_idx]
        if self.num_dataloaders:
            args.append(dataloader_idx)

        model_ref = self.trainer.lightning_module

        model_ref._current_fx_name = "predict"
        predictions = self.trainer.accelerator.predict_step(args)

        if predictions is None:
            self.warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        self._predictions[dataloader_idx].append(predictions)

        if self.trainer._progress_bar_callback is not None:
            self.trainer._progress_bar_callback.on_predict_batch_end(
                self.trainer, model_ref, predictions, batch, batch_idx, dataloader_idx
            )
        return

    def on_predict_epoch_end(self):
        self.trainer.profiler.describe()

        if self.trainer._progress_bar_callback is not None:
            self.trainer._progress_bar_callback.on_predict_end(self.trainer, self.trainer.lightning_module)

        results = self._predictions

        def _convert_to_numpy(v):
            return v.cpu().numpy()

        results = apply_to_collection(results, torch.Tensor, _convert_to_numpy)

        if len(results) == 1:
            return results[0]

        return results

    def on_predict_start(self):
        # hook
        self.trainer.call_hook("on_predict_start")

    def on_predict_end(self):
        # hook
        self.trainer.call_hook("on_predict_end")
