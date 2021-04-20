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
from typing import Any, List, Optional

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class PredictionWriter(Callback):

    write_intervals = ("step", "epoch")

    def __init__(self, output_dir: Optional[str], write_interval: Optional[str] = "step"):
        if write_interval not in self.write_intervals:
            raise MisconfigurationException(
                f"write_interval should be within {self.write_intervals}. Found {write_interval}"
            )
        self._output_dir = output_dir
        self._write_interval = write_interval

    def on_predict_batch_end(
        self, trainer, pl_module: 'LightningModule', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if self._write_interval == "step":
            pass
            #torch.save({"batch":batch, "indices": self.trainer.predict_loop.batch_indices}, "output_dir/preds_rank_batch_idx_{dl_idx}.pt"

    def on_predict_epoch_end(self, trainer, pl_module: 'LightningModule', outputs: List[Any]) -> None:
        if self._write_interval == "epoch":
            pass
            # torch.save({"batch":batch, "indices": self.trainer.predict_loop.batch_indices}, "output_dir/preds_rank_batch_idx_{dl_idx}.pt"
