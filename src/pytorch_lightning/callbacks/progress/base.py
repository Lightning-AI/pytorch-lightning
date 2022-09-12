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
from typing import Dict, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress import get_standard_metrics as new_get_standard_metrics
from pytorch_lightning.callbacks.progress.progress import ProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class ProgressBarBase(ProgressBar):
    r"""
    The base class for progress bars in Lightning. It is a :class:`~pytorch_lightning.callbacks.Callback`
    that keeps track of the batch progress in the :class:`~pytorch_lightning.trainer.trainer.Trainer`.
    You should implement your highly custom progress bars with this as the base class.

    Example::

        class LitProgressBar(ProgressBar):

            def __init__(self):
                super().__init__()  # don't forget this :)
                self.enable = True

            def disable(self):
                self.enable = False

            def on_train_batch_end(self, trainer, pl_module, outputs, batch_idx):
                super().on_train_batch_end(trainer, pl_module, outputs, batch_idx)  # don't forget this :)
                percent = (self.train_batch_idx / self.total_train_batches) * 100
                sys.stdout.flush()
                sys.stdout.write(f'{percent:.01f} percent complete \r')

        bar = LitProgressBar()
        trainer = Trainer(callbacks=[bar])

    .. deprecated:: v1.8.0

        `pytorch_lightning.callbacks.progress.base.ProgressBarBase` was deprecated in v1.8.0 and will be removed
        in v1.10.0. Use the equivalent `pytorch_lightning.callbacks.progress.progress.ProgressBar` instead.
    """

    def __init__(self) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.callbacks.progress.base.ProgressBarBase` was deprecated in v1.8.0 and will be removed"
            " in v1.10.0. Use the equivalent `pytorch_lightning.callbacks.progress.progress.ProgressBar` instead."
        )
        super().__init__()


def get_standard_metrics(trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
    r"""
    Returns several standard metrics displayed in the progress bar, including the average loss value,
    split index of BPTT (if used) and the version of the experiment when using a logger.

    .. code-block::

        Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, loss=4.501, v_num=10]

    Return:
        Dictionary with the standard metrics to be displayed in the progress bar.

    .. deprecated:: v1.8.0

        `pytorch_lightning.callbacks.progress.base.get_standard_metrics` was deprecated in v1.8.0 and will be removed
        in v1.10.0. Use the equivalent `pytorch_lightning.callbacks.progress.progress.get_standard_metrics` instead.
    """
    rank_zero_deprecation(
        "`pytorch_lightning.callbacks.progress.base.get_standard_metrics` was deprecated in v1.8.0 and will be removed"
        " in v1.10.0. Use the equivalent `pytorch_lightning.callbacks.progress.progress.get_standard_metrics` instead."
    )
    return new_get_standard_metrics(trainer, pl_module)
