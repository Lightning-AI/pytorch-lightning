# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#

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
"""
hpu Stats Monitor
=================

Monitor and logs hpu stats during training.

"""
from typing import Any, Dict, List, Optional, Tuple

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only


class HPUStatsMonitor(Callback):
    """Automatically monitors and logs hpu stats during training stage.

    Args:
        save_dir: directory to save the logs.
        exp_name: name of the experiment.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import HPUStatsMonitor
        >>> hpu_stats = HPUStatsMonitor()
        >>> trainer = Trainer(hpus=1, callbacks=[hpu_stats])

    you can also optionally provide save_dir and exp_name in HPUStatsMonitor.
    No need to provide logger in Trainer.
    """

    def __init__(self, log_save_dir: str = "habana_ptl_logs", exp_name: str = "default"):
        super().__init__()
        self.log_save_dir = log_save_dir
        self.exp_name = exp_name

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        from pytorch_lightning import loggers as pl_logger

        self.tb_logger = pl_logger.TensorBoardLogger(save_dir=self.log_save_dir, name=self.exp_name)
        trainer.logger = self.tb_logger

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        pl_module.log("Model_Loss", loss, on_step=True, on_epoch=True, enable_graph=False, logger=True)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        tensor_board = trainer.logger.experiment
        dict = vars(pl_module)
        modules = dict["_modules"]
        for module_name in modules:
            tensor_board.add_histogram(module_name + ".weight", modules[module_name].weight, pl_module.current_epoch)
            tensor_board.add_histogram(module_name + ".bias", modules[module_name].bias, pl_module.current_epoch)
