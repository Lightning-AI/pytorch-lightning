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
Differential Privacy
====================

Train your model with differential privacy using Opacus(https://github.com/pytorch/opacus).

"""

import os
import re
from typing import Optional

import numpy as np
import torch
from opacus import PrivacyEngine

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn


class DifferentialPrivacy(Callback):
    r"""
    Attach privacy engine to the optimizer before the training begins.
    """

    # take in privacy engine as arguement allow false and none and true
    # add to trainer
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        privacy_engine = PrivacyEngine(
            pl_module,
            batch_size=trainer.train_dataloader.batch_size,
            sample_size=len(trainer.train_dataloader.dataset),
            alphas=[1, 10, 100],
            noise_multiplier=1.3,
            max_grad_norm=1.0,
        )

        for optimizer in trainer.optimizers:
            privacy_engine.attach(optimizer)
# VIRTUAL_BATCH_SIZE
# take a real optimizer step after N_VIRTUAL_STEP steps t
#validate model
# from opacus.dp_model_inspector import DPModelInspector
#
# inspector = DPModelInspector()
# inspector.validate(model)

# from opacus.utils import module_modification
#
# model = module_modification.convert_batchnorm_modules(model)
# inspector = DPModelInspector()
# print(f"Is the model valid? {inspector.validate(model)}")
# epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(DELTA)