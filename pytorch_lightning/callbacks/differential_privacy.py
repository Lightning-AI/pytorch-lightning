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

from opacus import PrivacyEngine
from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only


class DifferentialPrivacy(Callback):
    r"""
    Attach privacy engine to the optimizer before the training begins.
    """

    # take in privacy engine as arguement allow false and none and true
    # add to trainer
    def __init__(
        self, alphas=(1, 10, 100), noise_multiplier=0.1, max_grad_norm=0.1,
    ):
        super().__init__()
        self.alphas = alphas
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def on_train_start(self, trainer, pl_module):
        # VIRTUAL_BATCH_SIZE
        # tune max grad
        # check that channels divisible by 32
        # get_privacy_spent

        trainer.model = module_modification.convert_batchnorm_modules(pl_module)
        inspector = DPModelInspector()
        inspector.validate(trainer.model)

        privacy_engine = PrivacyEngine(
            pl_module,
            batch_size=trainer.train_dataloader.batch_size,
            sample_size=len(trainer.train_dataloader.dataset),
            alphas=self.alphas,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        for optimizer in pl_module.optimizers():
            privacy_engine.attach(optimizer)
