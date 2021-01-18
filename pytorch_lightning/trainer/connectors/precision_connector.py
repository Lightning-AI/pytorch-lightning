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

import logging

from pytorch_lightning.plugins.apex import ApexPlugin
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from pytorch_lightning.utilities import AMPType, APEX_AVAILABLE, NATIVE_AMP_AVAILABLE, rank_zero_warn

log = logging.getLogger(__name__)


class PrecisionConnector:

    def __init__(self, trainer):
        self.trainer = trainer
        self.backend = None

    def on_trainer_init(self, precision: int, amp_level: str, amp_backend: str):
        # AMP init
        # These are the only lines needed after v0.8.0
        # we wrap the user's forward with autocast and give it back at the end of fit
        self.trainer.autocast_original_forward = None
        self.trainer.precision = precision
        self.trainer.scaler = None

        self.trainer.amp_level = amp_level
        self.init_amp(amp_backend)

    def init_amp(self, amp_type: str):
        assert self.trainer.precision in (16, 32), 'only 32 or 16 bit precision supported'
        self.trainer.amp_backend = None
        self._setup_amp_backend(amp_type)

    def _setup_amp_backend(self, amp_type: str):
        if self.trainer.precision != 16:
            # no AMP requested, so we can leave now
            return

        amp_type = amp_type.lower()
        assert amp_type in ('native', 'apex'), f'Unsupported amp type {amp_type}'
        if amp_type == 'native':
            if not NATIVE_AMP_AVAILABLE:
                rank_zero_warn('You have asked for native AMP but your PyTorch version does not support it.'
                               ' Consider upgrading with `pip install torch>=1.6`.'
                               ' We will attempt to use NVIDIA Apex for this session.')
                amp_type = 'apex'
            else:
                self.trainer.amp_backend = AMPType.NATIVE
                log.info('Using native 16bit precision.')
                self.backend = NativeAMPPlugin(self.trainer)

        if amp_type == 'apex':
            if not APEX_AVAILABLE:
                rank_zero_warn('You have asked for Apex AMP but you have not installed it yet.'
                               ' Install apex first using this guide: https://github.com/NVIDIA/apex#linux')
            else:
                log.info('Using APEX 16bit precision.')
                self.trainer.amp_backend = AMPType.APEX
                self.backend = ApexPlugin(self.trainer)
                log.warn("LightningOptimizer doesn't support Apex")

        if not self.trainer.amp_backend:
            raise ModuleNotFoundError(
                f'You have asked for AMP support {amp_type}, but there is no support on your side yet.'
                f' Consider installing torch >= 1.6 or NVIDIA Apex.'
            )

    def connect(self, model):
        if self.backend:
            model, optimizers = self.backend.connect(model, self.trainer.optimizers)
            self.trainer.optimizers = optimizers

        return model
