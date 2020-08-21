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

from abc import ABC

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import APEX_AVAILABLE, NATIVE_AMP_AVALAIBLE, rank_zero_warn, AMPType


class TrainerAMPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    precision: int

    def _setup_amp_backend(self, amp_type: str):
        if self.precision != 16:
            # no AMP requested, so we can leave now
            return
        amp_type = amp_type.lower()
        assert amp_type in ('native', 'apex'), f'Unsupported amp type {amp_type}'
        if amp_type == 'native':
            if not NATIVE_AMP_AVALAIBLE:
                rank_zero_warn('You have asked for native AMP but your PyTorch version does not support it.'
                               ' Consider upgrading with `pip install torch>=1.6`.'
                               ' We will attempt to use NVIDIA Apex for this session.')
                amp_type = 'apex'
            else:
                log.info('Using native 16bit precision.')
                self.amp_backend = AMPType.NATIVE
        if amp_type == 'apex':
            if not APEX_AVAILABLE:
                rank_zero_warn('You have asked for Apex AMP but you have not installed it yet.'
                               ' Install apex first using this guide: https://github.com/NVIDIA/apex#linux')
            else:
                log.info('Using APEX 16bit precision.')
                self.amp_backend = AMPType.APEX
        if not self.amp_backend:
            raise ModuleNotFoundError(
                f'You have asked for AMP support {amp_type}, but there is no support on your side yet.'
                f' Consider installing torch >= 1.6 or NVIDIA Apex.'
            )

    @property
    def use_amp(self) -> bool:
        return self.precision == 16
