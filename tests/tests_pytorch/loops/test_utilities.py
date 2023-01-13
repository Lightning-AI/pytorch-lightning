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
from unittest.mock import Mock

from pytorch_lightning.loops.utilities import _set_sampler_epoch


def test_set_sampler_epoch():
    # No samplers
    dataloader = Mock()
    dataloader.sampler = None
    dataloader.batch_sampler = None
    _set_sampler_epoch(dataloader, 55)

    # set_epoch not callable
    dataloader = Mock()
    dataloader.sampler.set_epoch = None
    dataloader.batch_sampler.set_epoch = None
    _set_sampler_epoch(dataloader, 55)

    # set_epoch callable
    dataloader = Mock()
    _set_sampler_epoch(dataloader, 55)
    dataloader.sampler.set_epoch.assert_called_once_with(55)
    dataloader.batch_sampler.set_epoch.assert_called_once_with(55)
