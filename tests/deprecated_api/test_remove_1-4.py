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
"""Test deprecated functionality which will be removed in vX.Y.Z"""
import pytest
import torch

from pytorch_lightning import Trainer
from tests.deprecated_api import _soft_unimport_module


def test_v1_4_0_deprecated_imports():
    _soft_unimport_module('pytorch_lightning.utilities.argparse_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.argparse_utils import from_argparse_args  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.model_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.model_utils import is_overridden  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.warning_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.warning_utils import WarningCache  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.xla_device_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils  # noqa: F811 F401


# todo: later add also checking deprecated warnings
def test_v1_4_0_deprecated_trainer_attributes():
    """Test that Trainer attributes works fine."""
    trainer = Trainer()
    trainer._distrib_type = None
    trainer._device_type = None

    trainer.on_cpu = True
    assert trainer.on_cpu

    trainer.on_gpu = True
    assert trainer.on_gpu

    trainer.on_tpu = True
    assert trainer.on_tpu
    trainer._device_type = None
    trainer.use_tpu = True
    assert trainer.use_tpu

    trainer.use_dp = True
    assert trainer.use_dp

    trainer.use_ddp = True
    assert trainer.use_ddp

    trainer.use_ddp2 = True
    assert trainer.use_ddp2

    trainer.use_horovod = True
    assert trainer.use_horovod


def test_v1_3_0_deprecated_metrics():
    from pytorch_lightning.metrics.functional.classification import stat_scores_multiple_classes
    with pytest.deprecated_call(match='will be removed in v1.3'):
        stat_scores_multiple_classes(pred=torch.tensor([0, 1]), target=torch.tensor([0, 1]))
