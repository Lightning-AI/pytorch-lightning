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
import time

import pytest

import pytorch_lightning.utilities.xla_device_utils as xla_utils
from pytorch_lightning.utilities import XLA_AVAILABLE, TPU_AVAILABLE
from tests.base.develop_utils import pl_multi_process_test

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


# lets hope that in or env we have installed XLA only for TPU devices, otherwise, it is testing in the cycle "if I am true test that I am true :D"
@pytest.mark.skipif(XLA_AVAILABLE, reason="test requires torch_xla to be absent")
def test_tpu_device_absence():
    """Check tpu_device_exists returns None when torch_xla is not available"""
    assert xla_utils.XLADeviceUtils.tpu_device_exists() is None


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires torch_xla to be installed")
@pl_multi_process_test
def test_tpu_device_presence():
    """Check tpu_device_exists returns True when TPU is available"""
    assert xla_utils.XLADeviceUtils.tpu_device_exists() is True


def test_result_returns_within_20_seconds():
    """Check that pl_multi_process returns within 10 seconds"""

    start = time.time()
    result = xla_utils.pl_multi_process(time.sleep)(25)
    end = time.time()
    elapsed_time = int(end - start)
    assert elapsed_time <= 20
    assert result is False
