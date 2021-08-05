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
from unittest.mock import patch

import pytest

import pytorch_lightning.utilities.xla_device as xla_utils
from pytorch_lightning.utilities import _XLA_AVAILABLE
from tests.helpers.runif import RunIf


@pytest.mark.skipif(_XLA_AVAILABLE, reason="test requires torch_xla to be absent")
def test_tpu_device_absence():
    """Check tpu_device_exists returns False when torch_xla is not available"""
    assert not xla_utils.XLADeviceUtils.tpu_device_exists()


@RunIf(tpu=True)
def test_tpu_device_presence():
    """Check tpu_device_exists returns True when TPU is available"""
    assert xla_utils.XLADeviceUtils.tpu_device_exists()


def sleep_fn(sleep_time: float) -> bool:
    time.sleep(sleep_time)
    return True


@patch("pytorch_lightning.utilities.xla_device.TPU_CHECK_TIMEOUT", 3)
@pytest.mark.skipif(not _XLA_AVAILABLE, reason="test requires torch_xla to be present")
def test_result_returns_within_timeout_seconds():
    """Check that pl_multi_process returns within 3 seconds"""
    fn = xla_utils.pl_multi_process(sleep_fn)

    start = time.time()
    result = fn(xla_utils.TPU_CHECK_TIMEOUT * 0.5)
    end = time.time()
    elapsed_time = int(end - start)

    assert elapsed_time <= xla_utils.TPU_CHECK_TIMEOUT
    assert result
