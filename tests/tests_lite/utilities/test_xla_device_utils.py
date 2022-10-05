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
from tests_lite.helpers.runif import RunIf

from lightning_lite.accelerators.tpu import _multi_process, _XLA_AVAILABLE, TPUAccelerator


@pytest.mark.skipif(_XLA_AVAILABLE, reason="test requires torch_xla to be absent")
def test_tpu_device_absence():
    """Check `is_available` returns True when TPU is available."""
    assert not TPUAccelerator.is_available()


@RunIf(tpu=True)
def test_tpu_device_presence():
    """Check `is_available` returns True when TPU is available."""
    assert TPUAccelerator.is_available()


def sleep_fn(sleep_time: float) -> bool:
    time.sleep(sleep_time)
    return True


@patch("lightning_lite.accelerators.tpu.TPU_CHECK_TIMEOUT", 3)
@pytest.mark.skipif(not _XLA_AVAILABLE, reason="test requires torch_xla to be present")
def test_result_returns_within_timeout_seconds():
    """Check that pl_multi_process returns within 3 seconds."""
    fn = _multi_process(sleep_fn)

    start = time.time()
    from lightning_lite.accelerators.tpu import TPU_CHECK_TIMEOUT

    result = fn(TPU_CHECK_TIMEOUT * 0.5)
    end = time.time()
    elapsed_time = int(end - start)

    assert elapsed_time <= TPU_CHECK_TIMEOUT
    assert result
