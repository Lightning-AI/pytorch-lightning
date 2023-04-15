# Copyright The Lightning AI team.
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

from lightning.fabric.accelerators.tpu import _XLA_AVAILABLE, TPUAccelerator
from tests_fabric.helpers.runif import RunIf


@pytest.mark.skipif(_XLA_AVAILABLE, reason="test requires torch_xla to be absent")
def test_tpu_device_absence():
    """Check `is_available` returns True when TPU is available."""
    assert not TPUAccelerator.is_available()


@RunIf(tpu=True)
def test_tpu_device_presence():
    """Check `is_available` returns True when TPU is available."""
    assert TPUAccelerator.is_available()


def test_result_returns_within_timeout_seconds(monkeypatch):
    """Check that the TPU availability process launch returns within 3 seconds."""
    from lightning.fabric.accelerators import tpu

    timeout = 3
    monkeypatch.setattr(tpu, "_XLA_AVAILABLE", True)
    monkeypatch.setattr(tpu, "TPU_CHECK_TIMEOUT", timeout)
    monkeypatch.setattr(tpu, "_has_tpu_device", lambda: time.sleep(1.5) or True)
    tpu.TPUAccelerator.is_available.cache_clear()

    start = time.monotonic()

    result = tpu.TPUAccelerator.is_available()

    end = time.monotonic()
    elapsed_time = end - start

    # around 1.5 but definitely not 3 (timeout time)
    assert 1 < elapsed_time < 2, elapsed_time
    assert result

    tpu.TPUAccelerator.is_available.cache_clear()


def test_timeout_triggered(monkeypatch):
    """Check that the TPU availability process launch returns within 3 seconds."""
    from lightning.fabric.accelerators import tpu

    timeout = 1.5
    monkeypatch.setattr(tpu, "_XLA_AVAILABLE", True)
    monkeypatch.setattr(tpu, "TPU_CHECK_TIMEOUT", timeout)
    monkeypatch.setattr(tpu, "_has_tpu_device", lambda: time.sleep(3) or True)
    tpu.TPUAccelerator.is_available.cache_clear()

    start = time.monotonic()

    with pytest.raises(TimeoutError, match="Timed out waiting"):
        tpu.TPUAccelerator.is_available()

    end = time.monotonic()
    elapsed_time = end - start

    # around 1.5 but definitely not 3 (fn time)
    assert 1 < elapsed_time < 2, elapsed_time

    tpu.TPUAccelerator.is_available.cache_clear()
