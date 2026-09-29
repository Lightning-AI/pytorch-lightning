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
import os
from unittest import mock

import pytest

import lightning.fabric
from lightning.fabric.accelerators.xla import _XLA_GREATER_EQUAL_2_1
from lightning.fabric.plugins.environments import XLAEnvironment
from tests_fabric.helpers.runif import RunIf


@RunIf(tpu=True)
# keep existing environment or else xla will default to pjrt
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_default_attributes(monkeypatch):
    """Test the default attributes when no environment variables are set."""
    # calling these creates side effects in other tests
    if _XLA_GREATER_EQUAL_2_1:
        from torch_xla import runtime

        monkeypatch.setattr(runtime, "world_size", lambda: 2)
        monkeypatch.setattr(runtime, "global_ordinal", lambda: 0)
        monkeypatch.setattr(runtime, "local_ordinal", lambda: 0)
        monkeypatch.setattr(runtime, "host_index", lambda: 1)
    else:
        from torch_xla.experimental import pjrt

        monkeypatch.setattr(pjrt, "world_size", lambda: 2)
        monkeypatch.setattr(pjrt, "global_ordinal", lambda: 0)
        monkeypatch.setattr(pjrt, "local_ordinal", lambda: 0)
        os.environ["XRT_HOST_ORDINAL"] = "1"

    env = XLAEnvironment()
    assert not env.creates_processes_externally
    assert env.world_size() == 2
    assert env.global_rank() == 0
    assert env.local_rank() == 0
    assert env.node_rank() == 1

    with pytest.raises(NotImplementedError):
        _ = env.main_address
    with pytest.raises(NotImplementedError):
        _ = env.main_port


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_attributes_from_environment_variables(monkeypatch):
    """Test that the default cluster environment takes the attributes from the environment variables."""
    if _XLA_GREATER_EQUAL_2_1:
        from torch_xla import runtime

        monkeypatch.setattr(runtime, "world_size", lambda: 2)
        monkeypatch.setattr(runtime, "global_ordinal", lambda: 0)
        monkeypatch.setattr(runtime, "local_ordinal", lambda: 2)
        monkeypatch.setattr(runtime, "host_index", lambda: 1)
    else:
        from torch_xla.experimental import pjrt

        monkeypatch.setattr(pjrt, "world_size", lambda: 2)
        monkeypatch.setattr(pjrt, "global_ordinal", lambda: 0)
        monkeypatch.setattr(pjrt, "local_ordinal", lambda: 2)
        os.environ["XRT_HOST_ORDINAL"] = "1"

    env = XLAEnvironment()
    with pytest.raises(NotImplementedError):
        _ = env.main_address
    with pytest.raises(NotImplementedError):
        _ = env.main_port
    assert env.world_size() == 2
    assert env.global_rank() == 0
    assert env.local_rank() == 2
    assert env.node_rank() == 1
    env.set_global_rank(100)
    assert env.global_rank() == 0
    env.set_world_size(100)
    assert env.world_size() == 2


def test_detect(monkeypatch):
    """Test the detection of a xla environment configuration."""
    monkeypatch.setattr(lightning.fabric.accelerators.xla.XLAAccelerator, "is_available", lambda: False)
    assert not XLAEnvironment.detect()

    monkeypatch.setattr(lightning.fabric.accelerators.xla.XLAAccelerator, "is_available", lambda: True)
    assert XLAEnvironment.detect()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("lightning.fabric.accelerators.xla._XLA_GREATER_EQUAL_2_1", True)
@mock.patch("lightning.fabric.plugins.environments.xla._XLA_GREATER_EQUAL_2_1", True)
def test_world_size_from_xla_runtime_greater_2_1(xla_available):
    """Test that world_size uses torch_xla.runtime when XLA >= 2.1."""
    env = XLAEnvironment()

    with mock.patch("torch_xla.runtime.world_size", return_value=4) as mock_world_size:
        env.world_size.cache_clear()
        assert env.world_size() == 4
        mock_world_size.assert_called_once()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("lightning.fabric.accelerators.xla._XLA_GREATER_EQUAL_2_1", True)
@mock.patch("lightning.fabric.plugins.environments.xla._XLA_GREATER_EQUAL_2_1", True)
def test_global_rank_from_xla_runtime_greater_2_1(xla_available):
    """Test that global_rank uses torch_xla.runtime when XLA >= 2.1."""
    env = XLAEnvironment()

    with mock.patch("torch_xla.runtime.global_ordinal", return_value=2) as mock_global_ordinal:
        env.global_rank.cache_clear()
        assert env.global_rank() == 2
        mock_global_ordinal.assert_called_once()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("lightning.fabric.accelerators.xla._XLA_GREATER_EQUAL_2_1", True)
@mock.patch("lightning.fabric.plugins.environments.xla._XLA_GREATER_EQUAL_2_1", True)
def test_local_rank_from_xla_runtime_greater_2_1(xla_available):
    """Test that local_rank uses torch_xla.runtime when XLA >= 2.1."""
    env = XLAEnvironment()

    with mock.patch("torch_xla.runtime.local_ordinal", return_value=1) as mock_local_ordinal:
        env.local_rank.cache_clear()
        assert env.local_rank() == 1
        mock_local_ordinal.assert_called_once()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("lightning.fabric.accelerators.xla._XLA_GREATER_EQUAL_2_1", True)
@mock.patch("lightning.fabric.plugins.environments.xla._XLA_GREATER_EQUAL_2_1", True)
def test_setters_readonly_when_xla_runtime_greater_2_1(xla_available):
    """Test that set_world_size and set_global_rank don't affect values when using XLA runtime >= 2.1."""
    env = XLAEnvironment()

    with (
        mock.patch("torch_xla.runtime.world_size", return_value=4),
        mock.patch("torch_xla.runtime.global_ordinal", return_value=2),
    ):
        env.world_size.cache_clear()
        env.global_rank.cache_clear()

        # Values should come from XLA runtime and not be affected by setters
        env.set_world_size(100)
        assert env.world_size() == 4

        env.set_global_rank(100)
        assert env.global_rank() == 2
