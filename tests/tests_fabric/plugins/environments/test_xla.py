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

import lightning.fabric
import pytest
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
