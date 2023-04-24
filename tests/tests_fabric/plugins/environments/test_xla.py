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
import torch

import lightning.fabric
from lightning.fabric.plugins.environments import XLAEnvironment
from tests_fabric.helpers.runif import RunIf


@RunIf(tpu=True)
# keep existing environment or else xla will default to pjrt
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_default_attributes(monkeypatch):
    """Test the default attributes when no environment variables are set."""
    from torch_xla.experimental import pjrt

    if pjrt.using_pjrt():
        # calling these creates side effects in other tests
        monkeypatch.setattr(pjrt, "world_size", lambda: 1)
        monkeypatch.setattr(pjrt, "global_ordinal", lambda: 0)
        monkeypatch.setattr(pjrt, "local_ordinal", lambda: 0)
    else:
        from torch_xla import _XLAC

        # avoid: "Cannot replicate if number of devices ... is different from ..."
        monkeypatch.setattr(_XLAC, "_xla_get_default_device", lambda: torch.device("xla:0"))

    env = XLAEnvironment()
    assert not env.creates_processes_externally
    assert env.world_size() == 1
    assert env.global_rank() == 0
    assert env.local_rank() == 0
    assert env.node_rank() == 0

    with pytest.raises(NotImplementedError):
        _ = env.main_address
    with pytest.raises(NotImplementedError):
        _ = env.main_port


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_attributes_from_environment_variables(monkeypatch):
    """Test that the default cluster environment takes the attributes from the environment variables."""
    from torch_xla.experimental import pjrt

    os.environ["XRT_HOST_ORDINAL"] = "3"
    if not pjrt.using_pjrt():
        os.environ.update(
            {
                "XRT_SHARD_WORLD_SIZE": "1",
                "XRT_SHARD_ORDINAL": "0",
                "XRT_SHARD_LOCAL_ORDINAL": "2",
            }
        )
    else:
        # PJRT doesn't pull these from envvars
        monkeypatch.setattr(pjrt, "world_size", lambda: 1)
        monkeypatch.setattr(pjrt, "global_ordinal", lambda: 0)
        monkeypatch.setattr(pjrt, "local_ordinal", lambda: 2)

    env = XLAEnvironment()
    with pytest.raises(NotImplementedError):
        _ = env.main_address
    with pytest.raises(NotImplementedError):
        _ = env.main_port
    assert env.world_size() == 1
    assert env.global_rank() == 0
    assert env.local_rank() == 2
    assert env.node_rank() == 3
    env.set_global_rank(100)
    assert env.global_rank() == 0
    env.set_world_size(100)
    assert env.world_size() == 1


def test_detect(monkeypatch):
    """Test the detection of a xla environment configuration."""
    monkeypatch.setattr(lightning.fabric.accelerators.tpu.TPUAccelerator, "is_available", lambda: False)
    assert not XLAEnvironment.detect()

    monkeypatch.setattr(lightning.fabric.accelerators.tpu.TPUAccelerator, "is_available", lambda: True)
    assert XLAEnvironment.detect()
