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
import os
from unittest import mock

import pytest
import torch

from lightning_lite.strategies import DDPStrategy


@pytest.mark.parametrize(
    ["process_group_backend", "env_var", "device_str", "expected_process_group_backend"],
    [
        pytest.param("foo", None, "cpu", "foo"),
        pytest.param("foo", "BAR", "cpu", "foo"),
        pytest.param("foo", "BAR", "cuda:0", "foo"),
        pytest.param(None, "BAR", "cuda:0", "BAR"),
        pytest.param(None, None, "cuda:0", "nccl"),
        pytest.param(None, None, "cpu", "gloo"),
    ],
)
def test_ddp_process_group_backend(process_group_backend, env_var, device_str, expected_process_group_backend):
    """Test settings for process group backend."""

    class MockDDPStrategy(DDPStrategy):
        def __init__(self, root_device, process_group_backend):
            self._root_device = root_device
            super().__init__(process_group_backend=process_group_backend)

        @property
        def root_device(self):
            return self._root_device

    strategy = MockDDPStrategy(process_group_backend=process_group_backend, root_device=torch.device(device_str))
    if not process_group_backend and env_var:
        with mock.patch.dict(os.environ, {"PL_TORCH_DISTRIBUTED_BACKEND": env_var}):
            with pytest.deprecated_call(
                match="Environment variable `PL_TORCH_DISTRIBUTED_BACKEND` was deprecated in v1.6"
            ):
                assert strategy._get_process_group_backend() == expected_process_group_backend
    else:
        assert strategy._get_process_group_backend() == expected_process_group_backend
