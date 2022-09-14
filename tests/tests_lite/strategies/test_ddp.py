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
import pytest
import torch

from lightning_lite.strategies import DDPStrategy


@pytest.mark.parametrize(
    ["process_group_backend", "device_str", "expected_process_group_backend"],
    [
        pytest.param("foo", "cpu", "foo"),
        pytest.param("foo", "cuda:0", "foo"),
        pytest.param(None, "cuda:0", "nccl"),
        pytest.param(None, "cpu", "gloo"),
    ],
)
def test_ddp_process_group_backend(process_group_backend, device_str, expected_process_group_backend):
    """Test settings for process group backend."""

    class MockDDPStrategy(DDPStrategy):
        def __init__(self, root_device, process_group_backend):
            self._root_device = root_device
            super().__init__(process_group_backend=process_group_backend)

        @property
        def root_device(self):
            return self._root_device

    strategy = MockDDPStrategy(process_group_backend=process_group_backend, root_device=torch.device(device_str))
    assert strategy._get_process_group_backend() == expected_process_group_backend
