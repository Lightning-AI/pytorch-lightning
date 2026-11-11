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

import pytest
import torch

from lightning.pytorch.utilities.memory import (
    is_cuda_out_of_memory,
    is_cudnn_snafu,
    is_oom_error,
    is_out_of_cpu_memory,
    recursive_detach,
)


def test_recursive_detach():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = {"foo": torch.tensor(0, device=device), "bar": {"baz": torch.tensor(1.0, device=device, requires_grad=True)}}
    y = recursive_detach(x, to_cpu=True)

    assert x["foo"].device.type == device
    assert x["bar"]["baz"].device.type == device
    assert x["bar"]["baz"].requires_grad

    assert y["foo"].device.type == "cpu"
    assert y["bar"]["baz"].device.type == "cpu"
    assert not y["bar"]["baz"].requires_grad


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError(1),
        RuntimeError(None),
        RuntimeError((1, "CUDA out of memory")),
        RuntimeError(),
        ValueError(1),
    ],
)
def test_is_oom_error_non_string_runtime_error_arg(exception):
    assert not is_cuda_out_of_memory(exception)
    assert not is_cudnn_snafu(exception)
    assert not is_out_of_cpu_memory(exception)
    assert not is_oom_error(exception)

