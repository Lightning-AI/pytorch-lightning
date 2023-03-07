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

import torch

from lightning.fabric.accelerators.cuda import _clear_cuda_memory
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12


def is_state_dict_equal(state0, state1):
    eq_fn = torch.equal if _TORCH_GREATER_EQUAL_1_12 else torch.allclose
    return all(eq_fn(w0.cpu(), w1.cpu()) for w0, w1 in zip(state0.values(), state1.values()))


def is_timing_close(timings_torch, timings_fabric, rtol=1e-3, atol=1e-3):
    # Drop measurements of the first iterations, as they may be slower than others
    # The median is more robust to outliers than the mean
    # Given relative and absolute tolerances, we want to satisfy: |torch – fabric| < RTOL * torch + ATOL
    return bool(torch.isclose(torch.median(timings_torch[3:]), torch.median(timings_fabric[3:]), rtol=rtol, atol=atol))


def is_cuda_memory_close(memory_stats_torch, memory_stats_fabric):
    # We require Fabric's peak memory usage to be smaller or equal to that of PyTorch
    return memory_stats_torch["allocated_bytes.all.peak"] >= memory_stats_fabric["allocated_bytes.all.peak"]


def make_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


def get_model_input_dtype(precision):
    if precision in ("16-mixed", "16", 16):
        return torch.float16
    elif precision in ("bf16-mixed", "bf16"):
        return torch.bfloat16
    elif precision in ("64-true", "64", 64):
        return torch.double
    return torch.float32


def cuda_reset():
    if torch.cuda.is_available():
        _clear_cuda_memory()
        torch.cuda.reset_peak_memory_stats()
