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
import multiprocessing

import torch
from lightning.pytorch.plugins import MixedPrecision

from tests_pytorch.helpers.runif import RunIf


# needs to be standalone to avoid other processes initializing CUDA
@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True)
def test_amp_gpus_ddp_fork():
    """Ensure the use of AMP with `ddp_fork` (or associated alias strategies) does not generate CUDA initialization
    errors."""
    _ = MixedPrecision(precision="16-mixed", device="cuda")
    with multiprocessing.get_context("fork").Pool(1) as pool:
        in_bad_fork = pool.apply(torch.cuda._is_in_bad_fork)
    assert not in_bad_fork
