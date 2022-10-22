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

import torch
import torch.distributed

from pytorch_lightning.utilities.distributed import _collect_states_on_rank_zero
from tests_pytorch.core.test_results import spawn_launch
from tests_pytorch.helpers.runif import RunIf


def collect_states_fn(strategy):
    rank = strategy.local_rank
    state = {"something": torch.tensor([rank])}
    collected_state = _collect_states_on_rank_zero(state)
    assert collected_state == {1: {"something": torch.tensor([1])}, 0: {"something": torch.tensor([0])}}


@RunIf(min_cuda_gpus=2, min_torch="1.10", skip_windows=True)
def test_collect_states():
    """This test ensures state are properly collected across processes.

    This would be used to collect dataloader states as an example.
    """
    spawn_launch(collect_states_fn, [torch.device("cuda:0"), torch.device("cuda:1")])
