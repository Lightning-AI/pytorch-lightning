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

from pytorch_lightning.utilities.rank_zero import _get_rank


@pytest.mark.parametrize(
    "environ,expected_rank",
    [
        ({"JSM_NAMESPACE_RANK": "3"}, 3),
        ({"JSM_NAMESPACE_RANK": "3", "SLURM_PROCID": "2"}, 2),
        ({"JSM_NAMESPACE_RANK": "3", "SLURM_PROCID": "2", "LOCAL_RANK": "1"}, 1),
        ({"JSM_NAMESPACE_RANK": "3", "SLURM_PROCID": "2", "LOCAL_RANK": "1", "RANK": "0"}, 0),
    ],
)
def test_rank_zero_priority(environ, expected_rank):
    """Test the priority in which the rank gets determined when multiple environment variables are available."""
    with mock.patch.dict(os.environ, environ):
        assert _get_rank() == expected_rank
