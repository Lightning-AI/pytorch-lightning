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

import pytest
import torch.distributed


@pytest.fixture()
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)


@pytest.fixture()
def reset_cudnn_benchmark():
    """Ensures that the `torch.backends.cudnn.benchmark` setting gets reset before the next test runs."""
    yield
    torch.backends.cudnn.benchmark = False
