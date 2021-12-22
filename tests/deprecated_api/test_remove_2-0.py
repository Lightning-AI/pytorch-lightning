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
"""Test deprecated functionality which will be removed in v2.0.0."""
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from tests.helpers.runif import RunIf


def test_v2_0_0_deprecated_num_processes(tmpdir):
    with pytest.deprecated_call(match=r"is deprecated in v1.6 and will be removed in v2.0."):
        _ = Trainer(default_root_dir=tmpdir, num_processes=2)


@mock.patch("torch.cuda.is_available", return_value=True)
def test_v2_0_0_deprecated_gpus(tmpdir):
    with pytest.deprecated_call(match=r"is deprecated in v1.6 and will be removed in v2.0."):
        _ = Trainer(default_root_dir=tmpdir, gpus=2)


def test_v2_0_0_deprecated_tpu_cores(tmpdir):
    with pytest.deprecated_call(match=r"is deprecated in v1.6 and will be removed in v2.0."):
        _ = Trainer(default_root_dir=tmpdir, tpu_cores=1)


@RunIf(ipu=True)
def test_v2_0_0_deprecated_ipus(tmpdir):
    with pytest.deprecated_call(match=r"is deprecated in v1.6 and will be removed in v2.0."):
        _ = Trainer(default_root_dir=tmpdir, ipus=2)
