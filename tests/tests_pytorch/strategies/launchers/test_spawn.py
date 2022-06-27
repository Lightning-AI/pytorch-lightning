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
from unittest import mock
from unittest.mock import ANY, Mock

import pytest

import pytorch_lightning
from pytorch_lightning.strategies.launchers.spawn import _SpawnLauncher


def test_spawn_launcher_forking_on_unsupported_platform(monkeypatch):
    monkeypatch.delattr(pytorch_lightning.strategies.launchers.spawn.os, "fork")
    with pytest.raises(ValueError, match="The start method 'fork' is not available on this platform"):
        _SpawnLauncher(strategy=Mock(), start_method="fork")


@pytest.mark.parametrize("start_method", ["spawn", "fork"])
@mock.patch("pytorch_lightning.strategies.launchers.spawn.mp")
def test_spawn_launcher_start_method(mp_mock, start_method):
    launcher = _SpawnLauncher(strategy=Mock(), start_method=start_method)
    launcher.launch(function=Mock())
    mp_mock.get_context.assert_called_with(start_method)
    mp_mock.start_processes.assert_called_with(
        ANY,
        args=ANY,
        nprocs=ANY,
        start_method=start_method,
    )
