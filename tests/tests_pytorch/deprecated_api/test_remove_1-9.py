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

import pytest

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.core.module import LightningModule


def test_old_lightningmodule_path():
    from pytorch_lightning.core.lightning import LightningModule

    with pytest.deprecated_call(
        match="pytorch_lightning.core.lightning.LightningModule has been deprecated in v1.7"
        " and will be removed in v1.9."
    ):
        LightningModule()


def test_old_loop_path():
    from pytorch_lightning.loops.base import Loop

    class MyLoop(Loop):
        def advance(self):
            ...

        def done(self):
            ...

        def reset(self):
            ...

    with pytest.deprecated_call(match="pytorch_lightning.loops.base.Loop has been deprecated in v1.7"):
        MyLoop()


def test_lightningCLI_seed_everything_default_to_None_deprecation_warning():
    with mock.patch("sys.argv", ["any.py"]), pytest.deprecated_call(
        match="Setting `LightningCLI.seed_everything_default` to `None` is deprecated in v1.7 "
        "and will be removed in v1.9. Set it to `False` instead."
    ):
        LightningCLI(LightningModule, run=False, seed_everything_default=None)


def test_old_callback_path():
    from pytorch_lightning.callbacks.base import Callback

    with pytest.deprecated_call(
        match="pytorch_lightning.callbacks.base.Callback has been deprecated in v1.7 and will be removed in v1.9."
    ):
        Callback()
