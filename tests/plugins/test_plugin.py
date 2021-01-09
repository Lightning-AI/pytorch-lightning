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

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base.boring_model import BoringModel


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
def test_custom_required_plugins(tmpdir, ddp_backend, gpus, num_processes):
    """
    Test to ensure that if a plugin requires certain plugin to be added, these are added automatically
    """

    class RequiredPlugin(NativeAMPPlugin):
        """
        My custom amp plugin that's required with my DDP plugin as default.
        This allows us to ensure this plugin is added when using CustomPlugin rather than ensuring
        the user passes it manually into the list.
        """

    class CustomPlugin(DDPPlugin):
        def required_plugins(self, amp_backend: AMPType, trainer: Trainer) -> list:
            return [RequiredPlugin(trainer=trainer)]

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend.ddp_plugin, CustomPlugin)
            assert isinstance(trainer.precision_connector.backend, RequiredPlugin)
            raise RuntimeError('finished plugin check')

    model = BoringModel()
    with pytest.warns(UserWarning,
                      match=f'plugin {type(CustomPlugin())} has added additional '
                            f'required plugins as default: {[type(RequiredPlugin())]}*'):
        trainer = Trainer(
            fast_dev_run=True,
            gpus=gpus,
            num_processes=num_processes,
            accelerator=ddp_backend,
            plugins=[CustomPlugin()],
            callbacks=[CB()],
        )
    with pytest.raises(RuntimeError, match='finished plugin check'):
        trainer.fit(model)


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
def test_invalid_custom_required_plugins(tmpdir, ddp_backend, gpus, num_processes):
    """
    Test to ensure if the user passes a plugin that conflicts with the required defaults of another plugin,
    we throw a warning and error.
    The user has to override the required defaults plugin.
    """

    class RequiredPlugin(NativeAMPPlugin):
        """
        My custom amp plugin that's required with my DDP plugin as default.
        This allows us to ensure this plugin is added when using CustomPlugin rather than ensuring
        the user passes it manually into the list.
        """

    class CustomPlugin(DDPPlugin):
        def required_plugins(self, amp_backend: AMPType, trainer: Trainer) -> list:
            return [RequiredPlugin(trainer=trainer)]

    with pytest.warns(UserWarning, match=f'plugin {type(CustomPlugin())} has added additional '
                                         f'required plugins as default: {[type(RequiredPlugin())]}*'), \
         pytest.raises(MisconfigurationException, match=f"you can only use one {type(NativeAMPPlugin)}"
                                                        f" in plugins. You passed in: {2}"):
        Trainer(
            fast_dev_run=True,
            gpus=gpus,
            num_processes=num_processes,
            accelerator=ddp_backend,
            plugins=[CustomPlugin(), NativeAMPPlugin()],
        )
