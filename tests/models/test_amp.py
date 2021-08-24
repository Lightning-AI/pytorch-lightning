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
import torch
from torch import optim
from torch.utils.data import DataLoader

import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import CPUNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.utilities import _TORCH_CPU_AMP_AVAILABLE, _TORCH_GREATER_EQUAL_1_10
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


class AMPTestModel(BoringModel):
    def _step(self, batch, batch_idx):
        self._assert_autocast_enabled()
        output = self(batch)
        bfloat16 = self.trainer.precision_plugin.is_bfloat16
        assert output.dtype == torch.float16 if not bfloat16 else torch.bfloat16
        loss = self.loss(batch, output)
        return loss

    def loss(self, batch, prediction):
        # todo (sean): convert bfloat16 to float32 as mse loss for cpu amp is currently not supported
        if isinstance(self.trainer.precision_plugin, CPUNativeMixedPrecisionPlugin):
            prediction = prediction.float()
        return super().loss(batch, prediction)

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        return {"loss": output}

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        return {"x": output}

    def test_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        return {"y": output}

    def predict(self, batch, batch_idx, dataloader_idx=None):
        self._assert_autocast_enabled()
        output = self(batch)
        bfloat16 = self.trainer.precision_plugin.is_bfloat16
        assert output.dtype == torch.float16 if not bfloat16 else torch.bfloat16
        return output

    def _assert_autocast_enabled(self):
        if isinstance(self.trainer.precision_plugin, CPUNativeMixedPrecisionPlugin):
            assert torch.is_autocast_cpu_enabled()
        else:
            assert torch.is_autocast_enabled()


@pytest.mark.skipif(not _TORCH_CPU_AMP_AVAILABLE, reason="CPU AMP not available")
@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        pytest.param("dp", marks=pytest.mark.skip("dp + amp not supported currently")),  # TODO
        "ddp_spawn",
    ],
)
@pytest.mark.parametrize(
    "precision",
    [
        pytest.param(16, marks=pytest.mark.skip("CPU precision 16 is not supported in PyTorch yet.")),  # TODO
        "bf16",
    ],
)
@pytest.mark.parametrize("num_processes", [1, 2])
def test_amp_cpus(tmpdir, accelerator, precision, num_processes):
    """Make sure combinations of AMP and training types work if supported."""
    tutils.reset_seed()

    trainer = Trainer(
        default_root_dir=tmpdir, num_processes=num_processes, max_epochs=1, accelerator=accelerator, precision=precision
    )

    model = AMPTestModel()
    # tutils.run_model_test(trainer_options, model)
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model, DataLoader(RandomDataset(32, 64)))

    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        pytest.param("dp", marks=pytest.mark.skip("dp + amp not supported currently")),  # TODO
        "ddp_spawn",
    ],
)
@pytest.mark.parametrize(
    "precision",
    [
        16,
        pytest.param(
            "bf16",
            marks=pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_10, reason="torch.bfloat16 not available"),
        ),
    ],
)
@pytest.mark.parametrize("gpus", [1, 2])
def test_amp_gpus(tmpdir, accelerator, precision, gpus):
    """Make sure combinations of AMP and training types work if supported."""
    tutils.reset_seed()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, gpus=gpus, accelerator=accelerator, precision=precision)

    model = AMPTestModel()
    # tutils.run_model_test(trainer_options, model)
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model, DataLoader(RandomDataset(32, 64)))

    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(min_gpus=2)
@mock.patch.dict(
    os.environ,
    {
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
        "SLURM_PROCID": "0",
    },
)
def test_amp_gpu_ddp_slurm_managed(tmpdir):
    """Make sure DDP + AMP work."""
    # simulate setting slurm flags
    tutils.set_random_master_port()

    model = AMPTestModel()

    # exp file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=[0],
        accelerator="ddp_spawn",
        precision=16,
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)

    # correct result and ok accuracy
    assert trainer.state.finished, "amp + ddp model failed to complete"

    # test root model address
    assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
    assert trainer.training_type_plugin.cluster_environment.resolve_root_node_address("abc") == "abc"
    assert trainer.training_type_plugin.cluster_environment.resolve_root_node_address("abc[23]") == "abc23"
    assert trainer.training_type_plugin.cluster_environment.resolve_root_node_address("abc[23-24]") == "abc23"
    generated = trainer.training_type_plugin.cluster_environment.resolve_root_node_address("abc[23-24, 45-40, 40]")
    assert generated == "abc23"


@pytest.mark.skipif(torch.cuda.is_available(), reason="test is restricted only on CPU")
def test_cpu_model_with_amp(tmpdir):
    """Make sure model trains on CPU."""
    with pytest.raises(MisconfigurationException, match="AMP is only available on GPU"):
        Trainer(precision=16)


@mock.patch("pytorch_lightning.plugins.precision.apex_amp.ApexMixedPrecisionPlugin.backward")
def test_amp_without_apex(bwd_mock, tmpdir):
    """Check that even with apex amp type without requesting precision=16 the amp backend is void."""
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmpdir, amp_backend="native")
    assert trainer.amp_backend is None

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, amp_backend="apex")
    assert trainer.amp_backend is None
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert not bwd_mock.called


@RunIf(min_gpus=1, amp_apex=True)
@mock.patch("pytorch_lightning.plugins.precision.apex_amp.ApexMixedPrecisionPlugin.backward")
def test_amp_with_apex(bwd_mock, tmpdir):
    """Check calling apex scaling in training."""

    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=0.01)
            optimizer2 = optim.SGD(self.parameters(), lr=0.01)
            lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 1, gamma=0.1)
            lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)
            return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    model = CustomModel()
    model.training_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, max_steps=5, precision=16, amp_backend="apex", gpus=1)
    assert str(trainer.amp_backend) == "AMPType.APEX"
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert bwd_mock.call_count == 10

    assert isinstance(trainer.lr_schedulers[0]["scheduler"].optimizer, optim.Adam)
    assert isinstance(trainer.lr_schedulers[1]["scheduler"].optimizer, optim.SGD)
