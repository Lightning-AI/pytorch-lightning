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
from unittest import mock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DDPStrategy

from tests_pytorch.helpers.runif import RunIf

if torch.distributed.is_available():
    import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD


class TestDDPStrategy(DDPStrategy):
    def __init__(self, expected_ddp_comm_hook_name, *args, **kwargs):
        self.expected_ddp_comm_hook_name = expected_ddp_comm_hook_name
        super().__init__(*args, **kwargs)

    def teardown(self):
        # check here before unwrapping DistributedDataParallel in self.teardown
        attached_ddp_comm_hook_name = self.model._get_ddp_logging_data()["comm_hook"]
        assert attached_ddp_comm_hook_name == self.expected_ddp_comm_hook_name
        return super().teardown()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_ddp_fp16_compress_comm_hook(tmp_path):
    """Test for DDP FP16 compress hook."""
    model = BoringModel()
    strategy = TestDDPStrategy(
        expected_ddp_comm_hook_name=default.fp16_compress_hook.__qualname__,
        ddp_comm_hook=default.fp16_compress_hook,
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        default_root_dir=tmp_path,
        sync_batchnorm=True,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_ddp_sgd_comm_hook(tmp_path):
    """Test for DDP FP16 compress hook."""
    model = BoringModel()
    strategy = TestDDPStrategy(
        expected_ddp_comm_hook_name=powerSGD.powerSGD_hook.__qualname__,
        ddp_comm_state=powerSGD.PowerSGDState(process_group=None),
        ddp_comm_hook=powerSGD.powerSGD_hook,
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        default_root_dir=tmp_path,
        sync_batchnorm=True,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_ddp_fp16_compress_wrap_sgd_comm_hook(tmp_path):
    """Test for DDP FP16 compress wrapper for SGD hook."""
    model = BoringModel()
    strategy = TestDDPStrategy(
        expected_ddp_comm_hook_name=default.fp16_compress_wrapper(powerSGD.powerSGD_hook).__qualname__,
        ddp_comm_state=powerSGD.PowerSGDState(process_group=None),
        ddp_comm_hook=powerSGD.powerSGD_hook,
        ddp_comm_wrapper=default.fp16_compress_wrapper,
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        default_root_dir=tmp_path,
        sync_batchnorm=True,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_ddp_spawn_fp16_compress_comm_hook(tmp_path):
    """Test for DDP Spawn FP16 compress hook."""
    model = BoringModel()
    strategy = DDPStrategy(ddp_comm_hook=default.fp16_compress_hook, start_method="spawn")
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        default_root_dir=tmp_path,
        sync_batchnorm=True,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_ddp_post_local_sgd_comm_hook(tmp_path):
    """Test for DDP post-localSGD hook."""
    model = BoringModel()
    strategy = TestDDPStrategy(
        expected_ddp_comm_hook_name=post_localSGD.post_localSGD_hook.__qualname__,
        ddp_comm_state=post_localSGD.PostLocalSGDState(
            process_group=None,
            subgroup=None,
            start_localSGD_iter=8,
        ),
        ddp_comm_hook=post_localSGD.post_localSGD_hook,
        model_averaging_period=4,
    )
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        default_root_dir=tmp_path,
        sync_batchnorm=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(skip_windows=True, min_cuda_gpus=2, standalone=True)
@mock.patch("torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager.average_parameters")
def test_post_local_sgd_model_averaging(average_parameters_mock, tmp_path):
    """Test that when using DDP with post-localSGD, model averaging is called."""
    model = BoringModel()

    # test regular ddp does not call model averaging
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        default_root_dir=tmp_path,
        sync_batchnorm=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model)
    average_parameters_mock.assert_not_called()

    # test ddp with post-localSGD does call model averaging
    ddp_strategy = DDPStrategy(
        ddp_comm_state=post_localSGD.PostLocalSGDState(
            process_group=None,
            subgroup=None,
            start_localSGD_iter=8,
        ),
        ddp_comm_hook=post_localSGD.post_localSGD_hook,
        model_averaging_period=4,
    )

    trainer = Trainer(
        fast_dev_run=True,
        accelerator="gpu",
        devices=2,
        strategy=ddp_strategy,
        default_root_dir=tmp_path,
        sync_batchnorm=True,
    )

    trainer.fit(model)
    average_parameters_mock.assert_called()


@RunIf(skip_windows=True, min_cuda_gpus=2, standalone=True)
@mock.patch("torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager.average_parameters")
def test_post_local_sgd_model_averaging_raises(average_parameters_mock, tmp_path):
    """Test that when using DDP with post-localSGD a ValueError is thrown when the optimizer is
    ZeroRedundancyOptimizer."""
    from torch.distributed.optim import ZeroRedundancyOptimizer

    class OptimizerModel(BoringModel):
        def configure_optimizers(self):
            return ZeroRedundancyOptimizer(params=self.parameters(), optimizer_class=torch.optim.Adam, lr=0.01)

    model = OptimizerModel()
    strategy = DDPStrategy(
        ddp_comm_state=post_localSGD.PostLocalSGDState(
            process_group=None,
            subgroup=None,
            start_localSGD_iter=8,
        ),
        ddp_comm_hook=post_localSGD.post_localSGD_hook,
        model_averaging_period=4,
    )

    trainer = Trainer(
        fast_dev_run=True,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        default_root_dir=tmp_path,
        sync_batchnorm=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with pytest.raises(ValueError, match="Currently model averaging cannot work with a distributed optimizer"):
        trainer.fit(model)

    average_parameters_mock.assert_not_called()
