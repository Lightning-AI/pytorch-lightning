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
from typing import Optional
from unittest import mock
from unittest.mock import patch

import pytest
import torch
from torch.nn.parallel.distributed import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf
from tests.strategies import ddp_model
from tests.utilities.distributed import call_training_script

CLI_ARGS = "--max_epochs 1 --accelerator 'gpu' --devices 2 --strategy ddp"


@RunIf(min_gpus=2)
@pytest.mark.parametrize("as_module", [True, False])
def test_multi_gpu_model_ddp_fit_only(tmpdir, as_module):
    # call the script
    call_training_script(ddp_model, CLI_ARGS, "fit", tmpdir, timeout=120, as_module=as_module)

    # load the results of the script
    result_path = os.path.join(tmpdir, "ddp.result")
    result = torch.load(result_path)

    # verify the file wrote the expected outputs
    assert result["status"] == "complete"


@RunIf(min_gpus=2)
@pytest.mark.parametrize("as_module", [True, False])
def test_multi_gpu_model_ddp_test_only(tmpdir, as_module):
    # call the script
    call_training_script(ddp_model, CLI_ARGS, "test", tmpdir, as_module=as_module)

    # load the results of the script
    result_path = os.path.join(tmpdir, "ddp.result")
    result = torch.load(result_path)

    # verify the file wrote the expected outputs
    assert result["status"] == "complete"


@RunIf(min_gpus=2)
@pytest.mark.parametrize("as_module", [True, False])
def test_multi_gpu_model_ddp_fit_test(tmpdir, as_module):
    # call the script
    call_training_script(ddp_model, CLI_ARGS, "fit_test", tmpdir, timeout=20, as_module=as_module)

    # load the results of the script
    result_path = os.path.join(tmpdir, "ddp.result")
    result = torch.load(result_path)

    # verify the file wrote the expected outputs
    assert result["status"] == "complete"

    model_outs = result["result"]
    for out in model_outs:
        assert out["test_acc"] > 0.7


@RunIf(skip_windows=True)
@pytest.mark.skipif(torch.cuda.is_available(), reason="test doesn't requires GPU machine")
@mock.patch("torch.cuda.is_available", return_value=True)
def test_torch_distributed_backend_env_variables(tmpdir):
    """This test set `undefined` as torch backend and should raise an `Backend.UNDEFINED` ValueError."""
    _environ = {"PL_TORCH_DISTRIBUTED_BACKEND": "undefined", "CUDA_VISIBLE_DEVICES": "0,1", "WORLD_SIZE": "2"}
    with patch.dict(os.environ, _environ), patch("torch.cuda.device_count", return_value=2):
        with pytest.deprecated_call(match="Environment variable `PL_TORCH_DISTRIBUTED_BACKEND` was deprecated in v1.6"):
            with pytest.raises(ValueError, match="Invalid backend: 'undefined'"):
                model = BoringModel()
                trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy="ddp", gpus=2, logger=False)
                trainer.fit(model)


@RunIf(skip_windows=True)
@mock.patch("torch.cuda.set_device")
@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("pytorch_lightning.accelerators.gpu.GPUAccelerator.is_available", return_value=True)
@mock.patch.dict(os.environ, {"PL_TORCH_DISTRIBUTED_BACKEND": "gloo"}, clear=True)
def test_ddp_torch_dist_is_available_in_setup(
    mock_gpu_is_available, mock_device_count, mock_cuda_available, mock_set_device, tmpdir
):
    """Test to ensure torch distributed is available within the setup hook using ddp."""

    class TestModel(BoringModel):
        def setup(self, stage: Optional[str] = None) -> None:
            assert torch.distributed.is_initialized()
            raise SystemExit()

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy="ddp", gpus=1)
    with pytest.deprecated_call(match="Environment variable `PL_TORCH_DISTRIBUTED_BACKEND` was deprecated in v1.6"):
        with pytest.raises(SystemExit):
            trainer.fit(model)


@RunIf(min_gpus=2, min_torch="1.8.1", standalone=True)
@pytest.mark.parametrize("precision", (16, 32))
def test_ddp_wrapper(tmpdir, precision):
    """Test parameters to ignore are carried over for DDP."""

    class WeirdModule(torch.nn.Module):
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            return {"something": "something"}

    class CustomModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.weird_module = WeirdModule()

            # should be skip.
            self._ddp_params_and_buffers_to_ignore = ["something"]

    class CustomCallback(Callback):
        def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert isinstance(trainer.strategy.model, DistributedDataParallel)
            assert trainer.strategy.model.parameters_to_ignore == ["module.something"]
            assert trainer.strategy.model.module._ddp_params_and_buffers_to_ignore == ["module.something"]

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        precision=precision,
        strategy="ddp",
        gpus=2,
        callbacks=CustomCallback(),
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    ["process_group_backend", "env_var", "device_str", "expected_process_group_backend"],
    [
        pytest.param("foo", None, "cpu", "foo"),
        pytest.param("foo", "BAR", "cpu", "foo"),
        pytest.param("foo", "BAR", "cuda:0", "foo"),
        pytest.param(None, "BAR", "cuda:0", "BAR"),
        pytest.param(None, None, "cuda:0", "nccl"),
        pytest.param(None, None, "cpu", "gloo"),
    ],
)
def test_ddp_process_group_backend(process_group_backend, env_var, device_str, expected_process_group_backend):
    """Test settings for process group backend."""

    class MockDDPStrategy(DDPStrategy):
        def __init__(self, root_device, process_group_backend):
            self._root_device = root_device
            super().__init__(process_group_backend=process_group_backend)

        @property
        def root_device(self):
            return self._root_device

    strategy = MockDDPStrategy(process_group_backend=process_group_backend, root_device=torch.device(device_str))
    if not process_group_backend and env_var:
        with mock.patch.dict(os.environ, {"PL_TORCH_DISTRIBUTED_BACKEND": env_var}):
            with pytest.deprecated_call(
                match="Environment variable `PL_TORCH_DISTRIBUTED_BACKEND` was deprecated in v1.6"
            ):
                assert strategy._get_process_group_backend() == expected_process_group_backend
    else:
        assert strategy._get_process_group_backend() == expected_process_group_backend
