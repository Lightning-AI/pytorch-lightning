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

from pytorch_lightning import Trainer
from tests.accelerators import ddp_model, DDPLauncher
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf
from tests.utilities.distributed import call_training_script

CLI_ARGS = '--max_epochs 1 --gpus 2 --accelerator ddp'


@RunIf(min_gpus=2)
def test_multi_gpu_model_ddp_fit_only(tmpdir):
    # call the script
    call_training_script(ddp_model, CLI_ARGS, 'fit', tmpdir, timeout=120)

    # load the results of the script
    result_path = os.path.join(tmpdir, 'ddp.result')
    result = torch.load(result_path)

    # verify the file wrote the expected outputs
    assert result['status'] == 'complete'


@RunIf(min_gpus=2)
def test_multi_gpu_model_ddp_test_only(tmpdir):
    # call the script
    call_training_script(ddp_model, CLI_ARGS, 'test', tmpdir)

    # load the results of the script
    result_path = os.path.join(tmpdir, 'ddp.result')
    result = torch.load(result_path)

    # verify the file wrote the expected outputs
    assert result['status'] == 'complete'


@RunIf(min_gpus=2)
def test_multi_gpu_model_ddp_fit_test(tmpdir):
    # call the script
    call_training_script(ddp_model, CLI_ARGS, 'fit_test', tmpdir, timeout=20)

    # load the results of the script
    result_path = os.path.join(tmpdir, 'ddp.result')
    result = torch.load(result_path)

    # verify the file wrote the expected outputs
    assert result['status'] == 'complete'

    model_outs = result['result']
    for out in model_outs:
        assert out['test_acc'] > 0.7


@RunIf(min_gpus=2)
@DDPLauncher.run(
    "--max_epochs [max_epochs] --gpus 2 --accelerator [accelerator]",
    max_epochs=["1"],
    accelerator=["ddp", "ddp_spawn"]
)
def test_cli_to_pass(tmpdir, args=None):
    """
    This test verify we can call function using test_cli name
    """
    return '1'


@RunIf(skip_windows=True)
@pytest.mark.skipif(torch.cuda.is_available(), reason="test doesn't requires GPU machine")
def test_torch_distributed_backend_env_variables(tmpdir):
    """
    This test set `undefined` as torch backend and should raise an `Backend.UNDEFINED` ValueError.
    """
    _environ = {"PL_TORCH_DISTRIBUTED_BACKEND": "undefined", "CUDA_VISIBLE_DEVICES": "0,1", "WORLD_SIZE": "2"}
    with patch.dict(os.environ, _environ), \
         patch('torch.cuda.device_count', return_value=2):
        with pytest.raises(ValueError, match="Invalid backend: 'undefined'"):
            model = BoringModel()
            trainer = Trainer(
                default_root_dir=tmpdir,
                fast_dev_run=True,
                accelerator="ddp",
                gpus=2,
                logger=False,
            )
            trainer.fit(model)


@RunIf(skip_windows=True)
@mock.patch('torch.cuda.device_count', return_value=1)
@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.set_device')
@mock.patch.dict(os.environ, {'PL_TORCH_DISTRIBUTED_BACKEND': 'gloo'}, clear=True)
def test_ddp_torch_dist_is_available_in_setup(mock_set_device, mock_is_available, mock_device_count, tmpdir):
    """
    Test to ensure torch distributed is available within the setup hook using ddp
    """

    class TestModel(BoringModel):

        def setup(self, stage: Optional[str] = None) -> None:
            assert torch.distributed.is_initialized()
            raise SystemExit()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator="ddp",
        gpus=1,
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)
