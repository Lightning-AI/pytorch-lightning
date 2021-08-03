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
import subprocess

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities.collate_deepspeed_checkpoint import convert_zero_checkpoint_to_fp32_state_dict
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


@RunIf(min_gpus=2, deepspeed=True, special=False)
def test_deepspeed_collate_checkpoint(tmpdir):
    """
    Test to ensure that with DeepSpeed Stage 3 we can collate the sharded checkpoints into a single file.
    """
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir, plugins=[DeepSpeedPlugin(stage=3)], gpus=2, fast_dev_run=True, precision=16
    )
    trainer.fit(model)
    checkpoint_path = os.path.join(tmpdir, "model.pt")
    checkpoint_path = trainer.accelerator.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    trainer.accelerator.barrier()
    if trainer.is_global_zero:
        # ensure function call works
        output_path = os.path.join(tmpdir, "single_model.pt")
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, output_path)
        _assert_checkpoint_equal(model, output_path)

        # ensure utility script work
        output_path = os.path.join(tmpdir, "single_model_script.pt")
        cmd = f"python -m pytorch_lightning.utilities.collate_deepspeed_checkpoint {checkpoint_path} {output_path}"
        exit_code = subprocess.call(cmd, shell=True)
        assert exit_code == 0
        _assert_checkpoint_equal(model, output_path)


def _assert_checkpoint_equal(model, output_path):
    assert os.path.exists(output_path)
    single_output = torch.load(output_path)
    state_dict = model.state_dict()
    for orig_param, saved_model_param in zip(state_dict.values(), single_output["state_dict"].values()):
        if model.dtype == torch.half:
            # moved model to float32 for comparison with single fp32 saved weights
            saved_model_param = saved_model_param.half()
        assert torch.equal(orig_param.cpu(), saved_model_param)
