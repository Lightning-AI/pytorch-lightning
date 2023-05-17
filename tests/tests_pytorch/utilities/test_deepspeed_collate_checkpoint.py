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

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from tests_pytorch.helpers.runif import RunIf


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_collate_checkpoint(tmpdir):
    """Test to ensure that with DeepSpeed Stage 3 we can collate the sharded checkpoints into a single file."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    checkpoint_path = os.path.join(tmpdir, "model.pt")
    checkpoint_path = trainer.strategy.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)

    output_path = trainer.strategy.broadcast(os.path.join(tmpdir, "single_model.pt"))
    if trainer.is_global_zero:
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, output_path)
    
    trainer.strategy.barrier()

    import deepspeed
    with deepspeed.zero.GatheredParameters(model.parameters()):
        _inspect(output_path)

    # import deepspeed
    # with deepspeed.zero.GatheredParameters(model.parameters()):
    #     _assert_checkpoint_equal(model, output_path)


def _assert_checkpoint_equal(model, output_path):
    assert os.path.exists(output_path)
    single_output = torch.load(output_path)
    state_dict = model.state_dict()
    for orig_param, saved_model_param in zip(state_dict.values(), single_output["state_dict"].values()):
        if model.dtype == torch.half:
            # moved model to float32 for comparison with single fp32 saved weights
            saved_model_param = saved_model_param
        print(orig_param.shape, saved_model_param.shape)
        # assert torch.allclose(orig_param.cpu(), saved_model_param.half())


def _inspect(output_path):
    assert os.path.exists(output_path)
    single_output = torch.load(output_path)
    print(list(single_output["state_dict"].keys()))
    for saved_model_param in single_output["state_dict"].values():
        print(saved_model_param.shape)
        # assert torch.allclose(orig_param.cpu(), saved_model_param.half())
