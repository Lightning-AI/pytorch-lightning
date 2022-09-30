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

import pytest
import torch
from tests_lite.helpers.models import BoringLite
from tests_lite.helpers.runif import RunIf


class ShardedSaveAndLoad(BoringLite):
    def get_optimizer(self, module):
        optimizer = super().get_optimizer(module)
        if self.with_fairscale_oss:
            from fairscale.optim import OSS

            optimizer = OSS(params=optimizer.param_groups, optim=type(optimizer), **optimizer.defaults)
        return optimizer

    def run(self, tmpdir, with_fairscale_oss=False):
        self.with_fairscale_oss = with_fairscale_oss

        super().run()

        from fairscale.nn import ShardedDataParallel
        from fairscale.optim import OSS

        # the model and optimizer is wrapped correctly
        assert isinstance(self.model._forward_module, ShardedDataParallel)
        assert isinstance(self.optimizer.optimizer, OSS)

        self.model.cpu()

        checkpoint_path = os.path.join(tmpdir, "checkpoint.ckpt")
        checkpoint = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        self.save(checkpoint, checkpoint_path)

        self.barrier()

        loaded_checkpoint = self.load(checkpoint_path)
        new_model = self.get_model()
        new_model.load_state_dict(loaded_checkpoint["model"])

        # Assert model parameters are identical after loading
        for trained_param, loaded_param in zip(self.model.parameters(), new_model.parameters()):
            assert torch.equal(trained_param, loaded_param)


@RunIf(fairscale=True)
@pytest.mark.parametrize("accelerator", ["cpu", pytest.param("cuda", marks=RunIf(min_cuda_gpus=2))])
@pytest.mark.parametrize("strategy", (pytest.param("ddp_sharded", marks=RunIf(standalone=True)), "ddp_sharded_spawn"))
@pytest.mark.parametrize("with_fairscale_oss", (True, False))
def test_fairscale_multi_process_checkpoint_state_consolidation(with_fairscale_oss, strategy, accelerator, tmpdir):
    """Test that the sharded optimizer states get consolidated when saving the checkpoint, and that the loaded
    weights is identical to the saved one."""
    lite = ShardedSaveAndLoad(strategy=strategy, accelerator=accelerator, devices=2)
    lite.run(tmpdir, with_fairscale_oss=with_fairscale_oss)


@pytest.mark.parametrize(
    "strategy, expected_find_unused_parameters",
    [
        ("ddp_sharded", None),
        ("ddp_sharded_find_unused_parameters_false", False),
        ("ddp_sharded_spawn", None),
        ("ddp_sharded_spawn_find_unused_parameters_false", False),
    ],
)
def test_fairscale_find_unused_parameters_from_registry(strategy, expected_find_unused_parameters):
    lite = BoringLite(strategy=strategy)
    if expected_find_unused_parameters is not None:
        assert lite._strategy._ddp_kwargs["find_unused_parameters"] is False
