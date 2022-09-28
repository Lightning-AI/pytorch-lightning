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

from tests_lite.helpers.runif import RunIf
from tests_lite.helpers.models import BoringLite


class ShardedSaveAndLoad(BoringLite):
    def run(self, tmpdir):
        super().run()

        from fairscale.nn import ShardedDataParallel

        # the model is wrapped correctly
        assert isinstance(self.model._forward_module, ShardedDataParallel)

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
def test_fairscale_multi_process_checkpoint_state_consolidation(accelerator, tmpdir):
    """Test that the sharded optimizer states get consolidated when saving the checkpoint, and that the loaded weights
    is identical to the saved one."""
    lite = ShardedSaveAndLoad(strategy="ddp_sharded_spawn", accelerator=accelerator, devices=2)
    lite.run(tmpdir)
