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
import pytest
import torch
from tests_fabric.helpers.models import BoringFabric
from tests_fabric.helpers.runif import RunIf


class ShardedSaveAndLoad(BoringFabric):
    def run(self, tmp_path):
        super().run()

        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

        # the model is wrapped correctly
        assert isinstance(self.model._forward_module, FullyShardedDataParallel)

        self.model.cpu()

        checkpoint_path = tmp_path / "checkpoint.ckpt"
        # need to broadcast because tmpdir is different on each process
        checkpoint_path = self.broadcast(checkpoint_path)

        checkpoint = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        self.save(checkpoint, checkpoint_path)

        self.barrier()  # ensure the checkpoint is saved before load

        loaded_checkpoint = self.load(checkpoint_path)
        new_model = self.get_model()
        new_model.load_state_dict(loaded_checkpoint["model"])

        # Assert model parameters are identical after loading
        for trained_param, loaded_param in zip(self.model.parameters(), new_model.parameters()):
            assert torch.equal(trained_param, loaded_param)


@RunIf(standalone=True, min_cuda_gpus=2, min_torch="1.12")
@pytest.mark.parametrize("strategy", (pytest.param("ddp_sharded", marks=RunIf(standalone=True)), "ddp_sharded_spawn"))
def test_ddp_sharded_multi_process_checkpoint_state_consolidation(tmp_path, strategy):
    """Test that the sharded optimizer states get consolidated when saving the checkpoint, and that the loaded
    weights is identical to the saved one."""
    fabric = ShardedSaveAndLoad(strategy=strategy, accelerator="cuda", devices=2)
    fabric.run(tmp_path)
