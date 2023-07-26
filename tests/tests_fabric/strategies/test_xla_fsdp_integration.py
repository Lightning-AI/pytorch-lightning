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

import pytest
import torch
from torch.utils.data import DataLoader

from lightning.fabric import Fabric
from lightning.fabric.strategies import XLAFSDPStrategy
from tests_fabric.helpers.models import RandomDataset
from tests_fabric.helpers.runif import RunIf


def xla_fsdp_train_save_load(fabric: Fabric, tmp_path, state_dict_type):
    """Fabric launch function for test_xla_fsdp_train_save_load."""
    # check if multihost
    if fabric.strategy.all_reduce(fabric.node_rank, reduce_op="sum").item() > 0:
        return  # pytest.skip() is not pickleable

    checkpoint_path = fabric.broadcast(str(tmp_path))
    with fabric.init_module():
        model_1 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    model_1 = fabric.setup_module(model_1)

    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.1)
    optimizer_1 = fabric.setup_optimizers(optimizer_1)

    dataloader = DataLoader(RandomDataset(32, 64))
    dataloader = fabric.setup_dataloaders(dataloader)

    def step(model, batch):
        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))

    model_1.train()
    data_iter = iter(dataloader)
    batch = next(data_iter)
    loss = step(model_1, batch)
    fabric.backward(loss)
    optimizer_1.step()
    optimizer_1.zero_grad()

    state = {
        "model": model_1,
        "optimizer": optimizer_1,  # not needed in ckpt consolidation
        "step_count": 1,
    }

    fabric.save(checkpoint_path, state)

    world_size = fabric.world_size

    if state_dict_type == "sharded":
        expected_files = {f"checkpoint_rank-{i:08d}-of-{world_size:08d}.pth" for i in range(world_size)}
        assert set(os.listdir(checkpoint_path)) == expected_files

        # define a second set of model and optimizer
        with fabric.init_module():
            model_2 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        model_2 = fabric.setup_module(model_2)

        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)
        optimizer_2 = fabric.setup_optimizers(optimizer_2)

        # load sharded checkpoints into the second set of model and optimizer
        state = {
            "model": model_2,
            "optimizer": optimizer_2,
            "step_count": 0,
        }
        metadata = fabric.load(checkpoint_path, state)

        # check user data in loaded state
        assert not metadata
        assert state["step_count"] == 1

        # check correctness with loaded state
        for p0, p1 in zip(model_1._forward_module.parameters(), model_2.parameters()):
            torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)

        # attempt to load a key not in the metadata checkpoint
        state = {"model": model_2, "coconut": 11}
        with pytest.raises(KeyError, match="The requested state contains a key 'coconut' that does not exist"):
            fabric.load(checkpoint_path, state)

        # `strict=False` ignores the missing key
        state = {"model": model_2, "coconut": 11}
        fabric.load(checkpoint_path, state, strict=False)
        assert state["coconut"] == 11

    if state_dict_type == "full":
        assert set(os.listdir(checkpoint_path)) == {"checkpoint_consolidated.pth"}

        # define a second set of model and optimizer
        with fabric.init_module():
            model_2 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        model_2.to(device)

        # load sharded checkpoints into the second model
        state = {"model": model_2}
        fabric.load(checkpoint_path, state)

        # check that loaded state is different
        with pytest.raises(AssertionError, match="do not match"):
            for p0, p1 in zip(model_1.parameters(), model_2.parameters()):
                torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)


@RunIf(min_torch="2.0", tpu=True, standalone=True)
@pytest.mark.parametrize("use_auto_wrap_policy", [False, True])
@pytest.mark.parametrize("state_dict_type", ["sharded", "full"])
def test_xla_fsdp_train_save_load(tmp_path, use_auto_wrap_policy, state_dict_type):
    """Test XLAFSDP training, saving and loading checkpoint (both full and sharded)."""
    from torch_xla.distributed.fsdp.wrap import always_wrap_policy

    strategy = XLAFSDPStrategy(
        auto_wrap_policy=always_wrap_policy if use_auto_wrap_policy else None, state_dict_type=state_dict_type
    )
    fabric = Fabric(accelerator="tpu", strategy=strategy)
    fabric.launch(xla_fsdp_train_save_load, tmp_path, state_dict_type)
