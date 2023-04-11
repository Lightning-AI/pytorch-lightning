from copy import deepcopy
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig, LocalStateDictConfig, ShardedStateDictConfig, ShardedOptimStateDictConfig
# from torch.distributed.fsdp.

from torch.distributed._shard.checkpoint import FileSystemWriter, FileSystemReader, save_state_dict, load_state_dict


import torch.multiprocessing as mp

from lightning.fabric import Fabric
from lightning.fabric.strategies import FSDPStrategy


def worker(i):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    device = torch.device("cuda", i)
    torch.distributed.init_process_group("nccl", rank=i, world_size=2)

    model = nn.Linear(10, 10)
    model = FSDP(model, auto_wrap_policy=always_wrap_policy, device_id=device)

    print(i, model.weight.numel())
    print(i, model.bias.numel())


def main():
    # mp.spawn(worker, nprocs=2)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy))
    fabric.launch()

    model = nn.Linear(10, 10, bias=False)
    model = fabric.setup_module(model)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)

    checkpoint_path = Path("there.ckpt")
    
    weights_before = deepcopy(list(model.parameters()))
    # optim_state_before = deepcopy(optimizer.state)

    state = {"model": model, "optimizer": optimizer, "epoch": 3}
    fabric.save(checkpoint_path, state)

    assert set(os.listdir(checkpoint_path)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    # load checkpoint into fresh model and optimizer
    model = nn.Linear(10, 10, bias=False)
    model = fabric.setup_module(model)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)
    
    state = {"model": model, "optimizer": optimizer}
    remainder = fabric.load(checkpoint_path, state)

    weights_after = deepcopy(list(model.parameters()))

    assert all(torch.equal(p0, p1) for p0, p1 in zip(weights_before, weights_after))
    assert remainder["epoch"] == 3


    



if __name__ == "__main__":
    main()