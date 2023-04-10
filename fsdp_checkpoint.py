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
    model.weight.data.fill_(1.)
    
    model = fabric.setup_module(model)
    print(fabric.global_rank, model.weight.numel())
    print(fabric.global_rank, next(model.parameters()).numel())
    optimizer = torch.optim.Adam(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)

    checkpoint_path = Path("there.ckpt")
    
    weight_before = deepcopy(list(model.parameters()))

    state = {"model": model, "optimizer": optimizer}
    fabric.save(checkpoint_path, state)

    assert set(os.listdir(checkpoint_path)) == {".metadata", "__0_0.distcp", "__1_0.distcp"}

    # load checkpoint into fresh model
    model = nn.Linear(10, 10, bias=False)
    model = fabric.setup_module(model)
    state = {"model": model}
    fabric.load(checkpoint_path, state)

    weight_after = deepcopy(list(model.parameters()))
    print(weight_after)
    # for p0, p1 in zip(weight_before, weight_after):
    #     assert torch.equal(p0, p1)

    

    # with torch.no_grad():
    #     save_policy = ShardedStateDictConfig(offload_to_cpu=True)
    #     optim_policy = ShardedOptimStateDictConfig(offload_to_cpu=True)
    #     with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_dict_config=save_policy, optim_state_dict_config=optim_policy):
    #         state_dict = model._forward_module.state_dict()
    #         optim_state_dict = FSDP.optim_state_dict(model, optimizer.optimizer)

    # state = {"model": state_dict, "optimizer": optim_state_dict}
    # writer = FileSystemWriter(path="here.ckpt", single_file_per_rank=False)
    # save_state_dict(state, writer)

    # print(fabric.global_rank, state)

    



if __name__ == "__main__":
    main()