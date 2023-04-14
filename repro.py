import torch.multiprocessing as mp
import torch
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType


def run(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    torch.distributed.init_process_group("nccl", world_size=2, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = torch.nn.Linear(2, 2).to(device)

    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model = FSDP(model, use_orig_params=False, auto_wrap_policy=always_wrap_policy)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model(torch.rand(2, 2)).sum().backward()
    optimizer.step()

    # The state has `momentum_buffer = None`
    print(list(optimizer.state.values())[0]["momentum_buffer"])

    state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
    optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    state_dict_type = FSDP.state_dict_type(
        module=model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    )

    with state_dict_type:
        # AttributeError: 'NoneType' object has no attribute 'items'
        state = FSDP.optim_state_dict(model, optimizer)


if __name__ == "__main__":
    mp.spawn(run, nprocs=2)