import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


loss_fn = nn.MSELoss()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def _regular_forward(self, x):
        outputs = self.net2(self.relu(self.net1(x)))
        labels = torch.randn(20, 5)  # .to(rank)
        loss = loss_fn(outputs, labels)
        return loss

    def forward(self, x):
        yield self._regular_forward(x)
        yield self._regular_forward(x)


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    model = ToyModel()
    ddp_model = DDP(model)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    generator = ddp_model(torch.randn(20, 10))

    optimizer.zero_grad()
    next(generator).backward()
    print("step")
    optimizer.step()

    optimizer.zero_grad()
    next(generator).backward()
    print("step")
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    run_demo(demo_fn=demo_basic, world_size=2)
