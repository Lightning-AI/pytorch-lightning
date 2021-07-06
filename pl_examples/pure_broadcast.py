from argparse import Namespace
import torch
import torch.nn as nn
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning.utilities.cli import ArgumentParser


def run(args: Namespace):

    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)

    message = torch.tensor([local_rank], device=device)

    torch.distributed.init_process_group(backend="nccl", world_size=2, rank=local_rank)
    print("init successful")

    torch.distributed.broadcast(message, src=0)
    print("the message is:", message)

    print("before wrapping")
    model = nn.Linear(2, 2)
    model.to(device)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])

    print("after wrapping")
    torch.distributed.barrier()

    ddp_model(torch.rand(5, 2).to(device))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    run(args)
