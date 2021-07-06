from argparse import Namespace
import torch
import torch.nn as nn
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning.utilities.cli import ArgumentParser


def run(args: Namespace):

    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)

    message_tensor = torch.tensor([local_rank], device=device)
    message_str = f"this is local rank {local_rank}"

    torch.distributed.init_process_group(backend="nccl", world_size=2, rank=local_rank)
    print(f"[{local_rank}] init successful")

    torch.distributed.broadcast(message_tensor, src=0)
    print(f"[{local_rank}] the tensor message is:", message_tensor)

    # UNCOMMENT TO REPRODUCE HANG
    # torch.distributed.broadcast_object_list(message_str, src=0)
    # print(f"[{local_rank}] the string message is:", message_str)

    print(f"[{local_rank}] before wrapping")
    model = nn.Linear(2, 2)
    model.to(device)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])

    print(f"[{local_rank}] barrier")
    torch.distributed.barrier()

    torch.distributed.broadcast_object_list(message_str, src=0)
    print(f"[{local_rank}] the string message is:", message_str)

    print(f"[{local_rank}] before forward")
    ddp_model(torch.rand(5, 2).to(device))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    run(args)
