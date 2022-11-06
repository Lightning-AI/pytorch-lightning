import torch

import lightning.app as L
from lightning.app.components import MultiNode


def distributed_function(rank: int, main_address: str, main_port: int, nodes: int, node_rank: int, nprocs: int):
    global_rank = rank + node_rank * nprocs
    world_size = nodes * nprocs

    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=global_rank,
            world_size=world_size,
            init_method=f"tcp://{main_address}:{main_port}",
        )

    gathered = [torch.zeros(1) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, torch.tensor([global_rank]).float())
    print(f"Global Rank {global_rank}: {gathered}")


class PyTorchMultiNode(L.LightningWork):
    def run(
        self,
        main_address: str,
        main_port: int,
        nodes: int,
        node_rank: int,
    ):
        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 2
        torch.multiprocessing.spawn(
            distributed_function, args=(main_address, main_port, nodes, node_rank, nprocs), nprocs=nprocs
        )


compute = L.CloudCompute("gpu")
app = L.LightningApp(
    MultiNode(
        PyTorchMultiNode,
        nodes=2,
        cloud_compute=compute,
    )
)
