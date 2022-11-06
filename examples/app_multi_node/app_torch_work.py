import torch
from torch.nn.parallel.distributed import DistributedDataParallel

import lightning as L
from lightning.app.components import MultiNode


def distributed_train(local_rank: int, main_address: str, main_port: int, num_nodes: int, node_rank: int, nprocs: int):
    # 1. Setting distributed environment
    global_rank = local_rank + node_rank * nprocs
    world_size = num_nodes * nprocs

    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=global_rank,
            world_size=world_size,
            init_method=f"tcp://{main_address}:{main_port}",
        )

    # 2. Prepare distributed model
    model = torch.nn.Linear(32, 2)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    device_ids = device if torch.cuda.is_available() else None
    model = DistributedDataParallel(model, device_ids=device_ids).to(device)

    # 3. Prepare loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 4. Train the model for 50 steps.
    for step in range(50):
        model.zero_grad()
        x = torch.randn(64, 32).to(device)
        output = model(x)
        loss = criterion(output, torch.ones_like(output))
        print(f"global_rank: {global_rank} step: {step} loss: {loss}")
        loss.backward()
        optimizer.step()

    # 5. Verify all processes have the same weights at the end of training.
    weight = model.module.weight.clone()
    torch.distributed.all_reduce(weight)
    assert torch.equal(model.module.weight, weight / world_size)

    print("Multi Node Distributed Training Done!")


class PyTorchDistributed(L.LightningWork):
    def run(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ):
        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        torch.multiprocessing.spawn(
            distributed_train, args=(main_address, main_port, num_nodes, node_rank, nprocs), nprocs=nprocs
        )


compute = L.CloudCompute("gpu-fast-multi")  # 4xV100
app = L.LightningApp(
    MultiNode(
        PyTorchDistributed,
        num_nodes=2,
        cloud_compute=compute,
    )
)
