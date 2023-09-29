# app.py
# ! pip install torch
import torch
from lightning.app import CloudCompute, LightningApp, LightningWork
from lightning.app.components import MultiNode
from torch.nn.parallel.distributed import DistributedDataParallel


def distributed_train(local_rank: int, main_address: str, main_port: int, num_nodes: int, node_rank: int, nprocs: int):
    # 1. SET UP DISTRIBUTED ENVIRONMENT
    global_rank = local_rank + node_rank * nprocs
    world_size = num_nodes * nprocs

    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=global_rank,
            world_size=world_size,
            init_method=f"tcp://{main_address}:{main_port}",
        )

    # 2. PREPARE DISTRIBUTED MODEL
    model = torch.nn.Linear(32, 2)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    model = DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None).to(device)

    # 3. SETUP LOSS AND OPTIMIZER
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 4.TRAIN THE MODEL FOR 50 STEPS
    for step in range(50):
        model.zero_grad()
        x = torch.randn(64, 32).to(device)
        output = model(x)
        loss = criterion(output, torch.ones_like(output))
        print(f"global_rank: {global_rank} step: {step} loss: {loss}")
        loss.backward()
        optimizer.step()

    # 5. VERIFY ALL COPIES OF THE MODEL HAVE THE SAME WEIGTHS AT END OF TRAINING
    weight = model.module.weight.clone()
    torch.distributed.all_reduce(weight)
    assert torch.equal(model.module.weight, weight / world_size)

    print("Multi Node Distributed Training Done!")


class PyTorchDistributed(LightningWork):
    def run(self, main_address: str, main_port: int, num_nodes: int, node_rank: int):
        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        torch.multiprocessing.spawn(
            distributed_train, args=(main_address, main_port, num_nodes, node_rank, nprocs), nprocs=nprocs
        )


# 8 GPUs: (2 nodes x 4 v 100)
compute = CloudCompute("gpu-fast-multi")  # 4 x v100
component = MultiNode(PyTorchDistributed, num_nodes=2, cloud_compute=compute)
app = LightningApp(component)
