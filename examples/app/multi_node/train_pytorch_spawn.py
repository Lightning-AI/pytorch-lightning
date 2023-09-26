import torch
from lightning.app import CloudCompute, LightningApp, LightningWork
from lightning.app.components import PyTorchSpawnMultiNode
from torch.nn.parallel.distributed import DistributedDataParallel


class PyTorchDistributed(LightningWork):
    def run(
        self,
        world_size: int,
        node_rank: int,
        global_rank: str,
        local_rank: int,
    ):
        # 1. Prepare the model
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1),
            torch.nn.ReLU(),
            torch.nn.Linear(1, 1),
        )

        # 2. Setup distributed training
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        model = DistributedDataParallel(
            model.to(device), device_ids=[local_rank] if torch.cuda.is_available() else None
        )

        # 3. Prepare loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 4. Train the model for 1000 steps.
        for step in range(1000):
            model.zero_grad()
            x = torch.tensor([0.8]).to(device)
            target = torch.tensor([1.0]).to(device)
            output = model(x)
            loss = criterion(output, target)
            print(f"global_rank: {global_rank} step: {step} loss: {loss}")
            loss.backward()
            optimizer.step()


# 8 GPUs: (2 nodes x 4 v 100)
app = LightningApp(
    PyTorchSpawnMultiNode(
        PyTorchDistributed,
        num_nodes=2,
        cloud_compute=CloudCompute("gpu-fast-multi"),  # 4 x v100
    )
)
