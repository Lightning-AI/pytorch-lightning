import os

import torch

import lightning as L
from lightning.app.components import MultiNode
from lightning.lite import LightningLite


def distributed_train(lite: LightningLite):
    # 1. Prepare distributed model and optimizer
    model = torch.nn.Linear(32, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = lite.setup(model, optimizer)
    criterion = torch.nn.MSELoss()

    # 2. Train the model for 50 steps.
    for step in range(50):
        model.zero_grad()
        x = torch.randn(64, 32).to(lite.device)
        output = model(x)
        loss = criterion(output, torch.ones_like(output))
        print(f"global_rank: {lite.global_rank} step: {step} loss: {loss}")
        lite.backward(loss)
        optimizer.step()

    # 3. Verify all processes have the same weights at the end of training.
    weight = model.module.weight.clone()
    torch.distributed.all_reduce(weight)
    assert torch.equal(model.module.weight, weight / lite.world_size)

    print("Multi Node Distributed Training Done!")


class PyTorchDistributed(L.LightningWork):
    def run(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ):

        os.environ["MASTER_ADDR"] = main_address
        os.environ["MASTER_PORT"] = str(main_port)
        os.environ["NODE_RANK"] = str(node_rank)

        lite = LightningLite(accelerator="auto", devices="auto", strategy="ddp_spawn", num_nodes=num_nodes)
        lite.launch(function=distributed_train)


compute = L.CloudCompute("gpu-fast-multi")  # 4xV100
app = L.LightningApp(
    MultiNode(
        PyTorchDistributed,
        num_nodes=2,
        cloud_compute=compute,
    )
)
