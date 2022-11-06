import torch

import lightning as L
from lightning.app.components import MultiNode
from lightning.pytorch.lite import LightningLite


class LitePyTorchDistributed(L.LightningWork):
    @staticmethod
    def run(lite: LightningLite):
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


compute = L.CloudCompute("gpu-fast-multi")  # 4xV100
app = L.LightningApp(
    MultiNode(
        LitePyTorchDistributed,
        num_nodes=2,
        cloud_compute=compute,
        backend="lite",
    )
)
