import torch

import lightning as L
from lightning.app.components import LiteMultiNode
from lightning.lite import LightningLite


class LitePyTorchDistributed(L.LightningWork):

    # Note: Only staticmethod are support for now with `LiteMultiNode`
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


app = L.LightningApp(
    LiteMultiNode(
        LitePyTorchDistributed,
        cloud_compute=L.CloudCompute("gpu-fast-multi"),  # 4 x V100,
        num_nodes=2,
        precision="bf16",
    )
)
