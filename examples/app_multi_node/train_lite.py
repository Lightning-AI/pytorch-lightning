import torch

import lightning as L
from lightning.app.components import LiteMultiNode
from lightning.lite import LightningLite


class LitePyTorchDistributed(L.LightningWork):
    def run(self):
        # 1. Prepare the model
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1),
            torch.nn.ReLU(),
            torch.nn.Linear(1, 1),
        )

        # 2. Create LightningLite.
        lite = LightningLite(strategy="ddp", precision=16)
        model, optimizer = lite.setup(model, torch.optim.SGD(model.parameters(), lr=0.01))
        criterion = torch.nn.MSELoss()

        # 3. Train the model for 1000 steps.
        for step in range(1000):
            model.zero_grad()
            x = torch.tensor([0.8]).to(lite.device)
            target = torch.tensor([1.0]).to(lite.device)
            output = model(x)
            loss = criterion(output, target)
            print(f"global_rank: {lite.global_rank} step: {step} loss: {loss}")
            lite.backward(loss)
            optimizer.step()


# Run over 2 nodes of 4 x V100
app = L.LightningApp(
    LiteMultiNode(
        LitePyTorchDistributed,
        cloud_compute=L.CloudCompute("gpu-fast-multi"),  # 4 x V100
        num_nodes=2,
    )
)
