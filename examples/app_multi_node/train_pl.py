import lightning as L
from lightning.app.components import PyTorchLightningMultiNode
from lightning.pytorch.demos.boring_classes import BoringModel


class PyTorchLightningDistributed(L.LightningWork):
    @staticmethod
    def run():
        model = BoringModel()
        trainer = L.Trainer(
            max_epochs=10,
            strategy="ddp",
        )
        trainer.fit(model)


# Run over 2 nodes of 4 x V100
app = L.LightningApp(
    PyTorchLightningMultiNode(
        PyTorchLightningDistributed,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu-fast-multi"),  # 4 x V100
    )
)
