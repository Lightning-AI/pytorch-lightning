# app.py
import lightning as L
from lightning.app.components import PyTorchLightningMultiNode
from lightning.pytorch.demos.boring_classes import BoringModel


class LightningTrainerDistributed(L.LightningWork):
    @staticmethod
    def run():
        model = BoringModel()
        trainer = L.Trainer(max_epochs=10, strategy="ddp")
        trainer.fit(model)

# 8 GPU: (2 nodes of 4 x v100)
component = PyTorchLightningMultiNode(
    LightningTrainerDistributed,
    num_nodes=2,
    cloud_compute=L.CloudCompute("gpu-fast-multi"), # 4 x v100
)
app = L.LightningApp(component)
