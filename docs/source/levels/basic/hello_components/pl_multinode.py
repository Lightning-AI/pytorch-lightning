# app.py
import lightning as L
from lightning.app.components import LightningTrainerMultiNode
from lightning.pytorch.demos.boring_classes import BoringModel


class LightningTrainerDistributed(L.LightningWork):
    def run(self):
        model = BoringModel()
        trainer = L.Trainer(max_epochs=10, strategy="ddp")
        trainer.fit(model)

# 8 GPUs: (2 nodes of 4 x v100)
component = LightningTrainerMultiNode(
    LightningTrainerDistributed,
    num_nodes=4,
    cloud_compute=L.CloudCompute("gpu-fast-multi"), # 4 x v100
)
app = L.LightningApp(component)
