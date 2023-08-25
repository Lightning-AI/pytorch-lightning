# app.py
from lightning.app import CloudCompute, LightningApp, LightningWork
from lightning.app.components import LightningTrainerMultiNode
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


class LightningTrainerDistributed(LightningWork):
    def run(self):
        model = BoringModel()
        trainer = Trainer(max_epochs=10, strategy="ddp")
        trainer.fit(model)


# 8 GPUs: (2 nodes of 4 x v100)
component = LightningTrainerMultiNode(
    LightningTrainerDistributed,
    num_nodes=2,
    cloud_compute=CloudCompute("gpu-fast-multi"),  # 4 x v100
)
app = LightningApp(component)
