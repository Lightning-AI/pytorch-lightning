import lightning as L
from lightning.app.components import LightningTrainingComponent
from lightning.app.utilities.packaging.cloud_compute import CloudCompute

app = L.LightningApp(
    LightningTrainingComponent(
        "train.py",
        num_nodes=2,
        cloud_compute=CloudCompute("gpu-fast-multi"),
    ),
)
