from lightning import LightningApp
from lightning.app.components.training import LightningTrainingComponent
from lightning_app.utilities.packaging.cloud_compute import CloudCompute

app = LightningApp(
    LightningTrainingComponent(
        "train.py",
        num_nodes=2,
        cloud_compute=CloudCompute("gpu"),
    ),
)
