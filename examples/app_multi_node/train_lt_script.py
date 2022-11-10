import lightning as L
from lightning.app.components import LightningTrainerScript
from lightning.app.utilities.packaging.cloud_compute import CloudCompute

# Run over 2 nodes of 4 x V100
app = L.LightningApp(
    LightningTrainerScript(
        "pl_boring_script.py",
        num_nodes=2,
        cloud_compute=CloudCompute("gpu-fast-multi"),
    ),
)
