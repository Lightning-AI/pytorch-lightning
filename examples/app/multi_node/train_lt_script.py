from lightning.app import CloudCompute, LightningApp
from lightning.app.components import LightningTrainerScript

# 8 GPUs: (2 nodes of 4 x v100)
app = LightningApp(
    LightningTrainerScript(
        "pl_boring_script.py",
        num_nodes=2,
        cloud_compute=CloudCompute("gpu-fast-multi"),  # 4 x v100
    ),
)
