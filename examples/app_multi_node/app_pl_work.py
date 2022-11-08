import os

import lightning as L
from lightning.app.components import MultiNode
from lightning.pytorch.demos.boring_classes import BoringModel


class PyTorchLightningDistributed(L.LightningWork):
    def run(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ):
        os.environ["MASTER_ADDR"] = main_address
        os.environ["MASTER_PORT"] = str(main_port)
        os.environ["NODE_RANK"] = str(node_rank)

        model = BoringModel()
        trainer = L.Trainer(
            max_epochs=10,
            devices="auto",
            accelerator="auto",
            num_nodes=num_nodes,
            strategy="ddp_spawn",  # Only spawn based strategies are supported for now.
        )
        trainer.fit(model)


compute = L.CloudCompute("gpu-fast-multi")  # 4xV100
app = L.LightningApp(
    MultiNode(
        PyTorchLightningDistributed,
        num_nodes=2,
        cloud_compute=compute,
    )
)
