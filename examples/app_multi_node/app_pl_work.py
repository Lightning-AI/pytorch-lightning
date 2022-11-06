import os

import torch

import lightning.app as L
from lightning.app.components import MultiNode
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel


class PyTorchLightningMultiNode(L.LightningWork):
    def run(
        self,
        main_address: str,
        main_port: int,
        nodes: int,
        node_rank: int,
    ):
        os.environ["MASTER_ADDR"] = main_address
        os.environ["MASTER_PORT"] = str(main_port)
        os.environ["NODE_RANK"] = str(node_rank)

        devices = torch.cuda.device_count() if torch.cuda.is_available() else 2
        model = BoringModel()
        trainer = Trainer(
            max_epochs=10,
            devices=devices,
            accelerator="auto",
            num_nodes=nodes,
            strategy="ddp_spawn",
        )
        trainer.fit(model)


compute = L.CloudCompute("gpu-fast-multi")  # 4xV100
app = L.LightningApp(
    MultiNode(
        PyTorchLightningMultiNode,
        nodes=2,
        cloud_compute=compute,
    )
)
