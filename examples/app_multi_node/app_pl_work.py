import os

import lightning as L
from lightning.app.components import MultiNode
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel


class LoggingBoringModel(BoringModel):
    def training_step(self, batch, batch_idx: int):
        loss = super().training_step(batch, batch_idx)
        self.log("loss", loss["loss"])
        return loss


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

        model = LoggingBoringModel()
        trainer = L.Trainer(
            max_epochs=10,
            devices="auto",
            accelerator="auto",
            num_nodes=nodes,
            strategy="ddp_spawn",  # Only spawn based strategies are supported for now.
            callbacks=[ModelCheckpoint(monitor="loss")],
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
