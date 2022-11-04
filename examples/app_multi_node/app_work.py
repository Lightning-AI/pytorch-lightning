import lightning as L
from lightning.app.components import MultiNode


class AnyDistributedComponent(L.LightningWork):
    def run(
        self,
        master_address: str,
        master_port: int,
        node_rank: int,
    ):
        print(f"ADD YOUR DISTRIBUTED CODE: {master_address} {master_port} {node_rank}")


compute = L.CloudCompute("gpu")
app = L.LightningApp(
    MultiNode(
        AnyDistributedComponent,
        nodes=2,
        cloud_compute=compute,
    )
)
