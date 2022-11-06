import lightning as L
from lightning.app.components import MultiNode


class AnyDistributedComponent(L.LightningWork):
    def run(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ):
        print(f"ADD YOUR DISTRIBUTED CODE: {main_address} {main_port} {num_nodes} {node_rank}.")


compute = L.CloudCompute("gpu")
app = L.LightningApp(
    MultiNode(
        AnyDistributedComponent,
        num_nodes=2,
        cloud_compute=compute,
    )
)
