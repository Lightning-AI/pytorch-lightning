# !pip install torch
import lightning as L
from lightning.app.components import MultiNode


class MultiNodeComponent(L.LightningWork):
    def run(
        self,
        main_address: str,
        main_port: int,
        node_rank: int,
        world_size: int,
    ):
        print(f"ADD YOUR DISTRIBUTED CODE: {main_address=} {main_port=} {node_rank=} {world_size=}")
        print("supports ANY ML library")










# gpu-multi-fast has 4 GPUs x 8 nodes = 32 GPUs
component = MultiNodeComponent(cloud_compute=L.CloudCompute("gpu-multi-fast"))
component = MultiNode(component, nodes=8)
app = L.LightningApp(component)
