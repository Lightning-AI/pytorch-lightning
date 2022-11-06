# ! pip install torch
import lightning as L
from lightning.app.components import MultiNode
import torch

class MultiNodePytorchComponent(L.LightningWork):
    def run(
        self,
        main_address: str,
        main_port: int,
        node_rank: int,
        world_size: int,
    ):
        # this machine creates a group of processes and registers to the main node
        print(f"Init process group: {main_address=}, {main_port=}, {world_size=}, {node_rank=}")
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://{main_address}:{main_port}",
            world_size=world_size,
            rank=node_rank
        )
        for step in range(10000):
            gathered = [torch.zeros(1) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, torch.tensor([node_rank]).float())
            print(f'step: {step}, tensor: {gathered}')

# gpu-multi-fast has 4 GPUs x 8 nodes = 32 GPUs
component = MultiNodePytorchComponent(cloud_compute=L.CloudCompute("gpu-multi-fast"))
component = MultiNode(component, nodes=8)
app = L.LightningApp(component)
