import lightning as L
from lightning.components import MultiNode


class PyTorchComponent(L.LightningWork):
    def run(
        self,
        master_address: str,
        master_port: int,
        node_rank: int,
    ):
        print("YOUR DISTRIBUTED CODE")


compute = L.CloudCompute("gpu")
app = L.LightningApp(MultiNode(PyTorchComponent, nodes=10, cloud_compute=compute))
