import lightning as L
from lightning.app.components import MultiNode


class PyTorchComponent(L.LightningWork):
    def run(
        self,
        master_address: str,
        master_port: int,
        node_rank: int,
    ):
        print(master_address, master_port, node_rank)
        print("YOUR DISTRIBUTED CODE")


compute = L.CloudCompute("gpu")
app = L.LightningApp(MultiNode(PyTorchComponent, nodes=2, cloud_compute=compute), debug=True)
