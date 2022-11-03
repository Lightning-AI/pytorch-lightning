from typing import Type

import lightning as L
from lightning_app import structures
from lightning_app.core.flow import LightningFlow


class MultiNode(LightningFlow):
    def __init__(
        self,
        work_cls: Type["L.LightningWork"],
        nodes: int,
        cloud_compute: "L.CloudCompute",
    ):
        """This component enables performing distributed multi-node multi-device training.

        Example::

            import torch

            import lightning as L
            from lightning.components import MultiNode

            class PyTorchComponent(L.LightningWork):

                def run(
                        self,
                        master_address: str,
                        master_port: int,
                        node_rank: int,
                    ):
                        print("ADD YOUR DISTRIBUTED CODE")


            compute = L.CloudCompute('gpu')
            app = L.LightningApp(MultiNode(PyTorchComponent, nodes=10, cloud_compute=compute))

        Arguments:
            work_cls: The work to be executed
            nodes: Number of nodes.
            cloud_compute: The cloud compute object used in the cloud.
        """
        super().__init__()
        self.ws = structures.List()
        self._work_cls = work_cls
        self.nodes = nodes
        self.cloud_compute = cloud_compute
        self.has_initialized = False

    def run(self):
        # 1. Create & start the works
        if not self.has_initialized:
            for _ in range(self.nodes):
                work = self._work_cls(cloud_compute=self.cloud_compute)
                self.ws.append(work)
                work.start()
            self.has_initialized = True

        # 2. Run the user code
        if all(w.internal_ip for w in self.ws):
            for node_rank in range(self.nodes):
                self.ws[node_rank].run(
                    master_address=self.ws[node_rank].internal_ip,
                    master_port=self.ws[node_rank].port,
                    node_rank=node_rank,
                )

                # 3. Stop the machine when finished.
                if self.ws[node_rank].has_succceded:
                    self.ws[node_rank].stop()
