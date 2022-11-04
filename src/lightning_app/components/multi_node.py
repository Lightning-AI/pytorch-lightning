from typing import Type

import lightning as L
from lightning.app import structures
from lightning.app.core.flow import LightningFlow


class MultiNode(LightningFlow):
    def __init__(
        self,
        work_cls: Type["L.LightningWork"],
        nodes: int,
        cloud_compute: "L.CloudCompute",
        *work_args,
        **work_kwargs,
    ):
        """This component enables performing distributed multi-node multi-device training.

        Example::

            import torch

            import lightning as L
            from lightning.components import MultiNode

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
                    nodes=8,
                    cloud_compute=compute,
                )
            )

        Arguments:
            work_cls: The work to be executed
            nodes: Number of nodes.
            cloud_compute: The cloud compute object used in the cloud.
            work_args: Arguments to be provided to the work on instantiation.
            work_kwargs: Keywords arguments to be provided to the work on instantiation.
        """
        super().__init__()
        self.ws = structures.List()
        self._work_cls = work_cls
        self.nodes = nodes
        self.cloud_compute = cloud_compute
        self._work_args = work_args
        self._work_kwargs = work_kwargs
        self.has_initialized = False

    def run(self):
        # 1. Create & start the works
        if not self.has_initialized:
            for node_rank in range(self.nodes):
                self.ws.append(
                    self._work_cls(
                        *self._work_args,
                        cloud_compute=self.cloud_compute,
                        **self._work_kwargs,
                    )
                )
                # Starting node `node_rank``
                self.ws[-1].start()
            self.has_initialized = True

        # 2. Wait for all machines to be started !
        if all(not w.has_started for w in self.ws):
            return

        # Loop over all node machines
        for node_rank in range(self.nodes):

            # 3. Run the user code in a distributed way !
            self.ws[node_rank].run(
                master_address=self.ws[0].internal_ip,
                master_port=self.ws[0].port,
                node_rank=node_rank,
            )

            # 4. Stop the machine when finished.
            if self.ws[node_rank].has_succeeded:
                self.ws[node_rank].stop()
