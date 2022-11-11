from typing import Any, Type

from lightning_app import structures
from lightning_app.core.flow import LightningFlow
from lightning_app.core.work import LightningWork
from lightning_app.utilities.packaging.cloud_compute import CloudCompute


class MultiNode(LightningFlow):
    def __init__(
        self,
        work_cls: Type["LightningWork"],
        num_nodes: int,
        cloud_compute: "CloudCompute",
        *work_args: Any,
        **work_kwargs: Any,
    ) -> None:
        """This component enables performing distributed multi-node multi-device training.

        Example::

            import torch

            import lightning as L
            from lightning.components import MultiNode

            class AnyDistributedComponent(L.LightningWork):
                def run(
                    self,
                    main_address: str,
                    main_port: int,
                    node_rank: int,
                ):
                    print(f"ADD YOUR DISTRIBUTED CODE: {main_address} {main_port} {node_rank}")


            compute = L.CloudCompute("gpu")
            app = L.LightningApp(
                MultiNode(
                    AnyDistributedComponent,
                    num_nodes=8,
                    cloud_compute=compute,
                )
            )

        Arguments:
            work_cls: The work to be executed
            num_nodes: Number of nodes.
            cloud_compute: The cloud compute object used in the cloud.
            work_args: Arguments to be provided to the work on instantiation.
            work_kwargs: Keywords arguments to be provided to the work on instantiation.
        """
        super().__init__()
        self.ws = structures.List(
            *[
                work_cls(
                    *work_args,
                    cloud_compute=cloud_compute,
                    **work_kwargs,
                    parallel=True,
                )
                for _ in range(num_nodes)
            ]
        )

    def run(self) -> None:
        # 1. Wait for all works to be started !
        if not all(w.internal_ip for w in self.ws):
            return

        # 2. Loop over all node machines
        for node_rank in range(len(self.ws)):

            # 3. Run the user code in a distributed way !
            self.ws[node_rank].run(
                main_address=self.ws[0].internal_ip,
                main_port=self.ws[0].port,
                num_nodes=len(self.ws),
                node_rank=node_rank,
            )

            # 4. Stop the machine when finished.
            if self.ws[node_rank].has_succeeded:
                self.ws[node_rank].stop()
