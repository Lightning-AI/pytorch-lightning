# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, Type

from lightning_app import structures
from lightning_app.core.flow import LightningFlow
from lightning_app.core.work import LightningWork
from lightning_app.utilities.cloud import is_running_in_cloud
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
            num_nodes: Number of nodes. Gets ignored when running locally. Launch the app with --cloud to run on
                multiple cloud machines.
            cloud_compute: The cloud compute object used in the cloud. The value provided here gets ignored when
                running locally.
            work_args: Arguments to be provided to the work on instantiation.
            work_kwargs: Keywords arguments to be provided to the work on instantiation.
        """
        super().__init__()
        if num_nodes > 1 and not is_running_in_cloud():
            warnings.warn(
                f"You set {type(self).__name__}(num_nodes={num_nodes}, ...)` but this app is running locally."
                " We assume you are debugging and will ignore the `num_nodes` argument."
                " To run on multiple nodes in the cloud, launch your app with `--cloud`."
            )
            num_nodes = 1
        self.ws = structures.List(
            *[
                work_cls(
                    *work_args,
                    cloud_compute=cloud_compute.clone(),
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
