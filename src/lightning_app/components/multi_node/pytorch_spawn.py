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

from typing import Any, Callable, Type

from typing_extensions import Protocol, runtime_checkable

from lightning_app.components.multi_node.base import MultiNode
from lightning_app.core.queues import MultiProcessQueue
from lightning_app.core.work import LightningWork
from lightning_app.utilities.packaging.cloud_compute import CloudCompute
from lightning_app.utilities.proxies import _proxy_setattr, unwrap, WorkRunExecutor, WorkStateObserver


@runtime_checkable
class _PyTorchSpawnWorkProtocol(Protocol):
    def run(
        self,
        world_size: int,
        node_rank: int,
        global_rank: int,
        local_rank: int,
    ) -> None:
        pass


class _PyTorchSpawnRunExecutor(WorkRunExecutor):
    enable_start_observer: bool = False

    def __call__(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ):
        import torch

        with self.enable_spawn():
            nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
            queue = self.delta_queue if isinstance(self.delta_queue, MultiProcessQueue) else self.delta_queue.to_dict()
            torch.multiprocessing.spawn(
                self.dispatch_run,
                args=(self.__class__, self.work, queue, main_address, main_port, num_nodes, node_rank, nprocs),
                nprocs=nprocs,
            )

    @staticmethod
    def dispatch_run(local_rank, cls, work, delta_queue, *args, **kwargs):
        if local_rank == 0:
            if isinstance(delta_queue, dict):
                delta_queue = cls.process_queue(delta_queue)
                work._request_queue = cls.process_queue(work._request_queue)
                work._response_queue = cls.process_queue(work._response_queue)

            state_observer = WorkStateObserver(work, delta_queue=delta_queue)
            state_observer.start()
            _proxy_setattr(work, delta_queue, state_observer)

        cls.run(local_rank, unwrap(work.run), *args, **kwargs)

        if local_rank == 0:
            state_observer.join(0)

    @staticmethod
    def run(
        local_rank: int,
        work_run: Callable,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
        nprocs: int,
    ):

        import torch

        # 1. Setting distributed environment
        global_rank = local_rank + node_rank * nprocs
        world_size = num_nodes * nprocs

        if torch.distributed.is_available():
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    "nccl" if torch.cuda.is_available() else "gloo",
                    rank=global_rank,
                    world_size=world_size,
                    init_method=f"tcp://{main_address}:{main_port}",
                )
        elif world_size > 1:
            raise Exception("Torch distributed should be available.")

        return work_run(world_size, node_rank, global_rank, local_rank)


class PyTorchSpawnMultiNode(MultiNode):
    def __init__(
        self,
        work_cls: Type["LightningWork"],
        cloud_compute: "CloudCompute",
        num_nodes: int,
        *work_args: Any,
        **work_kwargs: Any,
    ) -> None:
        assert issubclass(work_cls, _PyTorchSpawnWorkProtocol)

        # Note: Private way to modify the work run executor
        # Probably exposed to the users in the future if needed.
        work_cls._run_executor_cls = _PyTorchSpawnRunExecutor

        super().__init__(work_cls, num_nodes, cloud_compute, *work_args, **work_kwargs)
