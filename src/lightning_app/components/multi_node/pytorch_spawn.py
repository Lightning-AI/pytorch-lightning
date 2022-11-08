from typing import Any, Callable, Type

from typing_extensions import Protocol, runtime_checkable

from lightning_app.components.multi_node.base import MultiNode
from lightning_app.core.work import LightningWork
from lightning_app.utilities.app_helpers import is_static_method
from lightning_app.utilities.packaging.cloud_compute import CloudCompute
from lightning_app.utilities.proxies import WorkRunExecutor


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
    def __call__(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ):
        import torch

        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        torch.multiprocessing.spawn(
            self.run, args=(self.work_run, main_address, main_port, num_nodes, node_rank, nprocs), nprocs=nprocs
        )

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

        work_run(world_size, node_rank, global_rank, local_rank)


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
        if not is_static_method(work_cls, "run"):
            raise TypeError(
                f"The provided {work_cls} run method needs to be static for now."
                "HINT: Remove `self` and add staticmethod decorator."
            )

        # Note: Private way to modify the work run executor
        # Probably exposed to the users in the future if needed.
        work_cls._run_executor_cls = _PyTorchSpawnRunExecutor

        super().__init__(work_cls, num_nodes, cloud_compute, *work_args, **work_kwargs)
