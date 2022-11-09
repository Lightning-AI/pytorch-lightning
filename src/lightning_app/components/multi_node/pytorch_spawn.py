from typing import Any, Type

from typing_extensions import Protocol, runtime_checkable

from lightning_app.components.multi_node.base import MultiNode
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

        # Remove the wrapper.
        setattr_fn = self.work._setattr_replacement
        self.work._setattr_replacement = None

        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        torch.multiprocessing.spawn(
            self.run,
            args=(self.work, self.delta_queue, main_address, main_port, num_nodes, node_rank, nprocs),
            nprocs=nprocs,
        )

        # Re-attach the wrapper.
        self.work._setattr_replacement = setattr_fn

    @staticmethod
    def run(
        local_rank: int,
        work: "LightningWork",
        delta_queue,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
        nprocs: int,
    ):
        if local_rank == 0:
            state_observer = WorkStateObserver(work, delta_queue=delta_queue)
            state_observer.start()
            _proxy_setattr(work, delta_queue, state_observer)
            pass

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

        unwrap(work.run)(world_size, node_rank, global_rank, local_rank)

        if local_rank == 0:
            state_observer.join(0)


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
