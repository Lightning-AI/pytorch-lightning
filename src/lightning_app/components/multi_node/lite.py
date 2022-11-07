import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Type

from typing_extensions import Protocol, runtime_checkable

from lightning_app.components.multi_node.base import MultiNode
from lightning_app.core.work import LightningWork
from lightning_app.utilities.app_helpers import is_static_method
from lightning_app.utilities.packaging.cloud_compute import CloudCompute
from lightning_app.utilities.proxies import WorkRunExecutor


@runtime_checkable
class LiteProtocol(Protocol):
    @staticmethod
    def run(lite) -> None:
        ...


@dataclass
class LiteRunExecutor(WorkRunExecutor):

    lite: Any

    def __call__(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ) -> None:
        from lightning.lite import LightningLite

        os.environ["MASTER_ADDR"] = main_address
        os.environ["MASTER_PORT"] = str(main_port)
        os.environ["NODE_RANK"] = str(node_rank)

        if self.lite:
            lite: LightningLite = self.lite
        else:
            lite = LightningLite(accelerator="auto", devices="auto", strategy="ddp_spawn", num_nodes=num_nodes)

        lite.launch(function=self.work_run)


class LiteMultiNode(MultiNode):
    def __init__(
        self,
        work_cls: Type["LightningWork"],
        cloud_compute: "CloudCompute",
        num_nodes: int,
        precision: Any,
        *work_args: Any,
        **work_kwargs: Any,
    ) -> None:
        assert issubclass(work_cls, LiteProtocol)
        if not is_static_method(work_cls, "run"):
            raise Exception(f"The provided {work_cls} run method needs to be static for now.")

        from lightning_lite import LightningLite

        lite = LightningLite(
            accelerator="auto",
            devices="auto",
            strategy="ddp_spawn",  # Only spawn based strategies are support for now.
            num_nodes=num_nodes,
            precision=precision,
        )

        super().__init__(
            work_cls,
            *work_args,
            num_nodes=num_nodes if lite else 1,
            cloud_compute=cloud_compute,
            executor_cls=partial(LiteRunExecutor, lite=lite),
            **work_kwargs,
        )
