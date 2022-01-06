import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import Module

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, unwrap_lightning_module
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.utilities.enums import _StrategyType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _BAGUA_AVAILABLE
from pytorch_lightning.utilities.seed import reset_seed

if _BAGUA_AVAILABLE:
    import bagua.torch_api as bagua
    from bagua.torch_api.algorithms import Algorithm
    from bagua.torch_api.algorithms.q_adam import QAdamOptimizer
    from bagua.torch_api.communication import allreduce_inplace, barrier, broadcast_object, is_initialized
    from bagua.torch_api.communication import ReduceOp as BaguaReduceOp
    from bagua.torch_api.data_parallel.distributed import DistributedDataParallel_V1_9_0 as BaguaDistributedDataParallel
else:
    BaguaReduceOp = None
    BaguaDistributedDataParallel = None

log = logging.getLogger(__name__)


class LightningBaguaModule(_LightningModuleWrapperBase):
    def __init__(self, pl_module: "pl.LightningModule") -> None:
        super().__init__(pl_module)
        # Bagua use `bagua_module_name` to distinguish different modules
        self._bagua_module_name = pl_module._get_name() + str(id(pl_module))


_bagua_reduce_ops: Dict[ReduceOp, BaguaReduceOp] = {}


def from_torch_reduce_op(op: ReduceOp) -> Optional[BaguaReduceOp]:
    """Convert a `torch.distributed.ReduceOp` to its equivalent `bagua.torch_api.ReduceOp`."""
    global _bagua_reduce_ops

    if len(_bagua_reduce_ops) == 0:
        _bagua_reduce_ops.update(
            {
                ReduceOp.SUM: BaguaReduceOp.SUM,
                ReduceOp.PRODUCT: BaguaReduceOp.PRODUCT,
                ReduceOp.MIN: BaguaReduceOp.MIN,
                ReduceOp.MAX: BaguaReduceOp.MAX,
                ReduceOp.BAND: BaguaReduceOp.BAND,
                ReduceOp.BOR: BaguaReduceOp.BOR,
                ReduceOp.BXOR: BaguaReduceOp.BXOR,
            }
        )

    return _bagua_reduce_ops.get(op, None)


class BaguaStrategy(DDPStrategy):
    distributed_backend = _StrategyType.BAGUA

    def __init__(
        self,
        algorithm: str = "gradient_allreduce",
        do_flatten: bool = True,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ):
        """Provides capabilities to run distributed training using the `Bagua <https://github.com/BaguaSys/bagua>`_
        library.

        Arguments:
            algorithm:  Distributed algorithm used to do the actual communication and update. Built-in algorithms
                include "gradient_allreduce", "bytegrad", "decentralized", "low_precision_decentralized", "qadam" and
                "async".
            do_flatten: Whether to flatten the Bagua communication buckets. The flatten operation will reset data
                pointer of bucket tensors so that they can use faster code paths. Default: ``True``.
            kwargs: Additional arguments that will be passed to initialize the Bagua algorithm.
        """

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

        self._bagua_algorithm = algorithm
        self._bagua_do_flatten = do_flatten
        self._bagua_kwargs = kwargs

    def setup_environment(self) -> None:
        # start the other scripts
        if not self.cluster_environment.creates_processes_externally:  # type: ignore[union-attr]
            self._call_children_scripts()

        self.setup_distributed()

    def setup_distributed(self) -> None:
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._init_bagua_distributed()

    def _init_bagua_distributed(self) -> None:

        self._set_node_environment_variables()
        log.info(
            "initializing Bagua distributed: "
            f"GLOBAL_RANK: {self.global_rank}, "
            f"MEMBER: {self.global_rank + 1}/{self.world_size}"
        )

        # need to set device first before initialize Bagua distributed environment
        torch.cuda.set_device(self.local_rank)

        if not is_initialized():
            bagua.init_process_group()

    def _set_node_environment_variables(self) -> None:
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address  # type: ignore[union-attr]
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)  # type: ignore[union-attr]
        os.environ["RANK"] = str(self.global_rank)
        os.environ["NODE_RANK"] = str(self.node_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def _check_qadam_optimizer(self) -> None:

        trainer = self.lightning_module.trainer
        has_qadam_optimizer = any([isinstance(opt, QAdamOptimizer) for opt in trainer.optimizers])

        if not has_qadam_optimizer or len(trainer.optimizers) > 1 or len(trainer.lr_schedulers) > 1:
            raise MisconfigurationException("Bagua QAdam can only accept one QAdamOptimizer and one LR Scheduler.")

    def configure_ddp(self) -> None:
        model = LightningBaguaModule(self.model)  # type: ignore[arg-type]
        self._model = self._setup_model(model)

        # start the background communication for async algorithm
        if self.lightning_module.trainer.training and self._bagua_algorithm == "async":
            self.model.bagua_algorithm.resume(self.model)  # type: ignore

    def _setup_model(self, model: Module) -> BaguaDistributedDataParallel:
        """Wraps the model into a Bagua distributed module."""

        if self._bagua_algorithm == "qadam":
            self._check_qadam_optimizer()
            self._bagua_kwargs["q_adam_optimizer"] = self.optimizers[0]

        algorithm = Algorithm.init(self._bagua_algorithm, **self._bagua_kwargs)
        return BaguaDistributedDataParallel(
            module=model,
            optimizers=self.optimizers,
            algorithm=algorithm,
            gradient_as_bucket_view=self._bagua_do_flatten,
        )

    @property
    def lightning_module(self) -> "pl.LightningModule":
        model = self._model
        if isinstance(model, BaguaDistributedDataParallel):
            model = model.module
        return unwrap_lightning_module(model)  # type: ignore[arg-type]

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register("bagua", cls, description="Default Bagua Plugin")

    def teardown(self) -> None:
        # abort the background communication for async algorithm
        if self.lightning_module.trainer.training and self._bagua_algorithm == "async":
            self.model.bagua_algorithm.abort(self.model)  # type: ignore

        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()

    def barrier(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if is_initialized():
            barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return broadcast_object(obj, src)

    def reduce(
        self, tensor: torch.Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> torch.Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: The tensor to sync and reduce.
            group: The process group to gather results from. Defaults to all processes (world).
            reduce_op: The reduction operation. Defaults to 'mean'.
                Can also be a string 'sum' or ReduceOp.

        Return:
            The reduced value, except when the input was not a tensor the output remains is unchanged.
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if group is not None:
            raise ValueError("Bagua does not support allreduce using a subcommunicator at this time. Unset `group`.")

        if reduce_op is None:
            op = BaguaReduceOp.AVG
        elif isinstance(reduce_op, str):
            if reduce_op.lower() in ("avg", "mean"):
                op = BaguaReduceOp.AVG
            elif reduce_op.lower() == "sum":
                op = BaguaReduceOp.SUM
            else:
                raise ValueError(f"unrecognized `reduce_op`: {reduce_op}")
        elif isinstance(reduce_op, ReduceOp):
            op = from_torch_reduce_op(reduce_op)
            if op is None:
                raise ValueError(f"unrecognized `reduce_op`: {reduce_op}")

        allreduce_inplace(tensor, op=op)
        return tensor
