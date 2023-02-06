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
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from lightning_utilities.core.imports import module_available
from torch import Tensor
from torch.nn import Module

import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import ReduceOp
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

_BAGUA_AVAILABLE = module_available("bagua.torch_api")

if _BAGUA_AVAILABLE:
    import bagua.torch_api as bagua
    from bagua.torch_api.algorithms import Algorithm
    from bagua.torch_api.algorithms.q_adam import QAdamOptimizer
    from bagua.torch_api.communication import allreduce_inplace, barrier, broadcast_object, is_initialized
    from bagua.torch_api.communication import ReduceOp as BaguaReduceOp
    from bagua.torch_api.data_parallel.distributed import DistributedDataParallel_V1_9_0 as BaguaDistributedDataParallel

    # Convert a reduce op to its equivalent `bagua.torch_api.ReduceOp`
    _bagua_reduce_ops = {
        ReduceOp.SUM: BaguaReduceOp.SUM,
        ReduceOp.PRODUCT: BaguaReduceOp.PRODUCT,
        ReduceOp.MIN: BaguaReduceOp.MIN,
        ReduceOp.MAX: BaguaReduceOp.MAX,
        ReduceOp.BAND: BaguaReduceOp.BAND,
        ReduceOp.BOR: BaguaReduceOp.BOR,
        ReduceOp.BXOR: BaguaReduceOp.BXOR,
        "avg": BaguaReduceOp.AVG,
        "mean": BaguaReduceOp.AVG,
        "sum": BaguaReduceOp.SUM,
    }
else:
    _bagua_reduce_ops = {}

log = logging.getLogger(__name__)


class LightningBaguaModule(_LightningModuleWrapperBase):
    def __init__(
        self,
        forward_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]] = None,
        pl_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]] = None,
    ) -> None:
        self._validate_init_arguments(pl_module, forward_module)
        forward_module = pl_module or forward_module
        super().__init__(forward_module=forward_module)
        # Bagua use `bagua_module_name` to distinguish different modules
        self._bagua_module_name = f"{forward_module.__class__.__name__}{id(forward_module)}"

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        pl_module = self.lightning_module
        trainer = pl_module._trainer

        if trainer is not None:
            if trainer.training:
                output = self._forward_module.training_step(*inputs, **kwargs)
                # In manual_optimization, we need to prevent DDP reducer as
                # it is done manually in `LightningModule.manual_backward`
                # `require_backward_grad_sync` will be reset in the
                # ddp_strategy `post_training_step` hook
                if not pl_module.automatic_optimization:
                    # Using bagua strategy, the model is redefined in model.inner
                    # and cannot be accessed directly. We need this to make manual
                    # backward work.
                    trainer.model.inner.require_backward_grad_sync = False  # type: ignore[union-attr]
                return output
            else:
                return super().forward(*inputs, **kwargs)
        return self._forward_module(*inputs, **kwargs)


class BaguaStrategy(DDPStrategy):
    strategy_name = "bagua"

    def __init__(
        self,
        algorithm: str = "gradient_allreduce",
        flatten: bool = True,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        **bagua_kwargs: Union[Any, Dict[str, Any]],
    ):
        """Strategy for training using the `Bagua <https://github.com/BaguaSys/bagua>`_ library, with advanced
        distributed training algorithms and system optimizations.

        This strategy requires the `bagua` package to be installed. See
        `installation guide <https://tutorials.baguasys.com/installation>`_ for more information.

        The :class:`BaguaStrategy` is only supported on GPU and on Linux systems.

        Arguments:
            algorithm: Distributed algorithm used to do the actual communication and update. Built-in algorithms
                include "gradient_allreduce", "bytegrad", "decentralized", "low_precision_decentralized", "qadam" and
                "async".
            flatten: Whether to flatten the Bagua communication buckets. The flatten operation will reset data
                pointer of bucket tensors so that they can use faster code paths.
            bagua_kwargs: Additional keyword arguments that will be passed to initialize the Bagua algorithm. More
                details on keyword arguments accepted for each algorithm can be found in the
                `documentation <https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/index.html>`_.
        """
        if not _BAGUA_AVAILABLE:
            raise MisconfigurationException(
                "To use the `BaguaStrategy`, you must have `Bagua` installed. Use `pip install bagua` to install it."
            )

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

        self._bagua_algorithm = algorithm
        self._bagua_flatten = flatten
        self._bagua_kwargs = bagua_kwargs

    def setup_distributed(self) -> None:
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._init_bagua_distributed()

    def _init_bagua_distributed(self) -> None:
        self._set_node_environment_variables()
        log.info(
            "Initializing Bagua Distributed: "
            f"GLOBAL_RANK: {self.global_rank}, "
            f"MEMBER: {self.global_rank + 1}/{self.world_size}"
        )

        # need to set device first before initialize Bagua distributed environment
        # Note: setup_environment calls super().setup_distributed after calling init_distributed()
        torch.cuda.set_device(self.local_rank)

        if not is_initialized():
            bagua.init_process_group()

    def _set_node_environment_variables(self) -> None:
        """Set the environment variables as required by the :func:`bagua.init_process_group` call.

        This enables the use of other cluster environments which don't set these exact variables, e.g., Bagua can be
        launched with ``torch.distributed.run``.
        """
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address  # type: ignore[union-attr]
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)  # type: ignore[union-attr]
        os.environ["RANK"] = str(self.global_rank)
        os.environ["NODE_RANK"] = str(self.node_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def setup(self, trainer: "pl.Trainer") -> None:
        self._rank_0_will_call_children_scripts = self.broadcast(self._rank_0_will_call_children_scripts)
        if self._should_run_deadlock_detection():
            self._share_information_to_prevent_deadlock()

        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        # move the model to the correct device
        self.model_to_device()

        trainer_fn = trainer.state.fn

        if trainer_fn == TrainerFn.FITTING:
            if self._layer_sync and self.model:
                self.model = self._layer_sync.apply(self.model)

        self.setup_precision_plugin()

        if trainer_fn == TrainerFn.FITTING:
            # set up optimizers after the module has been moved to the device
            # but before the module has been wrapped
            self.setup_optimizers(trainer)
            _optimizers_to_device(self.optimizers, self.root_device)

            # skip wrapping the model if we are not fitting as no gradients need to be exchanged
            self._configure_bagua_model(trainer)

    def _check_qadam_optimizer(self) -> None:
        has_qadam_optimizer = any([isinstance(opt, QAdamOptimizer) for opt in self.optimizers])

        if not has_qadam_optimizer or len(self.optimizers) > 1 or len(self.lr_scheduler_configs) > 1:
            raise MisconfigurationException("Bagua QAdam can only accept one QAdamOptimizer and one LR Scheduler.")

        self._bagua_kwargs["q_adam_optimizer"] = self.optimizers[0]

    def _configure_bagua_model(self, trainer: "pl.Trainer") -> None:
        model = LightningBaguaModule(self.model)  # type: ignore[arg-type]
        self.model = self._setup_model(model)

        # start the background communication for async algorithm
        if trainer.training and self._bagua_algorithm == "async":
            self.model.bagua_algorithm.resume(self.model)  # type: ignore

    def _setup_model(self, model: Module) -> "BaguaDistributedDataParallel":
        """Wraps the model into a Bagua distributed module."""

        if self._bagua_algorithm == "qadam":
            self._check_qadam_optimizer()

        algorithm = Algorithm.init(self._bagua_algorithm, **self._bagua_kwargs)
        return BaguaDistributedDataParallel(
            module=model,
            optimizers=self.optimizers,
            algorithm=algorithm,
            gradient_as_bucket_view=self._bagua_flatten,
        )

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def teardown(self) -> None:
        # abort the background communication for async algorithm
        assert self.lightning_module is not None
        if self.lightning_module.trainer.training and self._bagua_algorithm == "async":
            self.model.bagua_algorithm.abort(self.model)  # type: ignore

        if isinstance(self.model, BaguaDistributedDataParallel):
            self.model = self.lightning_module

        super().teardown()

    def barrier(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if is_initialized():
            barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return broadcast_object(obj, src)

    def post_training_step(self) -> None:
        assert self.lightning_module is not None
        # Using bagua strategy, the model is redefined in model.inner
        # and cannot be accessed directly. We need to redefine the
        # post_training_step function to make manual backward work.
        if not self.lightning_module.automatic_optimization:
            self.model.inner.require_backward_grad_sync = True  # type: ignore[union-attr]

    def reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: The tensor to sync and reduce.
            group: The process group to gather results from. Defaults to all processes (world).
            reduce_op: The reduction operation.
                Can also be a string 'sum' or ReduceOp.

        Return:
            The reduced value, except when the input was not a tensor the output remains is unchanged.
        """
        if not isinstance(tensor, Tensor):
            return tensor
        if group is not None:
            raise ValueError("`Bagua` does not support allreduce using a subcommunicator at this time. Unset `group`.")

        if reduce_op is None:
            op = BaguaReduceOp.AVG
        else:
            op = _bagua_reduce_ops.get(reduce_op, None)
            if op is None:
                raise ValueError(f"Unrecognized `reduce_op` for `BaguaStrategy`: {reduce_op}")

        allreduce_inplace(tensor, op=op)
        return tensor
