# Copyright The PyTorch Lightning team.
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
import contextlib
import logging
from typing import Any, Dict, Generator, List, Optional

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.overrides.fairscale import _FAIRSCALE_AVAILABLE
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation
from pytorch_lightning.utilities.types import STEP_OUTPUT

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn import default_auto_wrap_policy, enable_wrap
    from fairscale.nn.data_parallel import FullyShardedDataParallel
else:
    FullyShardedDataParallel = None

log = logging.getLogger(__name__)


class _DDPFullyShardedStrategyModuleWrapper(_LightningModuleWrapperBase):
    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        # this is required because with FSDP lightning_module is empty because weights are sharded.
        # So we need to call self.trainer.model.state_dict (wrapped version) and use this wraper to
        # avoid extra keys `_forward_module.layer.weight.` since we want `layer.weight.` in state_dict.
        return self._forward_module.state_dict(*args, **kwargs)


class DDPFullyShardedStrategy(DDPStrategy):

    strategy_name = "ddp_fully_sharded"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        cpu_offload: bool = False,
        flatten_parameters: bool = True,
        reshard_after_forward: bool = True,
        move_grads_to_cpu: Optional[bool] = None,
        fp32_reduce_scatter: Optional[bool] = None,
        compute_dtype: Optional[torch.dtype] = None,
        bucket_cap_mb: int = 25,
        min_num_params: int = 100_000_000,
        state_dict_to_cpu: bool = True,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = None,
    ):
        """Plugin for Fully Sharded Data Parallel provided by FairScale.

        .. warning:: ``DDPFullyShardedStrategy`` is in beta and subject to change.

        Full Sharded Training shards the entire model across all available GPUs, allowing you to scale model
        size, whilst using efficient communication to reduce overhead. In practice, this means we can remain
        at parity with PyTorch DDP, whilst scaling our model sizes dramatically. The technique is similar
        to ZeRO-Stage 3 but has been built for upstreaming to PyTorch.

        For more information
        `check out FairScale's docs <https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html>`__.

        Defaults have been set and options have been exposed, but may require configuration
        based on your level of memory/speed efficiency. We suggest having a look at
        `this PR for more information <https://github.com/facebookresearch/fairscale/pull/413>`__.

        Many of the helpful doc strings below came from the original
        `FairScale documentation <https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html>`__.

        Arguments:
            cpu_offload: Offload FP32 params to CPU. Only usable in precision=16 mode.
                (Default: False).
            move_grads_to_cpu: Moves gradient shards to CPU after reduction.
                Only disable if using CPU based optimizers
                (Default to ``cpu_offload``).
            flatten_parameters: Flattens parameter into single contiguous tensor for speed efficiency
                (Default: True).
            reshard_after_forward: Reshard parameters after the forward pass, which saves memory but slows
                down training. This is only relevant when resharding individual layers.
                (Default: True).
            fp32_reduce_scatter: Reduce-Scatter gradients in FP32. Only relevant in mixed precision
                (Default: None).
            compute_dtype: dtype for full parameters for computation. Default to torch.float32,
                unless using mixed precision, in which case defaults to torch.float16.
                (Default: None).
            bucket_cap_mb: bucket parameters so that gradient reduction
                can potentially overlap with backward computation.
                bucket_cap_mb controls the bucket size in MegaBytes (MB).
                Buckets are sub-divided based on world_size,
                so the max shard size is roughly bucket_cap_mb / world_size.
                Values <= 0 disable bucketing.
                (Default: 25).
            min_num_params: Number of parameters to wrap when using FairScale ``auto_wrap``.
                (Default: 1e8)
            state_dict_to_cpu: Whether to return parameters (returned by :func:`state_dict`) on CPU device.
                If ``False``, this will default to ``compute_device``.
                (Default: True).
        """
        rank_zero_deprecation(
            "PyTorch Lightning's sharded implementation using FairScale has been deprecated in v1.9.0 and will be"
            " removed in v2.0.0. You can try using the `Trainer(strategy='fsdp_native')` instead."
            " The difference is that native FSDP uses PyTorch's implementation and the current strategy uses"
            " FairScale's implementation (which was upstreamed to PyTorch). After removal, `strategy='fsdp'` will use"
            " the native version by default."
        )
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
        )
        self.cpu_offload = cpu_offload
        self.move_grads_to_cpu = move_grads_to_cpu
        self.flatten_parameters = flatten_parameters
        self.reshard_after_forward = reshard_after_forward
        self.fp32_reduce_scatter = fp32_reduce_scatter
        self.compute_dtype = compute_dtype
        self.bucket_cap_mb = bucket_cap_mb
        self.min_num_params = min_num_params
        self.state_dict_device = torch.device("cpu") if state_dict_to_cpu else None
        self._process_group = None

    @property
    def process_group(self) -> Any:
        if self._process_group is None:
            self._process_group = torch.distributed.new_group()
        return self._process_group

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """Returns model state."""
        assert self.model is not None
        return self.model.state_dict()

    def connect(self, model: "pl.LightningModule") -> None:
        """Called by the accelerator to connect the accelerator and the model with this plugin."""
        # TODO: Wait for this issue to resolve and remove this blocker
        # https://github.com/facebookresearch/fairscale/issues/648
        # Also make sure to update the tests
        if not is_overridden("configure_sharded_model", self.lightning_module) and len(list(model.parameters())) == 0:
            assert self.lightning_module is not None
            raise MisconfigurationException(
                f"Using the same instance of model with `trainer.{self.lightning_module.trainer.state.fn}()` is not"
                " supported with Fairscale FSDP auto-wrap. Please reinitialize your `LightningModule` and pass that."
            )

        super().connect(model)

    def setup_distributed(self) -> None:
        if not self.root_device.type == "cuda":
            raise MisconfigurationException(
                "You selected strategy to be `ddp_fully_sharded`, but GPU is not available."
            )
        super().setup_distributed()

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator
        self.accelerator.setup(trainer)

        if trainer.state.fn == TrainerFn.FITTING:
            if self._layer_sync:
                assert self.model
                self.model = self._layer_sync.apply(self.model)

        self.configure_ddp()
        assert isinstance(self.model, pl.LightningModule)
        self.model = _DDPFullyShardedStrategyModuleWrapper(self.model)
        assert self.lightning_module is not None
        if not is_overridden("configure_sharded_model", self.lightning_module):
            self.model = self._setup_model(self.model)
        self.setup_optimizers(self.lightning_module.trainer)
        _optimizers_to_device(self.optimizers, self.root_device)
        self.barrier()

        self.setup_precision_plugin()

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        invalid_params_error = False
        try:
            super().setup_optimizers(trainer)
        except ValueError as e:
            if "optimizer got an empty parameter list" not in str(e):
                raise
            invalid_params_error = True

        if invalid_params_error or any(not _optimizer_has_flat_params(optimizer) for optimizer in self.optimizers):
            raise ValueError(
                "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the"
                " optimizer after setting up the model by referencing `self.trainer.model.parameters()` in the"
                " `configure_optimizers()` hook."
            )

    def _setup_model(self, model: torch.nn.Module) -> FullyShardedDataParallel:
        """Wraps the model into a
        :class:`~fairscale.nn.data_parallel.fully_sharded_data_parallel.FullyShardedDataParallel` module."""
        log.detail(f"setting up `Fairscale FSDP` model with device id: {self.root_device.index}.")

        return FullyShardedDataParallel(
            module=model,
            process_group=self.process_group,
            cpu_offload=self.cpu_offload,
            move_grads_to_cpu=self.move_grads_to_cpu,
            flatten_parameters=self.flatten_parameters,
            mixed_precision=(self.precision_plugin.precision == "16"),
            reshard_after_forward=self.reshard_after_forward,
            fp32_reduce_scatter=self.fp32_reduce_scatter,
            compute_dtype=self.compute_dtype,
            bucket_cap_mb=self.bucket_cap_mb,
            state_dict_device=self.state_dict_device,
        )

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator:
        log.detail(f"{self.__class__.__name__}: entered model_sharded_context.")
        precision = self.precision_plugin.precision

        def wrap_policy(*args: Any, **kwargs: Any) -> Any:
            return default_auto_wrap_policy(*args, **kwargs, min_num_params=self.min_num_params)

        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            auto_wrap_policy=wrap_policy,
            process_group=self.process_group,
            cpu_offload=self.cpu_offload,
            move_grads_to_cpu=self.move_grads_to_cpu,
            flatten_parameters=self.flatten_parameters,
            mixed_precision=(precision == "16"),
            reshard_after_forward=self.reshard_after_forward,
            fp32_reduce_scatter=self.fp32_reduce_scatter,
            compute_dtype=self.compute_dtype,
            bucket_cap_mb=self.bucket_cap_mb,
            state_dict_device=self.state_dict_device,
        ):
            yield

        log.detail(f"{self.__class__.__name__}: exiting model_sharded_context.")

    def configure_ddp(self) -> None:
        log.detail(f"{self.__class__.__name__}: configuring FSDP... (cpu_offload: [{self.cpu_offload}])")
        if not self.cpu_offload:
            # When using CPU Offload, FSDP will manage the CUDA movement for us.
            # Note: this would be problematic for large model (which could not fit in one GPU)
            # as FSDP module.to(device) would first summon all parameters
            # (TODO: need to figure out solution)
            self.model_to_device()

    def model_to_device(self) -> None:
        log.detail(f"{self.__class__.__name__}: moving model to device [{self.root_device}]...")
        # ensure we update the device type in the lightning module
        assert self.lightning_module
        self.lightning_module.to(self.root_device)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # we don't need precision context since casting is done by FSDP
        # read `mixed_precision` docstring here: https://pytorch.org/docs/stable/fsdp.html
        assert self.model is not None
        return self.model(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        return self.model(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        return self.model(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.model is not None
        return self.model(*args, **kwargs)

    def post_training_step(self) -> None:
        pass

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "fsdp", cls, description="Fully sharded training with checkpointing the full state dict."
        )

        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )


def _optimizer_has_flat_params(optimizer: Optimizer) -> bool:
    from fairscale.nn.misc.flatten_params_wrapper import FlatParameter

    return any(isinstance(param, FlatParameter) for param in optimizer.param_groups[0]["params"])
