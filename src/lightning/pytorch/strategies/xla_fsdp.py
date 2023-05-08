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
from typing import Any, Callable, List, Optional

import torch

import lightning.pytorch as pl
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.strategies.xla_fsdp import XLAFSDP
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.types import _PATH, Optimizable
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.pytorch.strategies.xla import XLAStrategy
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import find_shared_parameters, set_shared_parameters
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only

log = logging.getLogger(__name__)


class XLAFSDPStrategy(XLAStrategy):
    """Strategy for training multiple TPU devices using the
    :func:`torch_xla.distributed.xla_fully_sharded_data_parallel.XlaFullyShardedDataParallel` method."""

    strategy_name = "xla_fsdp"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            debug=debug,
        )
        self.kwargs = kwargs

    def _setup_model(self, model: torch.nn.Module) -> XLAFSDP:
        """Wraps the model into a
        :class:`~torch_xla.distributed.xla_fully_sharded_data_parallel.XlaFullyShardedDataParalle` module."""
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        # If model is already wrapped, we need to avoid sending the `auto_wrap_policy`
        assert self.lightning_module is not None
        if "auto_wrap_policy" in self.kwargs and any(
            isinstance(mod, XLAFSDP) for mod in self.lightning_module.modules()
        ):
            del self.kwargs["auto_wrap_policy"]

        log.debug(f"setting up XLA FSDP model with device id: {self.root_device.index}, kwargs: {self.kwargs}")

        wrapped_module = XLAFSDP(
            module=model,
            **self.kwargs,
        )

        return wrapped_module

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = "1"

        assert self.lightning_module
        shared_params = find_shared_parameters(self.lightning_module)
        self.model_to_device()

        # we set the device so that optimizers can be created with distributed comms.
        assert self.lightning_module is not None
        self.lightning_module._device = self.root_device

        # assert isinstance(self.model, pl.LightningModule)
        # self.model = _LightningModuleWrapperBase(self.model)
        if is_overridden("configure_sharded_model", self.lightning_module):
            rank_zero_info(
                "You have overridden `LightningModule.configure_sharded_model` hook. It will assume that all the layers"
                " are already wrapped for sharding and won't wrap the entire model using `FullyShardedDataParallel`."
            )
        else:
            self.model = self._setup_model(self.model)
        self.barrier()

        set_shared_parameters(self.lightning_module, shared_params)
        self.setup_precision_plugin()

        from torch_xla.experimental import pjrt

        pjrt.broadcast_master_param(self.model)

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
            _optimizers_to_device(self.optimizers, self.root_device)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        try:
            super().setup_optimizers(trainer)
        except ValueError as e:
            if "optimizer got an empty parameter list" in str(e):
                raise ValueError(
                    "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the"
                    " optimizer after setting up the model by referencing `self.trainer.model.parameters()` in the"
                    " `configure_optimizers()` hook."
                )
            raise e

    def optimizer_step(
        self,
        optimizer: Optimizable,
        closure: Callable[[], Any],
        model: "pl.LightningModule" = None,
        **kwargs: Any,
    ) -> Any:
        """Overrides default tpu optimizer_step since FSDP should not call
        `torch_xla.core.xla_model.optimizer_step`. Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            **kwargs: Any extra arguments to ``optimizer.step``
        """
        return optimizer.step(closure=closure, **kwargs)

    def setup_distributed(self) -> None:
        from torch_xla.experimental.pjrt import using_pjrt

        assert self.parallel_devices is not None
        if using_pjrt() and len(self.parallel_devices) == 1:
            # spawning only 1 device with PjRT is not supported:
            # https://github.com/Lightning-AI/lightning/pull/17408#discussion_r1170671732
            raise NotImplementedError(
                "The `XLAFSDPStrategy` does not support running on a single device with the PjRT runtime."
                " Try using all devices or the `SingleTPUStrategy` strategy"
            )

        self._launched = True
        rank_zero_only.rank = self.global_rank

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint
        """
        pass

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register(
            "xla_fsdp_debug", cls, description="XLA FSDP strategy with `debug` as True", debug=True
        )
        strategy_registry.register(cls.strategy_name, cls, description=cls.__class__.__name__)
