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
import io
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.accelerators.xla import _using_pjrt, _XLA_AVAILABLE
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.io.xla import XLACheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import _StrategyRegistry, ParallelStrategy
from lightning.fabric.strategies.fsdp import _apply_filter
from lightning.fabric.strategies.launchers.xla import _XLALauncher
from lightning.fabric.strategies.strategy import _BackwardSyncControl, _validate_keys_for_strict_loading, TBroadcast
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_13, _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.utilities.init import _EmptyInit
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH, Optimizable, ReduceOp

if TYPE_CHECKING and _XLA_AVAILABLE:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader


class XLAFSDPStrategy(ParallelStrategy):
    """Strategy for training multiple XLA devices using the
    :func:`torch_xla.distributed.xla_fully_sharded_data_parallel.XlaFullyShardedDataParallel` method.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    For more information check out https://github.com/pytorch/xla/blob/master/docs/fsdp.md
    """

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        state_dict_type: Literal["full", "sharded"] = "sharded",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=XLAEnvironment(),
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        self._checkpoint_io: Optional[CheckpointIO]
        self._backward_sync_control = _XLAFSDPBackwardSyncControl()

        self._fsdp_kwargs = kwargs
        self._state_dict_type = state_dict_type
        self._launched = False

    @property
    def root_device(self) -> torch.device:
        if not self._launched:
            raise RuntimeError("Accessing the XLA device before processes have spawned is not allowed.")
        import torch_xla.core.xla_model as xm

        return xm.xla_device()

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = XLACheckpointIO()
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def global_rank(self) -> int:
        return super().global_rank if self._launched else 0

    @property
    def local_rank(self) -> int:
        return super().local_rank if self._launched else 0

    @property
    def node_rank(self) -> int:
        return super().node_rank if self._launched else 0

    @property
    def world_size(self) -> int:
        return super().world_size if self._launched else 1

    def _configure_launcher(self) -> None:
        self._launcher = _XLALauncher(self)

    def setup_environment(self) -> None:
        assert self.parallel_devices is not None
        if _using_pjrt() and len(self.parallel_devices) == 1:
            # spawning only 1 device with PjRT is not supported:
            # https://github.com/Lightning-AI/lightning/pull/17408#discussion_r1170671732
            raise NotImplementedError(
                f"The {type(self).__name__} does not support running on a single device with the PjRT runtime."
                " Try using all devices or the `SingleDeviceXLAStrategy` strategy"
            )

        self._launched = True
        rank_zero_only.rank = self.global_rank
        super().setup_environment()

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple[Module, List[Optimizer]]:
        """Returns NotImplementedError since for XLAFSDP optimizer setup must happen after module setup."""
        raise NotImplementedError(
            f"The `{type(self).__name__}` does not support the joint setup of module and optimizer(s)."
            " Please do it in this order: Create the model, call `setup_module`, create the optimizer,"
            " call `setup_optimizer`."
        )

    def setup_module(self, module: Module) -> Module:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        if any(isinstance(mod, XLAFSDP) for mod in module.modules()) and "auto_wrap_policy" in self._fsdp_kwargs:
            rank_zero_warn(
                "A XLAFSDP `auto_wrap_policy` is set, but at least one submodule is already wrapped."
                " The policy will be ignored."
            )
            del self._fsdp_kwargs["auto_wrap_policy"]

        # XLA FSDP requires that the root is wrapped, even if submodules are already wrapped
        module = XLAFSDP(module=module, **self._fsdp_kwargs)

        return module

    def module_to_device(self, module: Module) -> None:
        pass

    @contextmanager
    def module_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        # TODO: Use the meta device and reset parameters after https://github.com/pytorch/pytorch/issues/90465
        # is resolved. For now, the module will get moved to the device in `setup_module`.
        empty_init_context = _EmptyInit(enabled=bool(empty_init)) if _TORCH_GREATER_EQUAL_1_13 else nullcontext()
        with empty_init_context, self.precision.init_context(), self.module_sharded_context():
            yield

    @contextmanager
    def module_sharded_context(self) -> Generator:
        yield

    def process_dataloader(self, dataloader: DataLoader) -> "MpDeviceLoader":
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        if isinstance(dataloader, MpDeviceLoader):
            # dataloader is already wrapped by MpDeviceLoader
            return dataloader

        dataloader = MpDeviceLoader(dataloader, self.root_device)
        # Mimic interface to torch.utils.data.DataLoader
        dataloader.dataset = dataloader._loader.dataset
        dataloader.batch_sampler = getattr(dataloader._loader, "batch_sampler", None)
        return dataloader

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Set up an optimizer for a model wrapped with XLAFSDP.

        This setup method doesn't modify the optimizer or wrap the optimizer. The only thing it currently does is verify
        that the optimizer was created after the model was wrapped with :meth:`setup_module` with a reference to the
        flattened parameters.

        """
        if _TORCH_GREATER_EQUAL_2_0:
            return optimizer

        from torch_xla.distributed.fsdp.xla_flatten_params_wrapper import FlatParameter

        num_groups = len(optimizer.param_groups)
        if num_groups > 1:
            raise ValueError(
                "An optimizer used with an XLAFSDP model does not support multiple param groups."
                f" Found {num_groups} parameter groups."
            )

        if any(isinstance(param, FlatParameter) for param in optimizer.param_groups[0]["params"]):
            return optimizer

        raise ValueError(
            "The optimizer does not seem to reference any XLAFSDP parameters. HINT: Make sure to create the optimizer"
            " after setting up the model."
        )

    def optimizer_step(self, optimizer: Optimizable, **kwargs: Any) -> Any:
        """Overrides default tpu optimizer_step since FSDP should not call `torch_xla.core.xla_model.optimizer_step`.
        Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            **kwargs: Any extra arguments to ``optimizer.step``

        """
        loss = optimizer.step(**kwargs)
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        return loss

    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        """Clip gradients by norm."""
        self.precision.unscale_gradients(optimizer)
        return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)  # type: ignore[operator]

    def clip_gradients_value(self, module: Module, optimizer: Optimizer, clip_val: Union[float, int]) -> None:
        """Clip gradients by value."""
        raise NotImplementedError(
            "XLA's FSDP strategy does not support to clip gradients by value."
            " Consider clipping by norm instead or choose another strategy!"
        )

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Function to gather a tensor from several distributed processes.

        Args:
            tensor: tensor to all-gather.
            group: unused.
            sync_grads: flag that allows users to synchronize gradients for the all-gather operation.
        Return:
            A tensor of shape (world_size, ...)

        """
        if not self._launched:
            return tensor
        if not isinstance(tensor, Tensor):
            raise NotImplementedError(
                f"`{type(self).__name__}.all_gather` is only implemented for tensors. Given {tensor}"
            )
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.device.type != "xla":
            tensor = tensor.to(self.root_device)

        import torch_xla.core.functions as xf
        import torch_xla.core.xla_model as xm

        return xf.all_gather(tensor) if sync_grads else xm.all_gather(tensor)

    def all_reduce(
        self, output: Union[Tensor, Any], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> Tensor:
        if not isinstance(output, Tensor):
            output = torch.tensor(output, device=self.root_device)

        invalid_reduce_op = isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if invalid_reduce_op or invalid_reduce_op_str:
            raise ValueError(
                "Currently, the XLAFSDPStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )
        import torch_xla.core.xla_model as xm

        output = xm.mesh_reduce("reduce", output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        if not self._launched:
            return
        import torch_xla.core.xla_model as xm

        if name is None:
            # `None` is not supported: "TypeError: _xla_rendezvous(): incompatible function arguments"
            name = ""
        xm.rendezvous(name)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not self._launched:
            return obj

        import torch_xla.core.xla_model as xm

        is_tensor = isinstance(obj, Tensor)
        if is_tensor:
            if obj.dim() == 0:
                obj = obj.unsqueeze(0)
            if obj.device.type != "xla":
                obj = obj.to(self.root_device)
        else:
            # support for arbitrary pickle-ables
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            obj = torch.tensor(  # type: ignore[assignment]
                bytearray(buffer.getbuffer()), device=self.root_device, dtype=torch.float
            )

        obj = [obj]
        xm.collective_broadcast(obj, root_ordinal=src)
        obj = obj[0]

        if not is_tensor:
            buffer = io.BytesIO(obj.cpu().byte().numpy())
            obj = torch.load(buffer)

        return obj

    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state in the provided checkpoint directory.

        If the user specifies sharded checkpointing, the directory will contain one file per process, with model- and
        optimizer shards stored per file. If the user specifies full checkpointing, the directory will contain a
        consolidated checkpoint combining all of the sharded checkpoints.

        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `XLAFSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`."
            )
        if storage_options is not None:
            raise TypeError(
                "`XLAFSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `XLAFSDPStrategy` does not use the `CheckpointIO`."
            )
        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = self.broadcast(path)
        if Path(path).is_dir() and os.listdir(path):
            raise FileExistsError(f"The checkpoint directory already exists and is not empty: {path}")
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        modules = [module for module in state.values() if isinstance(module, XLAFSDP)]
        if len(modules) == 0:
            raise ValueError(
                "Could not find a XLAFSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple XLAFSDP modules in the given state. Saving checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )
        rank = self.local_rank
        world_size = self.world_size
        import torch_xla.core.xla_model as xm

        # ensure model parameters are updated
        xm.mark_step()

        converted_state: Dict[str, Any] = {}
        for key, obj in state.items():
            # convert the state
            if isinstance(obj, Module) and isinstance(obj, XLAFSDP):
                converted = obj.state_dict()
                # add shard_metadata to state
                converted_state["shard_metadata"] = obj.get_shard_metadata()
            elif isinstance(obj, Optimizer):
                converted = obj.state_dict()
            else:
                converted = obj
            _apply_filter(key, filter or {}, converted, converted_state)

        self.checkpoint_io.save_checkpoint(
            converted_state,
            os.path.join(path, f"checkpoint_rank-{rank:08d}-of-{world_size:08d}.pth"),
            storage_options=storage_options,
        )

        if self._state_dict_type == "full":
            from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints

            self.barrier("before_ckpt_consolidation")
            if self.is_global_zero:
                consolidate_sharded_model_checkpoints(
                    ckpt_prefix=os.path.join(path, "checkpoint"), ckpt_suffix="_rank-*-of-*.pth"
                )
            self.barrier("after_ckpt_consolidation")
            self.checkpoint_io.remove_checkpoint(
                os.path.join(path, f"checkpoint_rank-{rank:08d}-of-{world_size:08d}.pth")
            )

    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Given a folder, load the contents from a checkpoint and restore the state of the given objects.

        The strategy currently only supports saving and loading sharded checkpoints which are stored in form of a
        directory of multiple files rather than a single file.

        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `FSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`."
            )
        if not state:
            raise ValueError(
                f"Got `XLAFSDPStrategy.load_checkpoint(..., state={state!r})` but a state with at least "
                " a model instance to reload is required. Pass it in like so:"
                " `FSDPStrategy.load_checkpoint(..., state={'model': model, ...})`"
            )

        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = self.broadcast(path)
        if not os.path.isdir(path):
            raise NotImplementedError(
                f"The path `{path}` is a file or does not exist, but the `XLAFSDPStrategy` currently only supports"
                " loading from a checkpoint(s) in a directory."
            )

        if isinstance(state, (Module, Optimizer)):
            raise NotImplementedError(
                "Loading a single module or optimizer object from a checkpoint"
                " is not supported yet with the XLAFSDP strategy."
            )

        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        modules = {key: module for key, module in state.items() if isinstance(module, XLAFSDP)}
        optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
        if self._state_dict_type == "sharded":
            file = os.path.join(path, f"checkpoint_rank-{self.local_rank:08d}-of-{self.world_size:08d}.pth")
            if not Path(file).is_file():
                raise ValueError(
                    f"The path {str(file)!r} does not point to valid sharded checkpoints. Make sure the path points to"
                    " a directory with XLAFSDP checkpoint shards."
                )
            if len(modules) == 0:
                raise ValueError(
                    "Could not find a XLAFSDP model in the provided checkpoint state. Please provide the model as"
                    " part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure"
                    " you set up the model (and optimizers if any) through the strategy before loading the checkpoint."
                )
            if len(modules) > 1:
                raise ValueError(
                    "Found multiple XLAFSDP modules in the given state. Loading checkpoints with FSDP is"
                    " currently limited to a single model per checkpoint. To load multiple models, call the"
                    " load method for each model separately with a different path."
                )

            _, module = list(modules.items())[0]
            sharded_ckpt = torch.load(file)

            module.load_state_dict(sharded_ckpt["model"], strict=strict)
            for opt_key, opt in optimizers.items():
                opt.load_state_dict(sharded_ckpt[opt_key])

            # Load anything leftover from sharded_ckpt
            loaded_metadata_keys = sharded_ckpt.keys() - modules.keys() - optimizers.keys()
            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, loaded_metadata_keys, strict=strict)
            for key in requested_metadata_keys:
                if key in loaded_metadata_keys:
                    state[key] = sharded_ckpt[key]
                    loaded_metadata_keys.remove(key)

            metadata = {}
            if len(loaded_metadata_keys):
                for key in loaded_metadata_keys:
                    metadata[key] = sharded_ckpt[key]

            # remove "shard_metadata" that is loaded in
            if "shard_metadata" in metadata:
                metadata.pop("shard_metadata")

            return metadata

        if self._state_dict_type == "full":
            file = os.path.join(path, "checkpoint_consolidated.pth")
            if not Path(file).is_file():
                raise ValueError(
                    f"The path {str(file)!r} does not point to a valid full checkpoint. Make sure the path points to a"
                    " directory with a full XLAFSDP checkpoint."
                )
            if len(optimizers) > 0 or len(state.keys() - modules.keys() - optimizers.keys()) > 0:
                rank_zero_warn(
                    "Loading a full checkpoint will only load the full model."
                    " The optimizer and any additional metadata are not included."
                )
            if len(modules) > 0:
                raise ValueError(
                    "Found a XLAFSDP model in the provided checkpoint state."
                    " Please provide the model without any XLAFSDP wrapper."
                )
            if "model" not in state or not isinstance(model := state["model"], torch.nn.Module):
                raise NotImplementedError("XLAFSDP only supports a single model instance with 'model' as the key.")
            full_ckpt = torch.load(str(file))
            model.load_state_dict(full_ckpt.pop("model"), strict=strict)
            return full_ckpt

        raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("xla_fsdp", cls, description=cls.__class__.__name__)


class _XLAFSDPBackwardSyncControl(_BackwardSyncControl):
    @contextmanager
    def no_backward_sync(self, module: Module) -> Generator:
        """Blocks gradient synchronization inside the
        :class:`~torch_xla.distributed.fsdp.XlaFullyShardedDataParallel` wrapper."""
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        if not isinstance(module, XLAFSDP):
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{self.__class__.__name__}.no_backward_sync` is wrapped in `XlaFullyShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        with module.no_sync():
            yield
