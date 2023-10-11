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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import lightning.pytorch as pl
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE, _using_pjrt
from lightning.fabric.plugins import XLACheckpointIO
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.strategies.fsdp import _METADATA_FILENAME
from lightning.fabric.strategies.xla_fsdp import (
    _POLICY,
    _POLICY_SET,
    _activation_checkpointing_kwargs,
    _auto_wrap_policy_kwargs,
    _XLAFSDPBackwardSyncControl,
)
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.types import _PATH, ReduceOp
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.plugins.precision import XLAPrecisionPlugin
from lightning.pytorch.strategies.ddp import ParallelStrategy
from lightning.pytorch.strategies.launchers.xla import _XLALauncher
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.strategies.xla import _pod_progress_bar_force_stdout
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader


class XLAFSDPStrategy(ParallelStrategy):
    r"""Strategy for training multiple XLA devices using the
    :func:`torch_xla.distributed.xla_fully_sharded_data_parallel.XlaFullyShardedDataParallel` method.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    For more information check out https://github.com/pytorch/xla/blob/master/docs/fsdp.md

    Args:
        auto_wrap_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`.
            For convenience, this also accepts a set of the layer classes to wrap.
        activation_checkpointing_policy: Used when selecting the modules for
            which you want to enable activation checkpointing. Enabling this can free up a significant amount of memory
            at the cost of speed since activations in these layers need to be recomputed during backpropagation.
            This accepts a set of the layer classes to wrap.

        state_dict_type: The format in which the state of the model and optimizers gets saved into the checkpoint.

            - ``"full"``: The full weights and optimizer states get assembled on rank 0 and saved to a single file.
            - ``"sharded"``: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
              a folder with files for each shard in the host. Note that TPU VM multihost does not have a shared
              filesystem.

        sequential_save: With this enabled, individual ranks consecutively save their state dictionary shards, reducing
            peak system RAM usage, although it elongates the saving process.
        \**kwargs: See available parameters in :class:`torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`.

    """

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]] = None,
        precision_plugin: Optional[XLAPrecisionPlugin] = None,
        auto_wrap_policy: Optional[_POLICY] = None,
        activation_checkpointing_policy: Optional[_POLICY_SET] = None,
        state_dict_type: Literal["full", "sharded"] = "sharded",
        sequential_save: bool = False,
        **kwargs: Any,
    ) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=XLAEnvironment(),
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self._backward_sync_control = _XLAFSDPBackwardSyncControl()

        self._auto_wrap_policy = auto_wrap_policy
        self._activation_checkpointing_policy = activation_checkpointing_policy
        self._fsdp_kwargs = kwargs
        self._state_dict_type = state_dict_type
        self._sequential_save = sequential_save
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

    @property  # type: ignore[override]
    def checkpoint_io(self) -> Union[XLACheckpointIO, _WrappingCheckpointIO]:
        plugin = self._checkpoint_io
        if plugin is not None:
            assert isinstance(plugin, (XLACheckpointIO, _WrappingCheckpointIO))
            return plugin
        return XLACheckpointIO()

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]]) -> None:
        if io is not None and not isinstance(io, (XLACheckpointIO, _WrappingCheckpointIO)):
            raise TypeError(f"The XLA strategy can only work with the `XLACheckpointIO` plugin, found {io}")
        self._checkpoint_io = io

    @property  # type: ignore[override]
    def precision_plugin(self) -> XLAPrecisionPlugin:
        plugin = self._precision_plugin
        if plugin is not None:
            assert isinstance(plugin, XLAPrecisionPlugin)
            return plugin
        return XLAPrecisionPlugin()

    @precision_plugin.setter
    def precision_plugin(self, precision_plugin: Optional[XLAPrecisionPlugin]) -> None:
        if precision_plugin is not None:
            if not isinstance(precision_plugin, XLAPrecisionPlugin):
                raise TypeError(
                    f"The XLAFSDP strategy can only work with the `XLAPrecisionPlugin` plugin, found {precision_plugin}"
                )
            precision_plugin._using_fsdp = True
        self._precision_plugin = precision_plugin

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

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        return self._state_dict_type == "sharded"

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

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator
        self.accelerator.setup(trainer)

        # we set the device so that optimizers can be created with distributed comms.
        assert self.lightning_module is not None
        self.lightning_module._device = self.root_device

        if is_overridden("configure_sharded_model", self.lightning_module):
            # legacy: we don't skip setup with the `configure_model` alternative
            rank_zero_warn(
                "You have overridden `LightningModule.configure_sharded_model` hook. It will assume that all the layers"
                " are already wrapped for sharding and won't wrap the entire model using `XLAFullyShardedDataParallel`",
                category=PossibleUserWarning,
            )
        else:
            assert self.model is not None
            self.model = self._setup_model(self.model)
        self.barrier()

        self.setup_optimizers(trainer)
        _optimizers_to_device(self.optimizers, self.root_device)

        self.setup_precision_plugin()

    def _setup_model(self, model: Module) -> Module:  # type: ignore
        """Wraps the model into a :class:`~torch_xla.distributed.fsdp.XlaFullyShardedDataParallel` module."""
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        kwargs = self._parse_fsdp_kwargs()
        if any(isinstance(mod, XLAFSDP) for mod in model.modules()) and "auto_wrap_policy" in kwargs:
            rank_zero_warn(
                "A XLAFSDP `auto_wrap_policy` is set, but at least one submodule is already wrapped."
                " The policy will be ignored."
            )
            del kwargs["auto_wrap_policy"]
        # XLA FSDP requires that the root is wrapped, even if submodules are already wrapped
        if not isinstance(model, XLAFSDP):
            model = XLAFSDP(module=model, **kwargs)
        return model

    def model_to_device(self) -> None:
        pass

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    def process_dataloader(self, dataloader: object) -> "MpDeviceLoader":
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        if isinstance(dataloader, MpDeviceLoader):
            # dataloader is already wrapped by MpDeviceLoader
            return dataloader

        dataloader = MpDeviceLoader(dataloader, self.root_device)
        # Mimic interface to torch.utils.data.DataLoader
        dataloader.dataset = dataloader._loader.dataset
        dataloader.batch_sampler = getattr(dataloader._loader, "batch_sampler", None)
        return dataloader

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        super().setup_optimizers(trainer)
        for optimizer in self.optimizers:
            if not any(getattr(p, "_is_sharded", False) for group in optimizer.param_groups for p in group["params"]):
                raise ValueError(
                    f"The optimizer {optimizer} does not seem to reference any XLAFSDP parameters. HINT: Make sure to"
                    f" create the optimizer after setting up the model by referencing `self.trainer.model.parameters()`"
                    f" in the `configure_optimizers()` hook."
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
        original_device = tensor.device
        tensor = tensor.to(self.root_device)

        import torch_xla.core.functions as xf
        import torch_xla.core.xla_model as xm

        tensor = xf.all_gather(tensor) if sync_grads else xm.all_gather(tensor)
        tensor = tensor.to(original_device)
        return tensor

    def reduce(
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
            original_device = obj.device
            # XLA distributed requires that the data is on the XLA device
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
            # this will preserve the dtype and device of any tensors
            buffer = io.BytesIO(obj.cpu().byte().numpy())
            obj = torch.load(buffer)
        else:
            obj = obj.to(original_device)

        return obj

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        _pod_progress_bar_force_stdout(self.global_rank)

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        """Save model, optimizer, and other state in the provided checkpoint directory.

        If the user specifies sharded checkpointing, the directory will contain one file per process, with model- and
        optimizer shards stored per file. If the user specifies full checkpointing, the directory will contain a
        consolidated checkpoint combining all of the sharded checkpoints.

        """
        if not self._launched:
            raise NotImplementedError(
                "Saving checkpoints with `XLAFSDPStrategy` is not supported outside of the Trainer's execution"
            )
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `XLAFSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch`."
            )
        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(filepath))
        if path.is_dir() and any(path.iterdir()):
            raise FileExistsError(f"The checkpoint directory already exists and is not empty: {path}")

        # this is the output of `self.lightning_module_state_dict`
        state_dict = checkpoint.pop("state_dict")

        # not needed in ckpt consolidation
        state_dict["optimizer"] = checkpoint.pop("optimizer_states")

        import torch_xla.core.xla_model as xm

        # ensure model parameters are updated
        xm.mark_step()

        parallel_devices = self.parallel_devices
        assert parallel_devices is not None
        if self._sequential_save:
            # each host runs this in parallel, but the ranks in the host run it sequentially
            for rank in range(len(parallel_devices)):
                if rank == self.local_rank:
                    self._save_checkpoint_shard(path, state_dict, storage_options)
                self.barrier(f"wait-for-{rank}-save")
        else:
            self._save_checkpoint_shard(path, state_dict, storage_options)

        if self._state_dict_type == "full":
            ckpt_prefix = str(path / "checkpoint")
            ckpt_suffix = "_rank-*-of-*.pth"
            if len(parallel_devices) != self.world_size:  # multihost
                raise OSError(
                    "Multihost setups do not have a shared filesystem, so the checkpoint shards cannot be consolidated"
                    " into a single checkpoint after saving them. Please switch to"
                    " `XLAFSDPStrategy(state_dict_type='sharded')`. TIP: You can consolidate them manually by getting"
                    " them together into a single directory and running `python -m"
                    f" torch_xla.distributed.fsdp.consolidate_sharded_ckpts --ckpt_prefix {ckpt_prefix!r} --ckpt_suffix"
                    f" {ckpt_suffix!r} --save_path 'path/to/consolidated.ckpt'`."
                )

            from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints

            self.barrier("before_ckpt_consolidation")
            if self.is_global_zero:
                save_path = path.parent / "consolidated.ckpt"
                # save consolidated checkpoint separate to the shards
                consolidate_sharded_model_checkpoints(ckpt_prefix, ckpt_suffix, str(save_path))
                # remove the shards directory
                self.checkpoint_io.remove_checkpoint(path)
                # mv the consolidated checkpoint where the user would expect it
                get_filesystem(save_path).mv(str(save_path), str(path))
            self.barrier("after_ckpt_consolidation")

            # save metadata separately
            if self.global_rank == 0:
                torch.save(checkpoint, path.parent / _METADATA_FILENAME)
        else:
            if self.global_rank == 0:
                torch.save(checkpoint, path / _METADATA_FILENAME)

    def _save_checkpoint_shard(
        self,
        path: Path,
        converted_state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any],
    ) -> None:
        self.checkpoint_io.save_checkpoint(
            converted_state,
            path / f"checkpoint_rank-{self.global_rank:08d}-of-{self.world_size:08d}.pth",
            storage_options=storage_options,
        )

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        if not self._launched:
            raise NotImplementedError(
                "Saving checkpoints with `XLAFSDPStrategy` is not supported outside of the Trainer's execution"
            )
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `XLAFSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`."
            )

        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(checkpoint_path))

        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        if self._state_dict_type == "sharded":
            file = path / f"checkpoint_rank-{self.global_rank:08d}-of-{self.world_size:08d}.pth"
            if not file.is_file():
                raise ValueError(
                    f"The path {str(file)!r} does not point to valid sharded checkpoints. Make sure the path points to"
                    " a directory with XLAFSDP checkpoint shards."
                )
            if self.model is None or not isinstance(self.model, XLAFSDP):
                raise ValueError(f"Could not find a XLAFSDP model in the provided checkpoint state: {type(self.model)}")
            sharded_ckpt = torch.load(file)
            self.model.load_state_dict(sharded_ckpt.pop("model"))

            # Load metadata (anything not a module or optimizer)
            metadata = torch.load(path / _METADATA_FILENAME)
            return metadata

        if self._state_dict_type == "full":
            if not path.is_file():
                raise ValueError(
                    f"The path {str(path)!r} does not point to a valid full checkpoint. Make sure the path points to a"
                    " directory with a full XLAFSDP checkpoint."
                )
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

            assert self.lightning_module is not None
            if any(isinstance(mod, XLAFSDP) for mod in self.lightning_module.modules()):
                raise ValueError(
                    "`XLAFSDPStrategy` does not support loading full model checkpoint"
                    " if the model or any submodules are manually wrapped."
                )

            full_ckpt = torch.load(path)
            if is_overridden("configure_model", self.lightning_module):
                # if the LightningModule only defined the nn.Module in `configure_model` and we "full" load here, it
                # will error out. Warn the user about this
                rank_zero_warn(
                    "You have overridden `LightningModule.configure_model` hook. Full checkpoint loading with XLAFSDP"
                    " requires that the model is fully defined at `__init__`.",
                    category=PossibleUserWarning,
                )
            self.lightning_module.load_state_dict(full_ckpt.pop("model"))

            # Load metadata (anything not a module or optimizer)
            metadata = torch.load(path.parent / _METADATA_FILENAME)
            return metadata
        raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Override to do nothing, FSDP already loaded the states in `load_checkpoint()`
        pass

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("xla_fsdp", cls, description=cls.__name__)

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        assert self.model is not None
        # this format is defined by:
        # https://github.com/pytorch/xla/blob/v2.1.0/torch_xla/distributed/fsdp/state_dict_utils.py#L122-L125
        return {
            "model": self.model.state_dict(),  # type: ignore
            "shard_metadata": self.model.get_shard_metadata(),  # type: ignore[operator]
        }

    def _parse_fsdp_kwargs(self) -> Dict:
        # this needs to be delayed because `self.precision` isn't available at init
        kwargs = self._fsdp_kwargs.copy()
        precision_plugin = self.precision_plugin
        if isinstance(precision_plugin, XLAPrecisionPlugin):
            # the `compute_dtype` will be passed to the `auto_wrapper_callable` automatically, so we don't need to pass
            # it when creating it
            kwargs.setdefault("compute_dtype", precision_plugin._desired_dtype)
        kwargs = _auto_wrap_policy_kwargs(self._auto_wrap_policy, kwargs)
        return _activation_checkpointing_kwargs(self._activation_checkpointing_policy, kwargs)

    def teardown(self) -> None:
        super().teardown()
        self._launched = False  # after the Trainer finishes, we aren't inside the spawned region
