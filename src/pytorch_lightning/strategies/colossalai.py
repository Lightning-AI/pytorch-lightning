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
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, TYPE_CHECKING, Union

import torch
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing_extensions import OrderedDict

import pytorch_lightning as pl
from lightning_fabric.accelerators.cuda import _patch_cuda_is_available
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.distributed import ReduceOp
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import ColossalAIPrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT

_COLOSSALAI_AVAILABLE = RequirementCache("colossalai")
if TYPE_CHECKING and _COLOSSALAI_AVAILABLE:
    with _patch_cuda_is_available():
        from colossalai.utils.model.colo_init_context import ColoInitContext
else:
    ColoInitContext = Any


class ColossalAIStrategy(DDPStrategy):
    """ColossalAI strategy. It only supports a single optimizer, which must be
    :class:`colossalai.nn.optimizer.CPUAdam` or :class:`colossalai.nn.optimizer.HybridAdam` now. Your model must
    be created in the function ``LightningModule.configure_sharded_model()``. Thus, you should overwrite this function.
    More details can be found in the below example.

    It configures accelerator and precision, and you should not configure them when initializing ``Trainer``.
    CUDA is essential for this strategy. Please make sure CUDA is available.

    Example::

        class GLUETransformer(LightningModule):
            ...
            def configure_sharded_model(self) -> None:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        trainer = Trainer(..., accelerator="gpu", precision=16, strategy="colossalai")

    Args:
        use_chunk: Whether to use chunk-based memory management.
            It can speed up training, but slightly more memory will be used.

        chunk_size: The size of a chunk.
            It will be ignored when ``use_chunk=False``.
            If it's None, a best chunk size will be searched out based on ``chunk_search_range``,
            ``chunk_search_n_grids`` and ``min_chunk_size``.

        enable_distributed_storage: Whether to storage model in a distributed manner.
            It reduces memory from 1 to 1/N, but it may slow down training.

        placement_policy: It can be "cpu", "cuda" and "auto".

            * If it's "cpu", parameters, gradients and optimizer states will be offloaded to CPU,
                which means min CUDA memory will be used.
            * If it's "cuda", they won't be offloaded, which means max CUDA memory will be used. It's the fastest.
            * If it's "auto", they are moving dynamically based on CPU and CUDA memory usage.
                It will utilize heterogeneous memory space evenly and well.
                Note that "auto" policy can only work well when no other processes use CUDA during your training.

        force_outputs_fp32: Whether to cast outputs to fp32.

        gpu_margin_mem_ratio: The ratio of GPU remaining memory (after the first forward-backward)
            which will be used by optimizer.
            This argument will be ignored when ``placement_policy`` is not "auto".

        chunk_search_range: The range of chunk size to search.
            The actual search range will be from
            ``max(min_chunk_size, max_param_size)`` to ``max(min_chunk_size, max_param_size) + chunk_search_range``.

        chunk_search_n_grids: The number of intervals in the search range.

        min_chunk_size: The minimum size for a chunk in bytes.

        initial_scale: The initial dynamic loss scale value.

        min_scale: The minimum dynamic loss scaling value.

        growth_factor: The multiplication factor for increasing loss scale.

        backoff_factor: The multiplication factor for decreasing loss scale.

        growth_interval: The number of steps to increase loss scale when no overflow occurs.

        hysteresis: The number of overflows before decreasing loss scale.

        max_scale: The maximum dynamic loss scaling value.

    .. _colossalai.nn.optimizer.CPUAdam:
        https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.cpu_adam.html

    .. _colossalai.nn.optimizer.HybridAdam:
        https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.hybrid_adam.html

    """

    strategy_name = "colossalai"

    def __init__(
        self,
        use_chunk: bool = True,
        chunk_size: Optional[int] = None,
        enable_distributed_storage: bool = True,
        placement_policy: str = "auto",
        force_outputs_fp32: bool = False,
        gpu_margin_mem_ratio: float = 0.0,
        chunk_search_range: int = 64 * 1024**2,
        chunk_search_n_grids: int = 4096,
        min_chunk_size: int = 32 * 1024**2,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[ColossalAIPrecisionPlugin] = None,
    ) -> None:
        if not _COLOSSALAI_AVAILABLE:
            raise ModuleNotFoundError(
                "To use the `ColossalAIStrategy`, please install `colossalai` first. "
                "Download `colossalai` by consulting `https://colossalai.org/download`."
            )
        with _patch_cuda_is_available():
            from colossalai.logging import get_dist_logger

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

        self.use_chunk = use_chunk
        self.chunk_size = chunk_size
        self.enable_distributed_storage = enable_distributed_storage
        self.placement_policy = placement_policy
        self.force_outputs_fp32 = force_outputs_fp32
        self.gpu_margin_mem_ratio = gpu_margin_mem_ratio
        self.chunk_size_search_kwargs = {
            "search_range": chunk_search_range,
            "n_grids": chunk_search_n_grids,
            "min_chunk_size": min_chunk_size,
        }
        self.amp_kwargs = {
            "initial_scale": initial_scale,
            "min_scale": min_scale,
            "growth_factor": growth_factor,
            "backoff_factor": backoff_factor,
            "growth_interval": growth_interval,
            "hysteresis": hysteresis,
            "max_scale": max_scale,
        }
        self._num_nodes = 1
        self._logger = get_dist_logger()

    @property
    def root_device(self) -> torch.device:
        with _patch_cuda_is_available():
            from colossalai.utils import get_current_device

        if self.parallel_devices is not None:
            return self.parallel_devices[self.local_rank]
        return get_current_device()

    @property
    def handles_gradient_accumulation(self) -> bool:
        """Whether the plugin handles gradient accumulation internally."""
        return True

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        """Override to delay restoring from checkpoint till after pre-dispatch."""
        return True

    def setup_distributed(self) -> None:
        with _patch_cuda_is_available():
            from colossalai.context import ParallelMode
            from colossalai.core import global_context as gpc
            from colossalai.logging import disable_existing_loggers

        assert self.cluster_environment is not None
        self.set_world_ranks()
        if not gpc.is_initialized(ParallelMode.GLOBAL):
            disable_existing_loggers()
            gpc.init_global_dist(
                rank=self.global_rank,
                world_size=self.world_size,
                backend="nccl",
                host=self.cluster_environment.main_address,
                port=self.cluster_environment.main_port,
            )
            gpc.set_device(self.local_rank)

    def model_sharded_context(self) -> "ColoInitContext":
        """Provide hook to create modules in a distributed aware context. This is useful for when we'd like to
        shard the model instantly, which is useful for extremely large models which can save memory and
        initialization time.

        Returns: Model parallel context.
        """
        with _patch_cuda_is_available():
            from colossalai.utils.model.colo_init_context import ColoInitContext

        class ModelShardedContext(ColoInitContext):
            def _post_init_method(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
                if getattr(module, "_colossalai_module", False) is True:
                    return
                super()._post_init_method(module, *args, **kwargs)
                for sub_module in module.modules():
                    sub_module._colossalai_module = True  # type: ignore[assignment]

        return ModelShardedContext()

    def setup_precision_plugin(self) -> None:
        with _patch_cuda_is_available():
            from colossalai.nn.optimizer import CPUAdam, HybridAdam
            from colossalai.zero import ZeroOptimizer

        super().setup_precision_plugin()
        assert self.lightning_module is not None
        is_training = self.lightning_module.trainer and self.lightning_module.trainer.training

        if is_training:
            if len(self.optimizers) > 1:
                raise ValueError("`ColossalAIStrategy` only supports single Optimizer now.")
            optimizer = self.optimizers[0]
            if not isinstance(optimizer, (CPUAdam, HybridAdam)):
                raise ValueError(
                    "`ColossalAIStrategy` only supports `colossalai.nn.optimizer.CPUAdam` "
                    "and `colossalai.nn.optimizer.HybridAdam` as its optimizer."
                )
        assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
        pl_module = self.model

        if not hasattr(pl_module, "_colossalai_zero"):
            with _patch_cuda_is_available():
                from colossalai.nn.parallel import GeminiDDP
                from colossalai.utils import get_current_device
            if not self.use_chunk:
                raise ValueError("`ColossalAIStrategy` must use chunk in versions higher than 0.1.10")
            chunk_search_range: int = self.chunk_size_search_kwargs.get("search_range", 32 * 1024**2)
            search_range_mb: float = chunk_search_range / 1024**2
            search_n_grids: int = self.chunk_size_search_kwargs.get("n_grids", 4096)
            search_interval: int = math.ceil(chunk_search_range / search_n_grids)
            min_chunk_size_mb = int(self.chunk_size_search_kwargs["min_chunk_size"] // (1024**2))

            model = _LightningModuleWrapperBase(self.model)
            self.model = GeminiDDP(
                module=model,
                device=get_current_device(),
                placement_policy=self.placement_policy,
                pin_memory=True,
                force_outputs_fp32=self.force_outputs_fp32,
                search_range_mb=search_range_mb,
                hidden_dim=search_interval,
                min_chunk_size_mb=min_chunk_size_mb,
            )

            assert self.model is not None
            pl_module._colossalai_zero = [self.model]  # type: ignore[assignment]
        else:
            self.model = pl_module._colossalai_zero[0]  # type: ignore[index, assignment]
        if is_training:
            self.optimizers = [
                ZeroOptimizer(optimizer, self.model, gpu_margin_mem_ratio=self.gpu_margin_mem_ratio, **self.amp_kwargs)
            ]

    def setup(self, trainer: "pl.Trainer") -> None:
        precision = self.precision_plugin.precision
        if precision != "16":
            raise ValueError(
                f"`Trainer(strategy='colossalai', precision={precision!r})` is not supported."
                " Consider setting `precision=16`."
            )

        if not isinstance(self.accelerator, CUDAAccelerator):
            raise ValueError(
                "`ColossalAIStrategy` is only supported on `CUDAAccelerator`, "
                f"but `{self.accelerator.__class__.__name__}` is used."
            )

        if trainer.state.fn == TrainerFn.FITTING:
            if is_overridden("backward", trainer.lightning_module):
                rank_zero_warn(
                    "You have overridden the `LightningModule.backward` hook"
                    " but it will be ignored since ColossalAI handles"
                    " the backward logic internally."
                )

            if trainer.accumulate_grad_batches > 1:
                raise ValueError(
                    "ColossalAI does not support gradient accumulation now. Please set `accumulate_grad_batches` to 1."
                )

            accumulation_scheduler = trainer.accumulation_scheduler
            if accumulation_scheduler.epochs != [0]:
                raise ValueError(
                    "ColossalAI currently does not support different `accumulate_grad_batches` at different epochs."
                )

        if not isinstance(self.precision_plugin, ColossalAIPrecisionPlugin):
            raise ValueError("`ColossalAIStrategy` is only compatible with `ColossalAIPrecisionPlugin`.")

        self.accelerator.setup(trainer)
        assert self.lightning_module is not None
        self.lightning_module._device = self.root_device
        self.ignore_no_grad_parameters(self.root_device)
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        self.model_to_device()

    def ignore_no_grad_parameters(self, running_device: torch.device) -> None:
        # for those parameters with no gradients
        # we shold ignore them on DDP and move them to CUDA
        assert self.model is not None
        for param in self.model.parameters():
            if not param.requires_grad:
                setattr(param, "_ddp_to_ignore", True)
                param.data = param.data.to(running_device)

    def model_to_device(self) -> None:
        assert self.lightning_module is not None
        pl_module = self.lightning_module
        for child in pl_module.modules():
            if child is not pl_module and not getattr(child, "_colossalai_module", False):
                child.to(self.root_device)

    def teardown(self) -> None:
        optimizers = self.optimizers
        self.optimizers = list()
        zero_model = self.model
        self.model = None
        pl_module = self._lightning_module
        self._lightning_module = None

        super().teardown()

        self.optimizers = optimizers
        self.model = zero_model
        self._lightning_module = pl_module

    def optimizer_step(
        self,
        optimizer: Optimizer,
        opt_idx: int,
        closure: Callable[[], Any],
        model: Optional[Union["pl.LightningModule", Module]] = None,
        **kwargs: Any,
    ) -> Any:
        model = model or self.lightning_module
        # TODO(fabric): remove assertion once strategy's optimizer_step typing is fixed
        assert isinstance(model, pl.LightningModule)
        return self.precision_plugin.optimizer_step(
            optimizer, model=model, optimizer_idx=opt_idx, closure=closure, **kwargs
        )

    def lightning_module_state_dict(self, rank_zero_only: bool = False) -> Dict[str, Any]:
        """Returns a dictionary containing a whole state of the module. But all the tensors in the dictionary are
        detached from their parameters and located in cpu memory.

        Args:
            rank_zero_only: If True, only process rank 0 gets the correct dictionary.
                Otherwise, all processes get the same dictionary.
        """
        with _patch_cuda_is_available():
            from colossalai.nn.parallel import ZeroDDP

        assert isinstance(self.model, ZeroDDP)
        org_dict = self.model.state_dict(only_rank_0=rank_zero_only)

        children = list(self.model.named_children())
        assert len(children) == 1
        prefix, child = children[0]
        prefix += "."
        assert child is self.lightning_module

        mapping_dict = dict()
        for key in org_dict.keys():
            mapping_dict[key] = key.replace(prefix, "")  # remove "_forward_module." from the key

        return {mapping_dict[key]: value for key, value in org_dict.items()}

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        orig_dict = checkpoint["state_dict"]

        assert self.model is not None
        children = list(self.model.named_children())
        assert len(children) == 1
        prefix, child = children[0]
        prefix += "."
        assert child is self.lightning_module

        mapping_dict = dict()
        for key in orig_dict.keys():
            mapping_dict[key] = prefix + key  # add "_forward_module." to the key

        load_dict = OrderedDict({mapping_dict[key]: value for key, value in orig_dict.items()})
        self.model.load_state_dict(load_dict)

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            return self.model(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            return self.model(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            return self.model(*args, **kwargs)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register("colossalai", cls, description="Default ColossalAI Strategy")

    def reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "sum"
    ) -> Tensor:
        with _patch_cuda_is_available():
            from colossalai.communication.collective import reduce
            from colossalai.context import ParallelMode
            from colossalai.core import global_context as gpc

        if not isinstance(tensor, Tensor):
            return tensor

        if isinstance(reduce_op, str):
            if reduce_op.lower() in ("avg", "mean"):
                reduce_op = ReduceOp.SUM
                div_factor = gpc.get_world_size(parallel_mode=ParallelMode.GLOBAL)
                with torch.no_grad():
                    tensor = tensor / div_factor
            else:
                reduce_op = getattr(ReduceOp, reduce_op.upper())

        tensor = reduce(tensor, dst=0, parallel_mode=ParallelMode.GLOBAL, op=reduce_op)
        return tensor

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        """Broadcasts an object to all processes.

        Args:
            obj: the object to broadcast
            src: source rank
        """
        with _patch_cuda_is_available():
            from colossalai.communication.collective import broadcast
            from colossalai.context import ParallelMode
            from colossalai.core import global_context as gpc

        if isinstance(obj, Tensor):
            return broadcast(obj, src=src, parallel_mode=ParallelMode.GLOBAL)
        else:
            obj_list = [obj]
            torch.distributed.broadcast_object_list(obj_list, src, group=gpc.get_group(ParallelMode.GLOBAL))
            return obj_list[0]

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Perform a all_gather on all processes."""
        with _patch_cuda_is_available():
            from colossalai.communication.collective import all_gather
            from colossalai.context import ParallelMode

        assert sync_grads is False
        return all_gather(tensor, dim=0, parallel_mode=ParallelMode.GLOBAL)
