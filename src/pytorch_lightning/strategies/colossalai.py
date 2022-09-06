import torch
import pytorch_lightning as pl
import contextlib
from typing import Optional, Generator, Any
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins.precision import ColossalAIPrecisionPlugin
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.utilities.imports import _RequirementAvailable
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

_COLOSSALAI_AVAILABLE = _RequirementAvailable("colossalai")
if _COLOSSALAI_AVAILABLE:
    from colossalai.gemini import ChunkManager, GeminiManager
    from colossalai.utils.model.colo_init_context import ColoInitContext
    from colossalai.utils import get_current_device
    from colossalai.nn.parallel import ZeroDDP
    from colossalai.zero import ZeroOptimizer
    from colossalai.tensor import ProcessGroup
    from colossalai.nn.optimizer import CPUAdam, HybridAdam
    from colossalai.logging import get_dist_logger, disable_existing_loggers
    from colossalai.core import global_context as gpc
    from colossalai.context import ParallelMode


class ModelShardedContext(ColoInitContext):
    def _post_init_method(self, module: torch.nn.Module, *args, **kwargs):
        if getattr(module, '_colossalai_module', False) is True:
            return
        super()._post_init_method(module, *args, **kwargs)
        module._colossalai_module = True


class ColossalAIStrategy(DDPStrategy):
    """ColossalAI strategy.
    It only supports single optimizer which must be  `colossalai.nn.optimizer.CPUAdam`_ or
    `colossalai.nn.optimizer.HybridAdam`_ now.
    You must initialize your model in ``configure_sharded_model()``.

    It configures accelerator and precision, and you should not configure them when initializing ``Trainer``.
    CUDA is essential for this strategy. Please make sure CUDA is available.

    Example::

        class GLUETransformer(LightningModule):
            ...
            def configure_sharded_model(self) -> None:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            def on_load_checkpoint(self, checkpoint) -> None:
                if not hasattr(self, 'model'):
                    self.configure_sharded_model()
        trainer = Trainer(..., strategy=ColossalAIStrategy())

    Args:
        use_chunk (bool, optional): Whether to use chunk-based memory management.
            It can speed up training, but slightly more memory will be used. Defaults to True.
        chunk_size (Optional[int], optional): The size of a chunk.
            It will be ignored when ``use_chunk=False``.
            If it's None, a best chunk size will be searched out based on ``chunk_search_range``,
            ``chunk_search_n_grids`` and ``min_chunk_size``.
            Defaults to None.
        enable_distributed_storage (bool, optional): Whether to storage model in a distributed manner.
            It reduces memory from 1 to 1/N, but it may slow down training.
            Defaults to True.
        placement_policy (str, optional): It can be "cpu", "cuda" and "auto".
            If it's "cpu", parameters, gradients and optimizer states will be offloaded to CPU,
            which means min CUDA memory will be used.
            If it's "cuda", they won't be offloaded, which means max CUDA memory will be used. It's the fastest.
            If it's "auto", they are moving dynamically based on CPU and CUDA memory usage.
            It will utilize heterogeneous memory space evenly and well.
            Note that "auto" policy can only work well when no other processes use CUDA during your training.
            Defaults to 'auto'.
        force_outputs_fp32 (bool, optional): Whether to cast outputs to fp32. Defaults to False.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward)
            which will be used by optimizer.
            This argument will be ignored when ``placement_policy`` is not "auto".
            Defaults to 0.0.
        chunk_search_range (int, optional): The range of chunk size to search.
            The actual search range will be from
            ``max(min_chunk_size, max_param_size)`` to ``max(min_chunk_size, max_param_size) + chunk_search_range``.
            Defaults to 64*1024**2.
        chunk_search_n_grids (int, optional): The number of intervals in the search range. Defaults to 1024.
        min_chunk_size (Optional[int], optional): The minimum size for a chunk. Defaults to None.
        initial_scale (float, optional): The initial dynamic loss scale value. Defaults to 2**32.
        min_scale (float, optional): The minimum dynamic loss scaling value. Defaults to 1.
        growth_factor (float, optional): The multiplication factor for increasing loss scale. Defaults to 2.
        backoff_factor (float, optional): The multiplication factor for decreasing loss scale. Defaults to 0.5.
        growth_interval (int, optional):
            The number of steps to increase loss scale when no overflow occurs.
            Defaults to 1000.
        hysteresis (int, optional): The number of overflows before decreasing loss scale. Defaults to 2.
        max_scale (float, optional): The maximum dynamic loss scaling value. Defaults to 2**32.

    .. _colossalai.nn.optimizer.CPUAdam:
        https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.cpu_adam.html
    .. _colossalai.nn.optimizer.HybridAdam:
        https://colossalai.readthedocs.io/en/latest/colossalai/colossalai.nn.optimizer.hybrid_adam.html
    """

    def __init__(
        self,
        use_chunk: bool = True,
        chunk_size: Optional[int] = None,
        enable_distributed_storage: bool = True,
        placement_policy: str = 'auto',
        force_outputs_fp32: bool = False,
        gpu_margin_mem_ratio: float = 0.0,
        chunk_search_range: int = 64 * 1024**2,
        chunk_search_n_grids: int = 1024,
        min_chunk_size: Optional[int] = None,
        initial_scale: float = 2**32,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
    ) -> None:
        if not _COLOSSALAI_AVAILABLE:
            raise MisconfigurationException(
                "To use the `ColossalAIStrategy`, please install `colossalai` first. "
                "Download `colossalai` by consulting https://colossalai.org/download."
            )

        accelerator = CUDAAccelerator()
        precision_plugin = ColossalAIPrecisionPlugin()
        super().__init__(accelerator=accelerator, precision_plugin=precision_plugin)
        self.use_chunk = use_chunk
        self.chunk_size = chunk_size
        self.enable_distributed_storage = enable_distributed_storage
        self.placement_policy = placement_policy
        self.force_outputs_fp32 = force_outputs_fp32
        self.gpu_margin_mem_ratio = gpu_margin_mem_ratio
        self.chunk_size_search_kwargs = {
            'search_range': chunk_search_range,
            'n_grids': chunk_search_n_grids,
            'min_chunk_size': min_chunk_size
        }
        self.amp_kwargs = {
            'initial_scale': initial_scale,
            'min_scale': min_scale,
            'growth_factor': growth_factor,
            'backoff_factor': backoff_factor,
            'growth_interval': growth_interval,
            'hysteresis': hysteresis,
            'max_scale': max_scale
        }
        self._num_nodes = 1
        self._logger = get_dist_logger()

    def setup_distributed(self):
        if not gpc.is_initialized(ParallelMode.GLOBAL):
            disable_existing_loggers()
            gpc.init_global_dist(rank=self.global_rank, world_size=self.world_size, backend='nccl',
                                 host=self.cluster_environment.main_address, port=self.cluster_environment.main_port)
            gpc.set_device(self.local_rank)
        self.process_group = ProcessGroup()

    def model_sharded_context(self) -> Generator:
        """Provide hook to create modules in a distributed aware context. This is useful for when we'd like to
        shard the model instantly, which is useful for extremely large models which can save memory and
        initialization time.

        Returns: Model parallel context.
        """
        return ModelShardedContext()

    def setup_precision_plugin(self) -> None:
        super().setup_precision_plugin()
        is_training = self.lightning_module.trainer and self.lightning_module.trainer.training
        if is_training:
            assert len(self.optimizers) == 1, 'ColossalAIStrategy only supports single Optimizer now.'
            optimizer = self.optimizers[0]
            assert isinstance(optimizer, (CPUAdam, HybridAdam)), \
                'ColossalAIStrategy only supports colossalai.nn.optimizer.CPUAdam and colossalai.nn.optimizer.HybridAdam.'
        pl_module = self.model
        if not hasattr(pl_module, '_colossalai_zero'):
            if self.use_chunk:
                chunk_size = self.chunk_size or ChunkManager.search_chunk_size(
                    self.model, **self.chunk_size_search_kwargs)
            else:
                chunk_size = None
            chunk_manager = ChunkManager(chunk_size, self.process_group, self.enable_distributed_storage,
                                         GeminiManager.get_default_device(self.placement_policy))
            gemini_manager = GeminiManager(self.placement_policy, chunk_manager)
            assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
            model = _LightningModuleWrapperBase(self.model)
            self.model = ZeroDDP(model, gemini_manager, self.force_outputs_fp32)
            pl_module._colossalai_zero = [self.model]
        else:
            self.model = pl_module._colossalai_zero[0]
        if is_training:
            self.optimizers = [ZeroOptimizer(optimizer, self.model,
                                             gpu_margin_mem_ratio=self.gpu_margin_mem_ratio, **self.amp_kwargs)]

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        self.lightning_module._device = self.root_device
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        self.model_to_device()

    @property
    def root_device(self) -> torch.device:
        if self.parallel_devices is not None:
            return self.parallel_devices[self.local_rank]
        return get_current_device()

    def model_to_device(self) -> None:
        pl_module = self.lightning_module
        pl_module._device = self.root_device
        for child in pl_module.modules():
            if child is not pl_module and getattr(child, '_colossalai_module', None) is not True:
                child.to(self.root_device)

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        if isinstance(self.model, ZeroDDP):
            return self.model.module.lightning_module
        return super().lightning_module

    def teardown(self) -> None:
        gpc.destroy()

    def optimizer_step(self, optimizer, opt_idx: int, closure, model=None, **kwargs: Any) -> Any:
        model = model or self.lightning_module
        return self.precision_plugin.optimizer_step(model, optimizer, opt_idx, closure, **kwargs)

    def lightning_module_state_dict(self):
        return self.model.state_dict()

    @property
    def handles_gradient_accumulation(self) -> bool:
        """Whether the plugin handles gradient accumulation internally."""
        return True

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
