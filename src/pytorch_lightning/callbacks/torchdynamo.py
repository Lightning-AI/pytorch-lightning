from functools import partial, wraps
from typing import Any, Callable, ContextManager, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loops.optimization.optimizer_loop import Closure
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.imports import _RequirementAvailable

_TORCHDYNAMO_AVAILABLE = _RequirementAvailable("torchdynamo")
_BACKEND = Union[str, Callable]


class TorchDynamo(Callback):
    """The TorchDynamo callback enables ``torchdynamo``'s optimizations.

    Args:
        backend: A backend is either a function/callable taking a :class:`torch.fx.GraphModule` and
            ``example_inputs`` and returning a callable. Or, a string. This argument accepts a backend or a dictionary
            that maps training stages to backends. Backends may require installing additional packages.

    Raises:
        ModuleNotFoundError:
            if ``torchdynamo`` is not installed.
        ValueError:
            If an invalid string backend or invalid stage is passed.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import TorchDynamo

        # defaults to always use `"nvfuser"`
        trainer = Trainer(callbacks=TorchDynamo())

        # custom backend per stage
        trainer = Trainer(callbacks=TorchDynamo({"train": "nvfuser", "predict": "fx2trt"})
    """

    def __init__(self, backend: Union[_BACKEND, Dict[str, _BACKEND]] = "nvfuser") -> None:
        if not _TORCHDYNAMO_AVAILABLE:
            raise ModuleNotFoundError(_TORCHDYNAMO_AVAILABLE.message)

        if isinstance(backend, str):
            _check_valid_backend(backend)
        elif isinstance(backend, dict):
            for stage, backend in backend.items():
                if stage not in list(RunningStage):
                    stages = [stage.value for stage in list(RunningStage)]
                    raise ValueError(f"The stage {stage!r} should be one of {stages}")
                if isinstance(backend, str):
                    _check_valid_backend(backend)
        self.backend = backend

        self.previous_closure_cls = Closure
        self.previous_training_step: Optional[Callable] = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins.

        NotImplementedError:
            If run in a distributed environment.
        """
        if trainer._accelerator_connector.is_distributed:
            raise NotImplementedError(
                f"`TorchDynamo` does not support the {type(trainer.strategy).__name__!r} at the moment."
            )

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        from torchdynamo import optimize

        backend = self.backend if isinstance(self.backend, str) else self.backend["train"]
        optimize_ctx_manager = optimize(backend)

        if pl_module.automatic_optimization:
            optimize_closure = partial(_ContextManagerClosure, optimize_ctx_manager)
            self.previous_closure_cls = trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.closure_cls
            trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.closure_cls = optimize_closure
        else:
            self.previous_training_step = pl_module.training_step
            pl_module.training_step = _torchdynamo_training_step(pl_module.training_step, optimize_ctx_manager)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.automatic_optimization:
            trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.closure_cls = self.previous_closure_cls
        elif self.previous_training_step is not None:
            # technically not necessary thanks to `functools.wraps`
            pl_module.training_step = self.previous_training_step

    # FIXME: add support for other stages


def _check_valid_backend(backend: str):
    from torchdynamo import list_backends

    backends = list_backends()
    if backend not in backends:
        raise ValueError(f"TorchDynamo's backend {backend!r} must be one of {backends}")


class _ContextManagerClosure(Closure):
    def __init__(self, context_manager: ContextManager, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.context_manager = context_manager

    def closure(self, *args: Any, **kwargs: Any) -> Any:
        with self.context_manager:
            return super().closure(*args, **kwargs)


def _torchdynamo_training_step(training_step: Callable, context_manager: ContextManager) -> Callable:
    @wraps(training_step)
    def wrapped(self, *args: Any, **kwargs: Any) -> Any:
        with context_manager:
            return training_step(self, *args, **kwargs)

    return wrapped
