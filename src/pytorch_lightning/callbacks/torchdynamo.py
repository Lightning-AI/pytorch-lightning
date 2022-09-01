from functools import partial, update_wrapper, wraps
from typing import Any, Callable, ContextManager, Dict, Optional, TYPE_CHECKING, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loops.optimization.optimizer_loop import Closure
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.imports import _RequirementAvailable, _TORCH_GREATER_EQUAL_1_13

_TORCHDYNAMO_CACHE = _RequirementAvailable("torchdynamo")
_TORCHDYNAMO_AVAILABLE = _TORCH_GREATER_EQUAL_1_13 and bool(_TORCHDYNAMO_CACHE)
_BACKEND = Union[str, Callable]

if TYPE_CHECKING and _TORCHDYNAMO_AVAILABLE:
    from torchdynamo.eval_frame import OptimizeContext
else:
    OptimizeContext = object()


class TorchDynamo(Callback):
    """The ``TorchDynamo`` callback enables ``pytorch/torchdynamo``'s optimizations.

    .. warning:: ``TorchDynamo`` is experimental and under active development.

    Args:
        backend: A backend is either a function/callable taking a :class:`torch.fx.GraphModule` and
            ``example_inputs`` and returning a callable. Or, a string. This argument accepts a backend or a dictionary
            that maps trainer entrypoints to backends. Backends may require installing additional packages.

    Raises:
        ModuleNotFoundError:
            if ``torchdynamo`` is not installed.
        ValueError:
            If an invalid string backend or invalid entrypoint is passed.

    Example::

        from pytorch_lightning.callbacks import TorchDynamo

        # defaults to using `"inductor"`
        callback = TorchDynamo()
        # equivalent to
        callback = TorchDynamo("inductor")
        # custom backend per entrypoint. entrypoint not defined will use the default backend
        callback = TorchDynamo({"fit": "nvfuser", "predict": "fx2trt"})
    """

    def __init__(self, backend: Optional[Union[_BACKEND, Dict[str, _BACKEND]]] = None) -> None:
        if not _TORCHDYNAMO_AVAILABLE:
            raise ModuleNotFoundError(str(_TORCHDYNAMO_CACHE))

        # choose a default backend. inductor seems to be the fastest and most reliable one at this moment
        # https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
        default_backend = "inductor"

        # `OptimizeContext` does not allow changing the backend without resetting the state:
        # https://github.com/pytorch/torchdynamo/blob/11873be7ec66b71cc065ca756bc6b5c292bef820/torchdynamo/eval_frame.py#L182-L187
        # so we either map entrypoints to backends or training stages to stages where we reset the state between fit's
        # training and validation. we choose the former in case there's a large performance impact caused by resetting
        # too often
        entrypoints = [e.value for e in list(TrainerFn)]

        # convert the input to a dictionary
        self.backends: Dict[str, _BACKEND]
        if backend is None:
            backend = default_backend
        if isinstance(backend, str):
            self.backends = dict.fromkeys(entrypoints, backend)
        elif isinstance(backend, dict):
            self.backends = dict.fromkeys(entrypoints, default_backend)
            self.backends.update(backend)
        else:
            raise ValueError(f"Unexpected backend argument: {backend!r}")

        # argument validation
        from torchdynamo import list_backends

        supported_backends = list_backends()
        for key, value in self.backends.items():
            if key not in entrypoints:
                raise ValueError(f"The entrypoint {key!r} should be one of {entrypoints}.")
            if isinstance(value, str) and value not in supported_backends:
                raise ValueError(f"TorchDynamo's backend {backend!r} must be one of {supported_backends}")

        self._previous_closure_cls = Closure
        self._previous_training_step: Optional[Callable] = None
        self._previous_validation_step: Optional[Callable] = None
        self._previous_test_step: Optional[Callable] = None
        self._previous_predict_step: Optional[Callable] = None

    def _optimize_context(self, entrypoint: TrainerFn) -> "OptimizeContext":
        import torchdynamo

        return torchdynamo.optimize(self.backends[entrypoint])

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins.

        NotImplementedError:
            If run in a distributed environment.
        """
        # TODO: maybe this doesn't apply to evaluation and prediction
        # https://github.com/pytorch/torchdynamo/issues/43
        if trainer._accelerator_connector.is_distributed:
            raise NotImplementedError(
                f"`TorchDynamo` does not support the {type(trainer.strategy).__name__!r} at the moment."
            )

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        entrypoint = trainer.state.fn
        assert entrypoint is not None
        optimize_ctx_manager = self._optimize_context(entrypoint)
        if pl_module.automatic_optimization:
            optimize_closure = partial(_ContextManagerClosure, optimize_ctx_manager)
            update_wrapper(optimize_closure, _ContextManagerClosure)
            self._previous_closure_cls = trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.closure_cls
            trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.closure_cls = (
                optimize_closure  # type: ignore[assignment]
            )
        else:
            self._previous_training_step = pl_module.training_step
            pl_module.training_step = _wrap_step(  # type: ignore[assignment]
                pl_module.training_step, optimize_ctx_manager
            )

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.automatic_optimization:
            trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.closure_cls = self._previous_closure_cls
        elif self._previous_training_step is not None:
            pl_module.training_step = self._previous_training_step  # type: ignore[assignment]
            self._previous_training_step = None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            return
        self._previous_validation_step = pl_module.validation_step
        entrypoint = trainer.state.fn
        assert entrypoint is not None
        pl_module.validation_step = _wrap_step(  # type: ignore[assignment]
            pl_module.validation_step, self._optimize_context(entrypoint)
        )

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._previous_validation_step is not None:
            pl_module.validation_step = self._previous_validation_step  # type: ignore[assignment]
            self._previous_validation_step = None

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._previous_test_step = pl_module.test_step
        entrypoint = trainer.state.fn
        assert entrypoint is not None
        pl_module.test_step = _wrap_step(  # type: ignore[assignment]
            pl_module.test_step, self._optimize_context(entrypoint)
        )

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._previous_test_step is not None:
            pl_module.test_step = self._previous_test_step  # type: ignore[assignment]
            self._previous_test_step = None

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._previous_predict_step = pl_module.predict_step
        entrypoint = trainer.state.fn
        assert entrypoint is not None
        pl_module.predict_step = _wrap_step(  # type: ignore[assignment]
            pl_module.predict_step, self._optimize_context(entrypoint)
        )

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._previous_predict_step is not None:
            pl_module.predict_step = self._previous_predict_step  # type: ignore[assignment]
            self._previous_predict_step = None

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        import torchdynamo

        torchdynamo.reset()

        self._previous_closure_cls = Closure
        self._previous_training_step = None
        self._previous_validation_step = None
        self._previous_test_step = None
        self._previous_predict_step = None

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        stage = trainer.state.stage
        assert stage is not None
        self.teardown(trainer, pl_module, stage)


class _ContextManagerClosure(Closure):
    def __init__(self, context_manager: ContextManager, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.context_manager = context_manager

    def closure(self, *args: Any, **kwargs: Any) -> Any:
        with self.context_manager:
            return super().closure(*args, **kwargs)


def _wrap_step(method: Callable, context_manager: ContextManager) -> Callable:
    @wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with context_manager:
            return method(*args, **kwargs)

    return wrapped
