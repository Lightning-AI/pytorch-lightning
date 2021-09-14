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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.optimization.closure import AbstractClosure, OutputResult
from pytorch_lightning.loops.utilities import (
    _block_parallel_sync_behavior,
    _build_training_step_kwargs,
    _extract_hiddens,
    check_finite_loss,
)
from pytorch_lightning.profiler import BaseProfiler, PassThroughProfiler
from pytorch_lightning.trainer.progress import OptimizationProgress
from pytorch_lightning.utilities import AMPType, DeviceType, grad_norm
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
from pytorch_lightning.utilities.imports import _TPU_AVAILABLE
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache


@dataclass
class ClosureResult(OutputResult):
    """A container to hold the result of a :class:`Closure` call.

    It is created from the output of :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`.

    Attributes:
        closure_loss: The loss with a graph attached.
        loss: A detached copy of the closure loss.
        extra: Any keys other than the loss returned.
    """

    closure_loss: Optional[Tensor]
    loss: Optional[Tensor] = field(init=False, default=None)
    extra: Dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TODO: remove with the deprecation removal in v1.6
        self.extra = self._check_extra_detach_deprecation(self.extra)

        self._clone_loss()

    def _clone_loss(self) -> None:
        if self.closure_loss is not None:
            # the loss will get scaled for amp. avoid any modifications to it
            self.loss = self.closure_loss.detach().clone()

    @classmethod
    def from_training_step_output(
        cls, training_step_output: Optional[STEP_OUTPUT], normalize: int = 1
    ) -> "ClosureResult":
        closure_loss, extra = None, {}

        if isinstance(training_step_output, dict):
            # this should not modify the `training_step_output`, as the user could be using it after `training_step_end`
            closure_loss = training_step_output.get("loss")
            extra = {k: v for k, v in training_step_output.items() if k not in ("loss", "hiddens")}
        elif isinstance(training_step_output, Tensor):
            closure_loss = training_step_output
        elif training_step_output is not None:
            raise MisconfigurationException(
                "In automatic optimization, `training_step` must either return a Tensor, "
                "a dict with key 'loss' or None (where the step will be skipped)."
            )

        if closure_loss is not None:
            # accumulate the loss. If ``accumulate_grad_batches == 1``, no effect
            closure_loss /= normalize

        return cls(closure_loss, extra=extra)

    def drop_closure_loss(self) -> "ClosureResult":
        """Return itself without the closure loss which could have a `grad_fn`"""
        self.closure_loss = None
        return self


class Closure(AbstractClosure[ClosureResult]):
    """An implementation of a :class:`AbstractClosure` for automatic optimization in Lightning that combines three
    elementary closures into one: ``training_step``, ``backward`` and ``zero_grad``.

    The Closure gets created by the training loop(s) and is then passed to the
    :meth:`torch.optim.Optimizer.step` method. An optimizer is responsible for calling the closure and optionally
    do something with the output.

    Args:
        step_fn: This is typically the :meth:`pytorch_lightning.core.lightning.LightningModule.training_step
            wrapped with processing for its outputs
        backward_fn: A function that takes a loss value as input, performs back-propagation and returns the loss value.
            Can be set to ``None`` to skip the backward operation.
        zero_grad_fn: A function that zeroes the gradients. Can be set to ``None`` to skip zero_grad, for example
            when accumulating gradients.
        profiler: A profiler for profiling the actions of the passed in closure functions.

    Example:

        closure = Closure()
        optimizer = torch.optim.Adam(...)
        optimizer.step(closure)
    """

    warning_cache = WarningCache()

    def __init__(
        self,
        step_fn: Callable[[], ClosureResult],
        backward_fn: Optional[Callable[[Tensor], Tensor]] = None,
        zero_grad_fn: Optional[Callable[[], None]] = None,
        profiler: Optional[BaseProfiler] = None,
    ):
        super().__init__()
        self._step_fn = step_fn
        self._backward_fn = backward_fn
        self._zero_grad_fn = zero_grad_fn
        self._profiler = PassThroughProfiler() if profiler is None else profiler

    def closure(self, *args: Any, **kwargs: Any) -> ClosureResult:
        with self._profiler.profile("training_step_and_backward"):
            step_output = self._step_fn()

            if step_output.closure_loss is None:
                self.warning_cache.warn(
                    "`training_step` returned `None`. If this was on purpose, ignore this warning..."
                )

            if self._zero_grad_fn is not None:
                with self._profiler.profile("zero_grad"):
                    self._zero_grad_fn()

            if self._backward_fn is not None and step_output.closure_loss is not None:
                with self._profiler.profile("backward"):
                    step_output.closure_loss = self._backward_fn(step_output.closure_loss)

        return step_output

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Tensor]:
        self._result = self.closure(*args, **kwargs)
        return self._result.loss


_OUTPUTS_TYPE = List[List[ClosureResult]]


class OptimizerLoop(Loop):
    """Runs over a sequence of optimizers.

    This loop implements what is known in Lightning as Automatic Optimization.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: use default dict here to simplify logic in loop
        self.outputs: _OUTPUTS_TYPE = []
        self.optim_progress: OptimizationProgress = OptimizationProgress()

        self._skip_backward: bool = False
        self._batch_idx: int = 0
        self._optimizers: List[Optimizer] = []
        self._hiddens: Optional[Any] = None

    @property
    def done(self) -> bool:
        """Returns ``True`` when the last optimizer in the sequence has run."""
        return self.optim_progress.optimizer_idx >= len(self._optimizers)

    def connect(self, **kwargs: "Loop") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def reset(self) -> None:
        if not self.restarting or self.done:
            self.optim_progress.optimizer_idx = 0
        self.outputs = [[] for _ in range(len(self.trainer.optimizers))]

    def on_run_start(self, batch: Any, optimizers: List[Optimizer], batch_idx: int) -> None:  # type: ignore[override]
        self._batch_idx = batch_idx
        self._optimizers = optimizers

    def advance(self, batch: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        result = self._run_optimization(
            batch,
            self._batch_idx,
            self._optimizers[self.optim_progress.optimizer_idx],
            self.optim_progress.optimizer_idx,
        )
        if result.loss is not None:
            self.outputs[self.optim_progress.optimizer_idx].append(result.drop_closure_loss())

        self.optim_progress.optimizer_idx += 1

    def on_run_end(self) -> _OUTPUTS_TYPE:
        outputs, self.outputs = self.outputs, []  # free memory
        return outputs

    def backward(
        self, loss: Tensor, optimizer: torch.optim.Optimizer, opt_idx: int, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Performs the backward step.

        Args:
            loss: The loss value to back-propagate on
            optimizer: Current optimizer being used
            opt_idx: Index of the current optimizer being used
        """
        self.trainer.accelerator.backward(loss, optimizer, opt_idx, *args, **kwargs)

        if not self.trainer.fit_loop.should_accumulate():
            # track gradients
            grad_norm_dict = self._track_and_norm_grad(optimizer=optimizer)
            if grad_norm_dict:
                self.trainer.lightning_module._current_fx_name = "on_after_backward"
                self.trainer.lightning_module.log_grad_norm(grad_norm_dict)
        return loss

    def _run_optimization(
        self, split_batch: Any, batch_idx: int, optimizer: torch.optim.Optimizer, opt_idx: int
    ) -> ClosureResult:
        """Runs closure (train step + backward) together with optimization if necessary.

        Args:
            split_batch: the current tbptt split of the whole batch
            batch_idx: the index of the current batch
            optimizer: the current optimizer
            opt_idx: the index of the current optimizer
        """
        # toggle model params
        self._run_optimization_start(opt_idx, optimizer)

        closure = self._make_closure(split_batch, batch_idx, opt_idx, optimizer)

        if (
            # when the training type plugin handles accumulation, we want to always call the optimizer step
            not self.trainer.training_type_plugin.handles_gradient_accumulation
            and self.trainer.fit_loop.should_accumulate()
        ):
            # For gradient accumulation

            # -------------------
            # calculate loss (train step + train step end)
            # -------------------
            # automatic_optimization=True: perform ddp sync only when performing optimizer_step
            with _block_parallel_sync_behavior(self.trainer, block=True):
                closure()

        # ------------------------------
        # BACKWARD PASS
        # ------------------------------
        # gradient update with accumulated gradients
        else:
            self._optimizer_step(optimizer, opt_idx, batch_idx, closure)

        result = closure.consume_result()

        if result.loss is not None:
            # if no result, user decided to skip optimization
            # otherwise update running loss + reset accumulated loss
            # TODO: find proper way to handle updating running loss
            assert self.trainer.fit_loop is not None
            assert self.trainer.fit_loop.epoch_loop is not None
            assert self.trainer.fit_loop.epoch_loop.batch_loop is not None
            self.trainer.fit_loop.epoch_loop.batch_loop._update_running_loss(result.loss)

        # untoggle model params
        self._run_optimization_end(opt_idx)
        return result

    def _make_closure(self, split_batch: Any, batch_idx: int, opt_idx: int, optimizer: Optimizer) -> Closure:
        """Build a closure object that captures the given arguments and runs the `training_step` function and
        optionally other functions such as `backward` and `zero_grad`."""
        step_fn = self._make_step_fn(split_batch, batch_idx, opt_idx)
        backward_fn = self._make_backward_fn(optimizer, opt_idx)
        zero_grad_fn = self._make_zero_grad_fn(batch_idx, opt_idx, optimizer)

        return Closure(
            step_fn=step_fn, backward_fn=backward_fn, zero_grad_fn=zero_grad_fn, profiler=self.trainer.profiler
        )

    def _make_step_fn(self, split_batch: Any, batch_idx: int, opt_idx: int) -> Callable[[], ClosureResult]:
        """Build the step function that runs the `training_step` and processes its output."""
        return partial(self._training_step, split_batch, batch_idx, opt_idx)

    def _make_zero_grad_fn(self, batch_idx: int, opt_idx: int, optimizer: Optimizer) -> Optional[Callable[[], None]]:
        """Build a `zero_grad` function that zeroes the gradients before back-propagation.

        Returns ``None`` in the case backward needs to be skipped.
        """

        if self._skip_backward:
            return None

        is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0
        if not is_first_batch_to_accumulate:
            return None

        def zero_grad_fn() -> None:
            self._on_before_zero_grad(optimizer)
            self._optimizer_zero_grad(batch_idx, optimizer, opt_idx)

        return zero_grad_fn

    def _make_backward_fn(self, optimizer: Optimizer, opt_idx: int) -> Optional[Callable[[Tensor], Tensor]]:
        """Build a `backward` function that handles back-propagation through the output produced by the
        `training_step` function.

        Returns ``None`` in the case backward needs to be skipped.
        """
        if self._skip_backward:
            return None

        def backward_fn(loss: Tensor) -> Tensor:
            self.backward(loss, optimizer, opt_idx)

            # check if model weights are nan
            if self.trainer.terminate_on_nan:
                detect_nan_parameters(self.trainer.lightning_module)

            return loss

        return backward_fn

    def _run_optimization_start(self, opt_idx: int, optimizer: torch.optim.Optimizer) -> None:
        """Toggles the optimizer to ensure the correct one is used and prevend dangling grads.

        Args:
            opt_idx: the index of the optimizer to use
            optimizer: the optimizer to use
        """
        # make sure only the gradients of the current optimizer's parameters are calculated
        # in the training step to prevent dangling gradients in multiple-optimizer setup.
        if len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.toggle_optimizer(optimizer, opt_idx)

    def _run_optimization_end(self, opt_idx: int) -> None:
        if len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.untoggle_optimizer(opt_idx)

    def _optimizer_step(
        self, optimizer: torch.optim.Optimizer, opt_idx: int, batch_idx: int, train_step_and_backward_closure: Callable
    ) -> None:
        """Performs the optimizer step and some sanity checking.

        Args:
            optimizer: the optimizer to perform the step with
            opt_idx: the index of the current :param:`optimizer`
            batch_idx: the index of the current batch
            train_step_and_backward_closure: the closure function performing the train step and computing the
                gradients. By default called by the optimizer (if possible)
        """
        lightning_module = self.trainer.lightning_module

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        using_native_amp = self.trainer.amp_backend is not None and self.trainer.amp_backend == AMPType.NATIVE

        # native amp + lbfgs is a no go right now
        if using_native_amp and is_lbfgs:
            raise MisconfigurationException(
                "native PyTorch amp and lbfgs are not compatible."
                " To request, please file a Github issue in PyTorch and tag @mcarilli"
            )

        # wraps into LightningOptimizer only for running step
        optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer, opt_idx)

        self.optim_progress.optimizer.step.increment_ready()

        # model hook
        lightning_module.optimizer_step(
            self.trainer.current_epoch,
            batch_idx,
            optimizer,
            opt_idx,
            train_step_and_backward_closure,
            on_tpu=(self.trainer._device_type == DeviceType.TPU and _TPU_AVAILABLE),
            using_native_amp=using_native_amp,
            using_lbfgs=is_lbfgs,
        )

        self.optim_progress.optimizer.step.increment_completed()

    def _on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Calls the ``on_before_zero_grad`` hook.

        Args:
            optimizer: the current optimizer
        """
        self.optim_progress.optimizer.zero_grad.increment_ready()
        self.trainer.call_hook("on_before_zero_grad", optimizer)
        self.optim_progress.optimizer.zero_grad.increment_started()

    def _optimizer_zero_grad(self, batch_idx: int, optimizer: torch.optim.Optimizer, opt_idx: int) -> None:
        """Zeroes out all gradients of parameters optimized by the current optimizer.

        Args:
            batch_idx: the index of the current batch
            optimizer: the current optimizer
            opt_idx: the index of the current optimizer
        """
        self.trainer.accelerator.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)
        self.optim_progress.optimizer.zero_grad.increment_completed()

    def _training_step(self, split_batch: Any, batch_idx: int, opt_idx: int) -> ClosureResult:
        """Performs the actual train step with the tied hooks.

        Args:
            split_batch: the current tbptt split of the current batch
            batch_idx: the index of the current batch
            opt_idx: the index of the current optimizer

        Returns:
            A ``ClosureResult`` containing the training step output.
        """
        # give the PL module a result for logging
        lightning_module = self.trainer.lightning_module

        with self.trainer.profiler.profile("model_forward"):

            step_kwargs = _build_training_step_kwargs(
                lightning_module, self.trainer.optimizers, split_batch, batch_idx, opt_idx, self._hiddens
            )

            # manually capture logged metrics
            lightning_module._current_fx_name = "training_step"
            with self.trainer.profiler.profile("training_step"):
                training_step_output = self.trainer.accelerator.training_step(step_kwargs)
                self.trainer.accelerator.post_training_step()

            del step_kwargs

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            self._hiddens = _extract_hiddens(training_step_output, lightning_module.truncated_bptt_steps)

            result = ClosureResult.from_training_step_output(training_step_output, self.trainer.accumulate_grad_batches)

            if self.trainer.terminate_on_nan:
                check_finite_loss(result.closure_loss)

            if self.trainer.move_metrics_to_cpu:
                # hiddens and the training step output are not moved as they are not considered "metrics"
                assert self.trainer._results is not None
                self.trainer._results.cpu()

        return result

    def _track_and_norm_grad(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Tracks gradient norms and clips the gradients of all parameters optimized by the current optimizer.

        Args:
            optimizer: the current optimizer
        """
        # track gradient norms
        grad_norm_dict = {}
        can_log = (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0
        should_track = float(self.trainer.track_grad_norm) > 0
        if should_track and can_log:
            grad_norm_dict = grad_norm(self.trainer.lightning_module, self.trainer.track_grad_norm)

        # clip gradients
        self.trainer.accelerator.clip_gradients(
            optimizer, self.trainer.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm
        )
        return grad_norm_dict
