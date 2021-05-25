from collections import OrderedDict
from contextlib import contextmanager
from copy import copy
from functools import partial, update_wrapper
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.plugins import ParallelPlugin
from pytorch_lightning.trainer.supporters import prefetch_iterator, TensorRunningAccum
from pytorch_lightning.utilities import AMPType, AttributeDict, DeviceType, grad_norm
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
from pytorch_lightning.utilities.imports import _TPU_AVAILABLE
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.warnings import WarningCache


class BatchLoop(Loop):
    """ Runs over a single batch of data. """

    def __init__(self):
        super().__init__()
        # self.accumulated_loss = None  # TODO: needs to be done over epoch
        self.warning_cache = WarningCache()
        # self._teardown_already_run = False
        self.running_loss = TensorRunningAccum(window_length=20)
        self.accumulated_loss = None
        self._skip_backward = False
        self._hiddens = None
        self._optimizer_freq_cumsum = None

        self.split_idx = None

    def connect(self, trainer, *args, **kwargs):
        self.trainer = trainer

    @property
    def done(self):
        return len(self._remaining_splits) == 0

    def run(self, batch, batch_idx, dataloader_idx):
        if batch is None:
            return AttributeDict(signal=0, grad_norm_dic={})

        # hook
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic={})

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, dataloader_idx)
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic={})

        super().run(batch, batch_idx, dataloader_idx)

        output = AttributeDict(
            signal=0,
            # todo: Properly aggregate grad_norm accros opt_idx and split_idx
            # grad_norm_dict=grad_norm_dict,
            grad_norm_dict={},
            training_step_output_for_epoch_end=self.batch_outputs,
        )
        return output

    def on_run_start(self, batch, batch_idx, dataloader_idx):
        self._hiddens = None
        self._remaining_splits = list(enumerate(self.tbptt_split_batch(batch)))

        # TODO: let loops track individual outputs
        self.batch_outputs = [[] for _ in range(len(self.trainer.optimizers))]

    def advance(self, batch, batch_idx, dataloader_idx):
        split_idx, split_batch = self._remaining_splits.pop(0)
        self.split_idx = split_idx

        # TODO: this list needs to go outside this loop
        # batch_outputs = [[] for _ in range(len(self.trainer.optimizers))]

        if self.trainer.lightning_module.automatic_optimization:
            for opt_idx, optimizer in self.get_active_optimizers(batch_idx):
                result = self._run_optimization(batch_idx, split_idx, split_batch, opt_idx, optimizer)
                if result:
                    self.batch_outputs[opt_idx].append(result.training_step_output_for_epoch_end)
                    grad_norm_dict = result.get("grad_norm_dict", {})
        else:
            # in manual optimization, there is no looping over optimizers
            result = self._run_optimization(batch_idx, split_idx, split_batch)
            if result:
                self.batch_outputs[0].append(result.training_step_output_for_epoch_end)


# ------------------------------------------------------------------------------------------------------------
# HELPER --- TO BE CLEANED UP
# ------------------------------------------------------------------------------------------------------------

    def num_active_optimizers(self, batch_idx: Optional[int] = None) -> int:
        return len(self.get_active_optimizers(batch_idx))

    @property
    def optimizer_freq_cumsum(self):
        if self._optimizer_freq_cumsum is None:
            self._optimizer_freq_cumsum = np.cumsum(self.trainer.optimizer_frequencies)
        return self._optimizer_freq_cumsum

    def _run_optimization(self, batch_idx, split_idx, split_batch, opt_idx=0, optimizer=None):
        # TODO: In v1.5, when optimizer_idx gets removed from training_step in manual_optimization, change
        #   opt_idx=0 to opt_idx=None in the signature here

        # toggle model params + set info to logger_connector
        self.run_train_split_start(split_idx, split_batch, opt_idx, optimizer)

        result = AttributeDict()
        closure = self.make_closure(split_batch, batch_idx, opt_idx, optimizer, self._hiddens, result)

        if self.should_accumulate():
            # For gradient accumulation

            # -------------------
            # calculate loss (train step + train step end)
            # -------------------
            # automatic_optimization=True: perform ddp sync only when performing optimizer_step
            # automatic_optimization=False: don't block synchronization here
            with self.block_ddp_sync_behaviour():
                closure()

        # ------------------------------
        # BACKWARD PASS
        # ------------------------------
        # gradient update with accumulated gradients
        else:
            if self.trainer.lightning_module.automatic_optimization:
                self.optimizer_step(optimizer, opt_idx, batch_idx, closure)
                if len(self.trainer.optimizers) > 1:
                    # revert back to previous state
                    self.trainer.lightning_module.untoggle_optimizer(opt_idx)
            else:
                result = self.training_step(split_batch, batch_idx, opt_idx, self._hiddens)

            if not result:
                # user decided to skip optimization
                return result

            # update running loss + reset accumulated loss
            self.update_running_loss(result.loss)

        self._process_closure_result(result)
        return result

    def training_step_and_backward_closure(
        self,
        split_batch: Any,
        batch_idx: int,
        opt_idx: int,
        optimizer: Optimizer,
        hiddens,
        return_result: AttributeDict,
    ) -> Optional[torch.Tensor]:

        step_result = self.training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
        if step_result is not None:
            return_result.update(step_result)
            return return_result.loss

    def make_closure(self, *closure_args, **closure_kwargs: Any) -> Callable:
        """ Wraps the training step closure into a partial object which will be called within ``optimizer.step``. """
        partial_func = partial(self.training_step_and_backward_closure, *closure_args, **closure_kwargs)
        return update_wrapper(partial_func, self.training_step_and_backward_closure)

    def _process_closure_result(self, opt_closure_result: Optional[AttributeDict]) -> None:
        if not opt_closure_result:
            return

        # cache metrics
        self.trainer.logger_connector.cache_training_step_metrics(opt_closure_result)

        # check if loss or model weights are nan
        if self.trainer.terminate_on_nan:
            self._check_finite(opt_closure_result.loss)

    def on_after_backward(self, training_step_output, batch_idx, untouched_loss):
        training_step_output.detach()

        # insert after step hook
        self.trainer.call_hook("on_after_backward")

        # when in dev debugging track the losses
        self.trainer.dev_debugger.track_train_loss_history(batch_idx, untouched_loss.detach())

    def _check_training_step_output(self, training_step_output):
        if isinstance(training_step_output, torch.Tensor) and not self.trainer.lightning_module.automatic_optimization:
            if training_step_output.grad_fn is None:
                # TODO: Find why - RuntimeError: Expected to mark a variable ready only once ...
                raise MisconfigurationException("In manual optimization, `training_step` should not return a Tensor")

    def training_step(self, split_batch, batch_idx, opt_idx, hiddens):
        # give the PL module a result for logging
        model_ref = self.trainer.lightning_module

        with self.trainer.profiler.profile("model_forward"):
            step_kwargs = self._build_kwargs(split_batch, batch_idx, opt_idx, hiddens)

            # manually capture logged metrics
            model_ref._current_fx_name = 'training_step'
            model_ref._results = Result()
            with self.trainer.profiler.profile("training_step"):
                training_step_output = self.trainer.accelerator.training_step(step_kwargs)
                self.trainer.accelerator.post_training_step()

            self.trainer.logger_connector.cache_logged_metrics()

            self._check_training_step_output(training_step_output)

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            training_step_output_for_epoch_end, training_step_output = self._process_training_step_output(
                training_step_output, split_batch
            )
            if training_step_output_for_epoch_end is None:
                return

        # enable empty loss when using manual opt
        closure_loss = None
        untouched_loss = None

        if self.trainer.lightning_module.automatic_optimization:
            # accumulate loss. if accumulate_grad_batches==1, no effect
            closure_loss = training_step_output.minimize / self.trainer.accumulate_grad_batches

            # the loss will get scaled for amp. avoid any modifications to it
            untouched_loss = closure_loss.detach().clone()

        # result
        result = AttributeDict(
            closure_loss=closure_loss,
            loss=untouched_loss,
            training_step_output=training_step_output,  # Result object
            training_step_output_for_epoch_end=training_step_output_for_epoch_end,  # Result object
        )
        return result

    def _process_training_step_output(self, training_step_output, split_batch):
        training_step_output_for_epoch_end = training_step_output

        # enable validation_step return None
        if training_step_output_for_epoch_end is None:
            return None, None

        result = self.trainer.lightning_module._results

        loss = None
        hiddens = None
        result["extra"] = {}

        # handle dict return
        if isinstance(training_step_output, dict):
            loss = training_step_output.pop("loss", None)
            hiddens = training_step_output.pop("hiddens", None)
            if hiddens is not None:
                hiddens = hiddens.detach()
            result["extra"] = training_step_output

        # handle scalar return
        elif isinstance(training_step_output, torch.Tensor):
            loss = training_step_output

        # map to results under the hood
        result.minimize = loss
        self._hiddens = hiddens

        # track batch for manual reduction with result
        result.track_batch_size(len(split_batch))

        # track metrics without grads for epoch reduction
        training_step_output_for_epoch_end = copy(result)
        training_step_output_for_epoch_end = training_step_output_for_epoch_end.detach()
        if self.trainer.move_metrics_to_cpu:
            training_step_output_for_epoch_end = training_step_output_for_epoch_end.cpu()

        return training_step_output_for_epoch_end, result

    def optimizer_step(self, optimizer, opt_idx, batch_idx, train_step_and_backward_closure):
        model_ref = self.trainer.lightning_module

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        using_native_amp = self.trainer.amp_backend == AMPType.NATIVE

        # native amp + lbfgs is a no go right now
        if using_native_amp and is_lbfgs:
            raise MisconfigurationException(
                'native PyTorch amp and lbfgs are not compatible.'
                ' To request, please file a Github issue in PyTorch and tag @mcarilli'
            )

        # wraps into LightningOptimizer only for running step
        optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer, opt_idx)

        # model hook
        model_ref.optimizer_step(
            self.trainer.current_epoch,
            batch_idx,
            optimizer,
            opt_idx,
            train_step_and_backward_closure,
            on_tpu=(self.trainer._device_type == DeviceType.TPU and _TPU_AVAILABLE),
            using_native_amp=using_native_amp,
            using_lbfgs=is_lbfgs,
        )

    def on_before_zero_grad(self, optimizer):
        self.trainer.call_hook('on_before_zero_grad', optimizer)

    def optimizer_zero_grad(self, batch_idx, optimizer, opt_idx):
        self.trainer.accelerator.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)

    def track_and_norm_grad(self, optimizer) -> dict:
        # track gradient norms
        grad_norm_dict = self._track_gradient_norm()

        # clip gradients
        self.trainer.accelerator.clip_gradients(
            optimizer, self.trainer.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm
        )
        return grad_norm_dict

    def _track_gradient_norm(self):
        grad_norm_dict = {}
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if float(self.trainer.track_grad_norm) > 0:
                model = self.trainer.lightning_module
                grad_norm_dict = grad_norm(model, self.trainer.track_grad_norm)
        return grad_norm_dict

    def _accumulated_batches_reached(self):
        return (self.iteration_count + 1) % self.trainer.accumulate_grad_batches == 0

    def _num_training_batches_reached(self, is_last_batch=False):
        return (self.iteration_count + 1) == self.trainer.num_training_batches or is_last_batch

    def should_accumulate(self):
        # checks if backward or backward + optimizer step (via closure)
        accumulation_done = self._accumulated_batches_reached()
        is_final_batch = self._num_training_batches_reached()
        return not (accumulation_done or is_final_batch)

    def tbptt_split_batch(self, batch):
        splits = [batch]
        if self.trainer.truncated_bptt_steps is not None:
            model_ref = self.trainer.lightning_module
            with self.trainer.profiler.profile("tbptt_split_batch"):
                splits = model_ref.tbptt_split_batch(batch, self.trainer.truncated_bptt_steps)
        return splits

    def build_train_args(self, batch, batch_idx, opt_idx, hiddens):
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx]

        if len(self.trainer.optimizers) > 1:
            if self.trainer.has_arg("training_step", "optimizer_idx"):
                if not self.trainer.lightning_module.automatic_optimization:
                    self.warning_cache.warn(
                        "`training_step` hook signature has changed in v1.3."
                        " `optimizer_idx` argument has been removed in case of manual optimization. Support for"
                        " the old signature will be removed in v1.5", DeprecationWarning
                    )
                args.append(opt_idx)
            elif not self.trainer.has_arg(
                "training_step", "optimizer_idx"
            ) and self.trainer.lightning_module.automatic_optimization:
                raise ValueError(
                    f"Your LightningModule defines {len(self.trainer.optimizers)} optimizers but"
                    ' `training_step` is missing the `optimizer_idx` argument.'
                )

        # pass hiddens if using tbptt
        if self.trainer.truncated_bptt_steps is not None:
            args.append(hiddens)

        return args

    def run_train_split_start(self, split_idx, split_batch, opt_idx, optimizer):
        # set split_idx to trainer for tracking
        self.trainer.split_idx = split_idx

        # make sure only the gradients of the current optimizer's parameters are calculated
        # in the training step to prevent dangling gradients in multiple-optimizer setup.
        if self.trainer.lightning_module.automatic_optimization and len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.toggle_optimizer(optimizer, opt_idx)

        # use to track metrics internally
        self.trainer.logger_connector.on_train_split_start(split_idx, opt_idx, split_batch)

    @contextmanager
    def block_ddp_sync_behaviour(self, should_block_sync: bool = False):
        """
        automatic_optimization = True
        Blocks ddp sync gradients behaviour on backwards pass.
        This is useful for skipping sync when accumulating gradients, reducing communication overhead

        automatic_optimization = False
        do not block ddp gradient sync when using manual optimization
        as gradients are needed within the training step

        Returns:
            context manager with sync behaviour off

        """
        if (
            isinstance(self.trainer.training_type_plugin, ParallelPlugin)
            and (self.trainer.lightning_module.automatic_optimization or should_block_sync)
        ):
            with self.trainer.training_type_plugin.block_backward_sync():
                yield None
        else:
            yield None

    def training_step_and_backward(self, split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """Wrap forward, zero_grad and backward in a closure so second order methods work"""
        with self.trainer.profiler.profile("training_step_and_backward"):
            # lightning module hook
            result = self.training_step(split_batch, batch_idx, opt_idx, hiddens)

            if not self._skip_backward and self.trainer.lightning_module.automatic_optimization:
                is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0

                if is_first_batch_to_accumulate:
                    self.on_before_zero_grad(optimizer)
                    self.optimizer_zero_grad(batch_idx, optimizer, opt_idx)

                # backward pass
                if result is not None:
                    with self.trainer.profiler.profile("backward"):
                        self.backward(result, optimizer, opt_idx)

                    # hook - call this hook only
                    # when gradients have finished to accumulate
                    if not self.should_accumulate():
                        self.on_after_backward(result.training_step_output, batch_idx, result.loss)

                    # check if loss or model weights are nan
                    if self.trainer.terminate_on_nan:
                        self._check_finite(result.loss)

                else:
                    self.warning_cache.warn(
                        "training_step returned None. If this was on purpose, ignore this warning..."
                    )

        return result

    def _check_finite(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss).all():
            raise ValueError(f'The loss returned in `training_step` is {loss}.')
        model = self.trainer.lightning_module
        detect_nan_parameters(model)

    def backward(self, result, optimizer, opt_idx, *args, **kwargs):
        self.trainer.dev_debugger.track_event("backward_call")

        should_accumulate = self.should_accumulate()

        # backward can be called manually in the training loop
        if isinstance(result, torch.Tensor):
            self.trainer.accelerator.backward(result, optimizer, opt_idx, should_accumulate, *args, **kwargs)
        else:
            result.closure_loss = self.trainer.accelerator.backward(
                result.closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs
            )

        if not self.should_accumulate():
            # track gradients
            result.grad_norm_dict = self.track_and_norm_grad(optimizer=optimizer)

    def update_running_loss(self, current_loss: torch.Tensor) -> None:
        if self.trainer.lightning_module.automatic_optimization:
            # track total loss for logging (avoid mem leaks)
            self.accumulated_loss.append(current_loss)

        accumulated_loss = self.accumulated_loss.mean()

        if accumulated_loss is not None:
            # calculate running loss for display
            self.running_loss.append(self.accumulated_loss.mean() * self.trainer.accumulate_grad_batches)

        # reset for next set of accumulated grads
        self.accumulated_loss.reset()

    def get_active_optimizers(self, batch_idx: Optional[int] = None) -> List[Tuple[int, Optimizer]]:
        """
        Returns the currently active optimizers. When multiple optimizers are used with different frequencies,
        only one of the optimizers is active at a time.

        Returns:
            A list of tuples (opt_idx, optimizer) of currently active optimizers.
        """
        if not self.trainer.optimizer_frequencies:
            # call training_step once per optimizer
            return list(enumerate(self.trainer.optimizers))

        optimizers_loop_length = self.optimizer_freq_cumsum[-1]
        current_place_in_loop = batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = int(np.argmax(self.optimizer_freq_cumsum > current_place_in_loop))
        return [(opt_idx, self.trainer.optimizers[opt_idx])]

    def _build_kwargs(self, batch, batch_idx, opt_idx, hiddens):
        # enable not needing to add opt_idx to training_step
        step_kwargs = OrderedDict([('batch', batch), ('batch_idx', batch_idx)])

        lightning_module = self.trainer.lightning_module

        if len(self.trainer.optimizers) > 1:
            training_step_fx = getattr(lightning_module, "training_step")
            has_opt_idx_in_train_step = is_param_in_hook_signature(training_step_fx, "optimizer_idx")
            if has_opt_idx_in_train_step:
                if not lightning_module.automatic_optimization:
                    self.warning_cache.warn(
                        "`training_step` hook signature has changed in v1.3."
                        " `optimizer_idx` argument has been removed in case of manual optimization. Support for"
                        " the old signature will be removed in v1.5", DeprecationWarning
                    )
                step_kwargs['optimizer_idx'] = opt_idx
            elif not has_opt_idx_in_train_step and self.trainer.lightning_module.automatic_optimization:
                raise ValueError(
                    f"Your LightningModule defines {len(self.trainer.optimizers)} optimizers but"
                    ' `training_step` is missing the `optimizer_idx` argument.'
                )

        # pass hiddens if using tbptt
        if self._truncated_bptt_enabled():
            step_kwargs['hiddens'] = hiddens

        return step_kwargs

    def _truncated_bptt_enabled(self) -> bool:
        """ Temporary tbptt utilities until this flag is fully migrated to the lightning module. """
        return self._truncated_bptt_steps() > 0

    def _truncated_bptt_steps(self) -> int:
        lightning_module = self.trainer.lightning_module
        # Give precedence to the LightningModule as the Trainer flag will be removed in v1.5
        if lightning_module.truncated_bptt_steps > 0:
            return lightning_module.truncated_bptt_steps
        return self.trainer.truncated_bptt_steps or 0
