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

from collections import OrderedDict
from contextlib import contextmanager, suppress
from copy import copy
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import Optimizer

from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins import ParallelPlugin
from pytorch_lightning.trainer.connectors.logger_connector.result import Result
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import _TPU_AVAILABLE, AMPType, DeviceType
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
from pytorch_lightning.utilities.grads import grad_norm
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.warnings import WarningCache


class TrainLoop:

    def __init__(
        self,
        trainer,
        max_epochs: Optional[int],
        min_epochs: Optional[int],
        max_steps: Optional[int],
        min_steps: Optional[int],
        num_sanity_val_steps: int,
    ):
        self.trainer = trainer
        self.accumulated_loss = None
        self.warning_cache = WarningCache()
        self._teardown_already_run = False
        self.running_loss = TensorRunningAccum(window_length=20)
        self._skip_backward = False
        self._optimizer_freq_cumsum = None
        self._hiddens = None

        self.global_step = 0
        self.current_epoch = 0
        self.trainer.should_stop = False

        # the total batch index across all epochs
        self.total_batch_idx = 0
        # the current batch index in the loop that runs over the dataloader(s)
        self.batch_idx = 0
        # the current split index when the batch gets split into chunks in truncated backprop through time
        self.split_idx = None

        self.trainer.num_training_batches = 0
        self.trainer.train_dataloader = None

        # If neither max_epochs or max_steps is set, then use existing default of max_epochs = 1000
        self.max_epochs = 1000 if (max_epochs is None and max_steps is None) else max_epochs
        # If neither min_epochs or min_steps is set, then use existing default of min_epochs = 1
        self.min_epochs = 1 if (min_epochs is None and min_steps is None) else min_epochs
        self.max_steps = max_steps
        self.min_steps = min_steps

        if num_sanity_val_steps == -1:
            self.trainer.num_sanity_val_steps = float("inf")
        else:
            self.trainer.num_sanity_val_steps = num_sanity_val_steps

    @property
    def num_active_optimizers(self) -> int:
        return len(self.get_active_optimizers())

    @property
    def optimizer_freq_cumsum(self):
        if self._optimizer_freq_cumsum is None:
            self._optimizer_freq_cumsum = np.cumsum(self.trainer.optimizer_frequencies)
        return self._optimizer_freq_cumsum

    def should_skip_training(self) -> bool:
        should_by_max_steps = self.max_steps is not None and self.global_step >= self.max_steps
        should_by_epoch = self.max_epochs is not None and self.current_epoch >= self.max_epochs
        return should_by_max_steps or should_by_epoch or self.trainer.num_training_batches == 0

    def on_train_start(self):
        # hook
        self.trainer.call_hook("on_train_start")

    def on_train_end(self):
        if self._teardown_already_run:
            return
        self._teardown_already_run = True

        # trigger checkpoint check. need to temporarily decrease the global step to avoid saving duplicates
        # when a checkpoint was saved at the last step
        self.global_step -= 1
        self.check_checkpoint_callback(should_update=True, is_last=True)
        self.global_step += 1

        # hook
        self.trainer.call_hook("on_train_end")

        # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
        # It might be related to xla tensors blocked when moving the cpu
        # kill loggers
        if self.trainer.logger is not None:
            self.trainer.logger.finalize("success")

        # summarize profile results
        self.trainer.profiler.describe()

        # give accelerators a chance to finish
        self.trainer.accelerator.on_train_end()

        # reset bookkeeping
        self.trainer.state.stage = None

    def check_checkpoint_callback(self, should_update, is_last=False):
        # TODO bake this logic into the ModelCheckpoint callback
        if should_update and self.trainer.checkpoint_connector.has_trained:
            callbacks = self.trainer.checkpoint_callbacks

            if is_last and any(cb.save_last and cb.verbose for cb in callbacks):
                rank_zero_info("Saving latest checkpoint...")

            model = self.trainer.lightning_module

            for cb in callbacks:
                cb.on_validation_end(self.trainer, model)

    def on_train_epoch_start(self, epoch):

        # update training progress in trainer
        self.current_epoch = epoch

        model = self.trainer.lightning_module

        # reset train dataloader
        if epoch != 0 and self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # todo: specify the possible exception
        with suppress(Exception):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.accumulated_loss = TensorRunningAccum(window_length=self.trainer.accumulate_grad_batches)

        # hook
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")

    def on_train_batch_end(self, epoch_output, batch_end_outputs, batch, batch_idx, dataloader_idx):
        batch_end_outputs = [opt_idx_out for opt_idx_out in batch_end_outputs if len(opt_idx_out)]

        processed_batch_end_outputs = TrainLoop._prepare_outputs(batch_end_outputs, batch_mode=True)

        # hook
        self.trainer.call_hook('on_train_batch_end', processed_batch_end_outputs, batch, batch_idx, dataloader_idx)
        self.trainer.call_hook('on_batch_end')

        # figure out what to track for epoch end
        self.track_epoch_end_reduce_metrics(epoch_output, batch_end_outputs)

        # reset batch logger internals
        self.trainer.logger_connector.on_train_batch_end()

    def reset_train_val_dataloaders(self, model) -> None:
        """
        Resets train and val dataloaders if none are attached to the trainer.

        The val dataloader must be initialized before training loop starts, as the training loop
        inspects the val dataloader to determine whether to run the evaluation loop.
        """
        if self.trainer.train_dataloader is None:
            self.trainer.reset_train_dataloader(model)

        if self.trainer.val_dataloaders is None:
            self.trainer.reset_val_dataloader(model)

    def track_epoch_end_reduce_metrics(self, epoch_output, batch_end_outputs):

        hook_overridden = self._should_add_batch_output_to_epoch_output()

        # track the outputs to reduce at the end of the epoch
        for opt_idx, opt_outputs in enumerate(batch_end_outputs):
            sample_output = opt_outputs[-1]

            # decide if we need to reduce at the end of the epoch automatically
            auto_reduce_tng_result = isinstance(sample_output, Result) and sample_output.should_reduce_on_epoch_end

            # only track when a) it needs to be autoreduced OR b) the user wants to manually reduce on epoch end
            if not (hook_overridden or auto_reduce_tng_result):
                continue

            # with 1 step (no tbptt) don't use a sequence at epoch end
            if isinstance(opt_outputs, list) and len(opt_outputs) == 1 and not isinstance(opt_outputs[0], Result):
                opt_outputs = opt_outputs[0]

            epoch_output[opt_idx].append(opt_outputs)

    def _should_add_batch_output_to_epoch_output(self) -> bool:
        # We add to the epoch outputs if
        # 1. The model defines training_epoch_end OR
        # 2. The model overrides on_train_epoch_end which has `outputs` in the signature
        # TODO: in v1.5 this only needs to check if training_epoch_end is overridden
        lightning_module = self.trainer.lightning_module
        if is_overridden("training_epoch_end", model=lightning_module):
            return True

        if is_overridden("on_train_epoch_end", model=lightning_module):
            model_hook_fx = getattr(lightning_module, "on_train_epoch_end")
            if is_param_in_hook_signature(model_hook_fx, "outputs"):
                return True

        return False

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

        batch_idx = self.total_batch_idx if batch_idx is None else batch_idx
        optimizers_loop_length = self.optimizer_freq_cumsum[-1]
        current_place_in_loop = batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = int(np.argmax(self.optimizer_freq_cumsum > current_place_in_loop))
        return [(opt_idx, self.trainer.optimizers[opt_idx])]

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
            training_step_output=training_step_output,
            training_step_output_for_epoch_end=training_step_output_for_epoch_end,
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

    @staticmethod
    def _prepare_outputs(
        outputs: List[List[List[Result]]],
        batch_mode: bool,
    ) -> Union[List[List[List[Dict]]], List[List[Dict]], List[Dict], Dict]:
        """
        Extract required information from batch or epoch end results.

        Args:
            outputs: A 3-dimensional list of ``Result`` objects with dimensions:
                [optimizer outs][batch outs][tbptt steps].

            batch_mode: If True, ignore the batch output dimension.

        Returns:
            The cleaned outputs with ``Result`` objects converted to dictionaries. All list dimensions of size one will
            be collapsed.
        """
        processed_outputs = []
        for opt_outputs in outputs:
            # handle an edge case where an optimizer output is the empty list
            if len(opt_outputs) == 0:
                continue

            processed_batch_outputs = []

            if batch_mode:
                opt_outputs = [opt_outputs]

            for batch_outputs in opt_outputs:
                processed_tbptt_outputs = []

                for tbptt_output in batch_outputs:
                    out = tbptt_output.extra
                    out['loss'] = tbptt_output.minimize
                    processed_tbptt_outputs.append(out)

                # if there was only one tbptt step then we can collapse that dimension
                if len(processed_tbptt_outputs) == 1:
                    processed_tbptt_outputs = processed_tbptt_outputs[0]
                processed_batch_outputs.append(processed_tbptt_outputs)

            # batch_outputs should be just one dict (or a list of dicts if using tbptt) per optimizer
            if batch_mode:
                processed_batch_outputs = processed_batch_outputs[0]
            processed_outputs.append(processed_batch_outputs)

        # if there is only one optimiser then we collapse that dimension
        if len(processed_outputs) == 1:
            processed_outputs = processed_outputs[0]
        return processed_outputs

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
            on_tpu=self.trainer._device_type == DeviceType.TPU and _TPU_AVAILABLE,
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
        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if float(self.trainer.track_grad_norm) > 0:
                model = self.trainer.lightning_module
                grad_norm_dict = grad_norm(model, self.trainer.track_grad_norm)
        return grad_norm_dict

    def _tbptt_split_batch(self, batch: Any) -> List[Any]:
        splits = [batch]
        truncated_bptt_enabled = self._truncated_bptt_enabled()
        if truncated_bptt_enabled:
            model_ref = self.trainer.lightning_module
            with self.trainer.profiler.profile("tbptt_split_batch"):
                splits = model_ref.tbptt_split_batch(batch, self._truncated_bptt_steps())
        return splits

    def run_training_epoch(self):
        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader)

        # track epoch output
        epoch_output = [[] for _ in range(self.num_active_optimizers)]

        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)
        dataloader_idx = 0
        batch_idx = None

        for batch_idx, (batch, is_last_batch) in train_dataloader:
            self.batch_idx = batch_idx

            # ------------------------------------
            # TRAINING_STEP + TRAINING_STEP_END
            # ------------------------------------
            with self.trainer.profiler.profile("run_training_batch"):
                batch_output = self.run_training_batch(batch, batch_idx, dataloader_idx)

            # when returning -1 from train_step, we end epoch early
            if batch_output.signal == -1:
                break

            # hook
            # TODO: add outputs to batches
            self.on_train_batch_end(
                epoch_output,
                batch_output.training_step_output_for_epoch_end,
                batch,
                batch_idx,
                dataloader_idx,
            )

            # -----------------------------------------
            # SAVE METRICS TO LOGGERS
            # -----------------------------------------
            self.trainer.logger_connector.log_train_step_metrics(batch_output)

            # -----------------------------------------
            # VALIDATE IF NEEDED
            # -----------------------------------------
            should_check_val = self._should_check_val_fx(batch_idx, is_last_batch)
            if should_check_val:
                self.trainer.validating = True
                self.trainer._run_evaluation()
                self.trainer.training = True

            # -----------------------------------------
            # SAVE LOGGERS (ie: Tensorboard, etc...)
            # -----------------------------------------
            self.save_loggers_on_train_batch_end()

            # update LR schedulers
            self.update_lr_schedulers('step')
            self.trainer.checkpoint_connector.has_trained = True

            self.total_batch_idx += 1

            # progress global step according to grads progress
            self.increment_accumulated_grad_global_step()

            max_steps_reached = (self.max_steps is not None and self.max_steps <= self.global_step)
            if max_steps_reached or self.trainer.should_stop or self._num_training_batches_reached(is_last_batch):
                break

        if batch_idx is None:
            # dataloader/iterator did not produce a batch
            return

        # handle epoch_output on epoch end
        self.on_train_epoch_end(epoch_output)

        # the global step is manually decreased here due to backwards compatibility with existing loggers
        # as they expect that the same step is used when logging epoch end metrics even when the batch loop has
        # finished. this means the attribute does not exactly track the number of optimizer steps applied.
        # TODO(@carmocca): deprecate and rename so users don't get confused
        self.global_step -= 1
        # log epoch metrics
        self.trainer.logger_connector.log_train_epoch_end_metrics(epoch_output)
        self.global_step += 1

        self.update_lr_schedulers('epoch')

        did_train_only = self.trainer.disable_validation or self.trainer.evaluation_loop.should_skip_evaluation(
            self.trainer.num_val_batches
        )
        if did_train_only:
            self.global_step -= 1
            self.check_checkpoint_callback(True)
            self.global_step += 1

    def on_train_epoch_end(self, epoch_output: List[List[List[Result]]]) -> None:
        # inform logger the batch loop has finished
        self.trainer.logger_connector.on_train_epoch_end()

        # prepare epoch output
        processed_epoch_output = TrainLoop._prepare_outputs(epoch_output, batch_mode=False)

        # get the model and call model.training_epoch_end
        model = self.trainer.lightning_module

        if is_overridden('training_epoch_end', model=model):
            # run training_epoch_end
            # refresh the result for custom logging at the epoch level
            model._current_fx_name = 'training_epoch_end'
            training_epoch_end_output = model.training_epoch_end(processed_epoch_output)

            if training_epoch_end_output is not None:
                raise MisconfigurationException(
                    'training_epoch_end expects a return of None. '
                    'HINT: remove the return statement in training_epoch_end'
                )

            # capture logging
            self.trainer.logger_connector.cache_logged_metrics()

        # call train epoch end hooks
        self._on_train_epoch_end_hook(processed_epoch_output)
        self.trainer.call_hook('on_epoch_end')

    def _on_train_epoch_end_hook(self, processed_epoch_output) -> None:
        # We cannot rely on Trainer.call_hook because the signatures might be different across
        # lightning module and callback
        # As a result, we need to inspect if the module accepts `outputs` in `on_train_epoch_end`

        # This implementation is copied from Trainer.call_hook
        hook_name = "on_train_epoch_end"

        # set hook_name to model + reset Result obj
        skip = self.trainer._reset_result_and_set_fx_name(hook_name)

        # always profile hooks
        with self.trainer.profiler.profile(hook_name):

            # first call trainer hook
            if hasattr(self.trainer, hook_name):
                trainer_hook = getattr(self.trainer, hook_name)
                trainer_hook(processed_epoch_output)

            # next call hook in lightningModule
            model_ref = self.trainer.lightning_module
            if is_overridden(hook_name, model_ref):
                hook_fx = getattr(model_ref, hook_name)
                if is_param_in_hook_signature(hook_fx, "outputs"):
                    self.warning_cache.warn(
                        "The signature of `ModelHooks.on_train_epoch_end` has changed in v1.3."
                        " `outputs` parameter has been deprecated."
                        " Support for the old signature will be removed in v1.5", DeprecationWarning
                    )
                    model_ref.on_train_epoch_end(processed_epoch_output)
                else:
                    model_ref.on_train_epoch_end()

            # if the PL module doesn't have the hook then call the accelerator
            # used to auto-reduce things for the user with Results obj
            elif hasattr(self.trainer.accelerator, hook_name):
                accelerator_hook = getattr(self.trainer.accelerator, hook_name)
                accelerator_hook()

        if not skip:
            self.trainer._cache_logged_metrics()

    def run_training_batch(self, batch, batch_idx, dataloader_idx):
        # track grad norms
        grad_norm_dict = {}

        # bookkeeping
        self._hiddens = None

        optimizers = list(enumerate(self.trainer.optimizers))

        # track all outputs across time and num of optimizers
        batch_outputs = [[] for _ in range(len(optimizers))]

        if batch is None:
            self.warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
            return AttributeDict(
                signal=0,
                grad_norm_dict={},
                training_step_output_for_epoch_end=batch_outputs,
            )

        # hook
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dict={})

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, dataloader_idx)
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dict={})

        # lightning module hook
        splits = self._tbptt_split_batch(batch)

        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx

            if self.trainer.lightning_module.automatic_optimization:
                for opt_idx, optimizer in self.get_active_optimizers(batch_idx):
                    result = self._run_optimization(batch_idx, split_idx, split_batch, opt_idx, optimizer)
                    if result:
                        batch_outputs[opt_idx].append(result.training_step_output_for_epoch_end)
                        grad_norm_dict = result.get("grad_norm_dict", {})
            else:
                # in manual optimization, there is no looping over optimizers
                result = self._run_optimization(batch_idx, split_idx, split_batch)
                if result:
                    batch_outputs[0].append(result.training_step_output_for_epoch_end)

        output = AttributeDict(
            signal=0,
            # todo: Properly aggregate grad_norm accros opt_idx and split_idx
            grad_norm_dict=grad_norm_dict,
            training_step_output_for_epoch_end=batch_outputs,
        )
        return output

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

        result = self.training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
        if result is not None:
            return_result.update(result)
            return return_result.loss

    def make_closure(self, *closure_args, **closure_kwargs: Any) -> Callable:
        """ Wraps the training step closure into a partial object which will be called within ``optimizer.step``. """
        partial_func = partial(self.training_step_and_backward_closure, *closure_args, **closure_kwargs)
        return update_wrapper(partial_func, self.training_step_and_backward_closure)

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

    def _process_closure_result(self, opt_closure_result: Optional[AttributeDict]) -> None:
        if not opt_closure_result:
            return

        # cache metrics
        self.trainer.logger_connector.cache_training_step_metrics(opt_closure_result)

        # check if loss or model weights are nan
        if self.trainer.terminate_on_nan:
            self._check_finite(opt_closure_result.loss)

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

    def update_lr_schedulers(self, interval: str) -> None:
        if interval == "step":
            finished_accumulation = self._accumulated_batches_reached()
            finished_epoch = self._num_training_batches_reached()
            if not finished_accumulation and not finished_epoch:
                return
        self.trainer.optimizer_connector.update_learning_rates(
            interval=interval,
            opt_indices=[opt_idx for opt_idx, _ in self.get_active_optimizers()],
        )

    def increment_accumulated_grad_global_step(self):
        num_accumulated_batches_reached = self._accumulated_batches_reached()
        num_training_batches_reached = self._num_training_batches_reached()

        # progress global step according to grads progress
        if num_accumulated_batches_reached or num_training_batches_reached:
            self.global_step = self.trainer.accelerator.update_global_step(self.total_batch_idx, self.global_step)

    def _accumulated_batches_reached(self):
        return (self.batch_idx + 1) % self.trainer.accumulate_grad_batches == 0

    def _num_training_batches_reached(self, is_last_batch=False):
        return (self.batch_idx + 1) == self.trainer.num_training_batches or is_last_batch

    def should_accumulate(self):
        # checks if backward or backward + optimizer step (via closure)
        accumulation_done = self._accumulated_batches_reached()
        is_final_batch = self._num_training_batches_reached()
        return not (accumulation_done or is_final_batch)

    def _should_check_val_fx(self, batch_idx: int, is_last_batch: bool) -> bool:
        """ Decide if we should run validation. """
        if not self.trainer.enable_validation:
            return False

        is_val_check_epoch = (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        if not is_val_check_epoch:
            return False

        # val_check_batch is inf for iterable datasets with no length defined
        is_infinite_dataset = self.trainer.val_check_batch == float('inf')
        if is_last_batch and is_infinite_dataset:
            return True

        if self.trainer.should_stop:
            return True

        # TODO: let training/eval loop handle logic around limit_*_batches and val_check_batch
        is_val_check_batch = is_last_batch
        if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
            is_val_check_batch = (batch_idx + 1) % self.trainer.limit_train_batches == 0
        elif self.trainer.val_check_batch != float('inf'):
            is_val_check_batch = (batch_idx + 1) % self.trainer.val_check_batch == 0
        return is_val_check_batch

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

    def save_loggers_on_train_batch_end(self):
        # when loggers should save to disk
        should_flush_logs = self.trainer.logger_connector.should_flush_logs
        if should_flush_logs and self.trainer.is_global_zero and self.trainer.logger is not None:
            self.trainer.logger.save()

    def run_train_split_start(self, split_idx, split_batch, opt_idx, optimizer):
        # make sure only the gradients of the current optimizer's parameters are calculated
        # in the training step to prevent dangling gradients in multiple-optimizer setup.
        if self.trainer.lightning_module.automatic_optimization and len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.toggle_optimizer(optimizer, opt_idx)

        # use to track metrics internally
        self.trainer.logger_connector.on_train_split_start(split_idx, opt_idx, split_batch)

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
