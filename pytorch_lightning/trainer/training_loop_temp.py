import subprocess
import numpy as np
import torch
import torch.distributed as torch_distrib
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.trainer.supporters import Accumulator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.core.step_result import EvalResult, Result
from pytorch_lightning.utilities.parsing import AttributeDict
from copy import copy


class TrainLoop:

    def __init__(self, trainer):
        self.trainer = trainer
        self.should_check_val = False
        self.early_stopping_accumulator = None
        self.checkpoint_accumulator = None
        self._teardown_already_run = False

    @property
    def num_optimizers(self):
        num_optimizers = len(self.get_optimizers_iterable())
        return num_optimizers

    def on_train_start(self):
        # clear cache before training
        if self.trainer.on_gpu and self.trainer.root_gpu is not None:
            # use context because of:
            # https://discuss.pytorch.org/t/out-of-memory-when-i-use-torch-cuda-empty-cache/57898
            with torch.cuda.device(f'cuda:{self.trainer.root_gpu}'):
                torch.cuda.empty_cache()

        # hook
        self.trainer.call_hook('on_train_start')

    def on_train_end(self):
        if self._teardown_already_run:
            return

        self._teardown_already_run = True

        # Save latest checkpoint
        log.info('Saving latest checkpoint..')
        self.check_checkpoint_callback(should_check_val=False)

        # hook
        self.trainer.call_hook('on_train_end')

        # kill loggers
        if self.trainer.logger is not None:
            self.trainer.logger.finalize("success")

        # summarize profile results
        if self.trainer.global_rank == 0:
            self.trainer.profiler.describe()

        if self.trainer.global_rank == 0:
            for proc in self.trainer.interactive_ddp_procs:
                subprocess.Popen.kill(proc)

        # clean up dist group
        if self.trainer.use_ddp or self.trainer.use_ddp2:
            torch_distrib.destroy_process_group()

        # clear mem
        if self.trainer.on_gpu:
            model = self.trainer.get_model()
            model.cpu()
            torch.cuda.empty_cache()

    def check_checkpoint_callback(self, should_check_val):
        model = self.trainer.get_model()

        # when no val loop is present or fast-dev-run still need to call checkpoints
        # TODO bake this logic into the checkpoint callback
        should_activate = not is_overridden('validation_step', model) and not should_check_val
        if should_activate:
            checkpoint_callbacks = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
            [c.on_validation_end(self.trainer, model) for c in checkpoint_callbacks]

    def on_train_epoch_start(self):
        # hook
        self.trainer.call_hook('on_epoch_start')
        self.trainer.call_hook('on_train_epoch_start')

        # bookkeeping
        self.should_check_val = False

        # structured result accumulators for callbacks
        self.early_stopping_accumulator = Accumulator()
        self.checkpoint_accumulator = Accumulator()

    def on_train_batch_end(self, epoch_output, epoch_end_outputs, batch, batch_idx, dataloader_idx):
        # figure out what to track for epoch end
        self.track_epoch_end_reduce_metrics(epoch_output, epoch_end_outputs)

        # hook
        self.trainer.call_hook('on_batch_end')
        self.trainer.call_hook('on_train_batch_end', batch, batch_idx, dataloader_idx)

    def reset_train_val_dataloaders(self, model):
        if not self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        if self.trainer.val_dataloaders is None and not self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_val_dataloader(model)

    def track_epoch_end_reduce_metrics(self, epoch_output, epoch_end_outputs):
        # track the outputs to reduce at the end of the epoch
        for opt_idx, opt_outputs in enumerate(epoch_end_outputs):
            # with 1 step (no tbptt) don't use a sequence at epoch end
            if isinstance(opt_outputs, list) and len(opt_outputs) == 1 and not isinstance(opt_outputs[0], Result):
                opt_outputs = opt_outputs[0]
            epoch_output[opt_idx].append(opt_outputs)

    def get_optimizers_iterable(self):
        """
        Generates an iterable with (idx, optimizer) for each optimizer.
        """
        if not self.trainer.optimizer_frequencies:
            # call training_step once per optimizer
            return list(enumerate(self.trainer.optimizers))

        optimizer_freq_cumsum = np.cumsum(self.trainer.optimizer_frequencies)
        optimizers_loop_length = optimizer_freq_cumsum[-1]
        current_place_in_loop = self.trainer.total_batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = np.argmax(optimizer_freq_cumsum > current_place_in_loop)
        return [(opt_idx, self.trainer.optimizers[opt_idx])]

    def backward(self, result, optimizer, opt_idx):
        # backward pass
        with self.trainer.profiler.profile('model_backward'):
            result.closure_loss = self.trainer.accelerator_backend.backward(result.closure_loss, optimizer, opt_idx)

    def on_after_backward(self, training_step_output, batch_idx, untouched_loss):
        is_result_obj = isinstance(training_step_output, Result)

        if is_result_obj:
            training_step_output.detach()
        else:
            training_step_output.batch_loss = training_step_output.batch_loss.detach()

        # insert after step hook
        self.trainer.call_hook('on_after_backward')

        # when in dev debugging track the losses
        self.trainer.dev_debugger.track_train_loss_history(batch_idx, untouched_loss.detach())

    def training_step(self, split_batch, batch_idx, opt_idx, hiddens):
        with self.trainer.profiler.profile('model_forward'):
            args = self.trainer.build_train_args(split_batch, batch_idx, opt_idx, hiddens)
            training_step_output = self.trainer.accelerator_backend.training_step(args)
            training_step_output = self.trainer.call_hook('training_step_end', training_step_output)

            # ----------------------------
            # PROCESS THE RESULT
            # ----------------------------
            # format and reduce outputs accordingly
            training_step_output_for_epoch_end = training_step_output
            is_result_obj = isinstance(training_step_output, Result)

            # track batch size for weighted average
            if is_result_obj:
                training_step_output.track_batch_size(len(split_batch))

            # don't allow EvalResult in the training_step
            if isinstance(training_step_output, EvalResult):
                raise MisconfigurationException('training_step cannot return EvalResult, '
                                                'use a dict or TrainResult instead')

            # handle regular dicts
            if not is_result_obj:
                training_step_output = self.trainer.process_output(training_step_output, train=True)

                training_step_output = AttributeDict(
                    batch_loss=training_step_output[0],
                    pbar_on_batch_end=training_step_output[1],
                    log_metrics=training_step_output[2],
                    callback_metrics=training_step_output[3],
                    hiddens=training_step_output[4],
                )

            # if the user decides to finally reduce things in epoch_end, save raw output without graphs
            if isinstance(training_step_output_for_epoch_end, torch.Tensor):
                training_step_output_for_epoch_end = training_step_output_for_epoch_end.detach()
            elif is_result_obj:
                training_step_output_for_epoch_end = copy(training_step_output)
                training_step_output_for_epoch_end.detach()
            else:
                training_step_output_for_epoch_end = recursive_detach(training_step_output_for_epoch_end)

        # accumulate loss
        # (if accumulate_grad_batches = 1 no effect)
        closure_loss = training_step_output.minimize if is_result_obj else training_step_output.batch_loss
        closure_loss = closure_loss / self.trainer.accumulate_grad_batches

        # the loss will get scaled for amp. avoid any modifications to it
        untouched_loss = closure_loss.detach().clone()

        # result
        result = AttributeDict(
            closure_loss=closure_loss,
            loss=untouched_loss,
            training_step_output=training_step_output,
            training_step_output_for_epoch_end=training_step_output_for_epoch_end,
            hiddens=training_step_output.hiddens,
        )
        return result
