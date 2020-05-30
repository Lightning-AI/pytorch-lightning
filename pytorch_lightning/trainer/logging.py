from abc import ABC
from typing import Union, Iterable

import torch

from pytorch_lightning.core import memory
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase, LoggerCollection
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.utilities.parsing import AttributeDict


class TrainerLoggingMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    current_epoch: int
    on_gpu: bool
    log_gpu_memory: ...
    logger: Union[LightningLoggerBase, bool]
    progress_bar_metrics: ...
    global_step: int
    proc_rank: int
    use_dp: bool
    use_ddp2: bool
    default_root_dir: str
    slurm_job_id: int
    num_gpus: int

    def configure_logger(self, logger):
        if logger is True:
            # default logger
            self.logger = TensorBoardLogger(
                save_dir=self.default_root_dir,
                version=self.slurm_job_id,
                name='lightning_logs'
            )
        elif logger is False:
            self.logger = None
        else:
            if isinstance(logger, Iterable):
                self.logger = LoggerCollection(logger)
            else:
                self.logger = logger

    def log_metrics(self, metrics, grad_norm_dic, step=None):
        """Logs the metric dict passed in.
        If `step` parameter is None and `step` key is presented is metrics,
        uses metrics["step"] as a step

        Args:
            metrics (dict): Metric values
            grad_norm_dic (dict): Gradient norms
            step (int): Step for which metrics should be logged. Default value corresponds to `self.global_step`
        """
        # add gpu memory
        if self.on_gpu and self.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.log_gpu_memory)
            metrics.update(mem_map)

        # add norms
        metrics.update(grad_norm_dic)

        # turn all tensors to scalars
        scalar_metrics = self.metrics_to_scalars(metrics)

        if "step" in scalar_metrics and step is None:
            step = scalar_metrics.pop("step")
        else:
            # added metrics by Lightning for convenience
            scalar_metrics['epoch'] = self.current_epoch
            step = step if step is not None else self.global_step
        # log actual metrics
        if self.proc_rank == 0 and self.logger is not None:
            self.logger.agg_and_log_metrics(scalar_metrics, step=step)
            self.logger.save()

    def add_progress_bar_metrics(self, metrics):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.progress_bar_metrics[k] = v

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def process_output(self, output, train=False):
        # TODO: DEPRECATE FULL FUNCTION IN 1.0.0. Use process_step_result instead
        """Reduces output according to the training mode.
        Separates loss from logging and progress bar metrics
        """
        # ---------------
        # EXTRACT CALLBACK KEYS
        # ---------------
        # all keys not progress_bar or log are candidates for callbacks
        callback_metrics = {}
        for k, v in output.items():
            if k not in ['progress_bar', 'log', 'hiddens']:
                callback_metrics[k] = v

        if train and (self.use_dp or self.use_ddp2):
            num_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, num_gpus)

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                progress_output = self.reduce_distributed_output(progress_output, num_gpus)

            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                log_output = self.reduce_distributed_output(log_output, num_gpus)

            log_metrics = log_output
        except Exception:
            log_metrics = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception:
                if isinstance(output, torch.Tensor):
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    )

            # when using dp need to reduce the loss
            if self.use_dp or self.use_ddp2:
                loss = self.reduce_distributed_output(loss, self.num_gpus)

        # use every metric passed in as a candidate for callback
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)

        # detach all metrics for callbacks to prevent memory leaks
        # no .item() because it will slow things down
        callback_metrics = recursive_detach(callback_metrics)

        result = AttributeDict(
            batch_loss=loss,
            pbar_on_batch_end=progress_bar_metrics,
            log_on_batch_end=log_metrics,
            callback_metrics=callback_metrics,
            hiddens=output.get('hiddens')
        )
        return result

    def process_step_result(self, step_result: Result, train=False):
        """
        Reduces output according to the training mode.
        Separates loss from logging and progress bar metrics
        """

        # for callbacks we only use two metrics right now
        # checkpoint_on and early_stop_on
        # import pdb; pdb.set_trace()
        callback_metrics = dict()
        if 'checkpoint_on' in step_result:
            callback_metrics['checkpoint_on'] = step_result.checkpoint_on
        if 'early_stop_on' in step_result:
            callback_metrics['early_stop_on'] = step_result.early_stop_on

        if train and (self.use_dp or self.use_ddp2):
            # {val: [x1, x2], ...} -> {val: x1_2_mean}
            num_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, num_gpus)

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            pbar_on_batch_end = step_result.get('pbar_on_batch_end', {})
            pbar_on_epoch_end = step_result.get('pbar_on_epoch_end', {})

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                pbar_on_batch_end = self.reduce_distributed_output(pbar_on_batch_end, num_gpus)
                pbar_on_epoch_end = self.reduce_distributed_output(pbar_on_batch_end, num_gpus)

        except Exception:
            pbar_on_batch_end = {}
            pbar_on_epoch_end = {}

        # ---------------
        # EXTRACT pass to epoch end
        # ---------------
        try:
            pass_to_epoch_end = step_result.get('pass_to_epoch_end', {})

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                pass_to_epoch_end = self.reduce_distributed_output(pass_to_epoch_end, num_gpus)

        except Exception:
            pass_to_epoch_end = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_on_batch_end = step_result.get('log_on_batch_end', {})
            log_on_epoch_end = step_result.get('log_on_epoch_end', {})

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                log_on_batch_end = self.reduce_distributed_output(log_on_batch_end, num_gpus)
                log_on_epoch_end = self.reduce_distributed_output(log_on_epoch_end, num_gpus)

        except Exception:
            log_on_batch_end = {}
            log_on_epoch_end = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = step_result.minimize
            except Exception:
                if isinstance(step_result, torch.Tensor):
                    loss = step_result
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    )

            # when using dp need to reduce the loss
            if self.use_dp or self.use_ddp2:
                loss = self.reduce_distributed_output(loss, self.num_gpus)

        # TODO: good place to dist all_reduce to aggregate metrics across gpus
        result = AttributeDict(
            batch_loss=loss,
            pbar_on_batch_end=pbar_on_batch_end,
            pbar_on_epoch_end=pbar_on_epoch_end,
            log_on_batch_end=log_on_batch_end,
            log_on_epoch_end=log_on_epoch_end,
            callback_metrics=callback_metrics,
            hiddens=step_result.get('hiddens'),
            pass_to_epoch_end=pass_to_epoch_end,
            pass_to_batch_end=step_result.get('pass_to_batch_end', {})
        )
        return result

    def reduce_distributed_output(self, output, num_gpus):
        if num_gpus <= 1:
            return output

        # when using DP, we get one output per gpu
        # average outputs and return
        if isinstance(output, torch.Tensor):
            return output.mean()

        for k, v in output.items():
            # recurse on nested dics
            if isinstance(output[k], dict):
                output[k] = self.reduce_distributed_output(output[k], num_gpus)

            # do nothing when there's a scalar
            elif isinstance(output[k], torch.Tensor) and output[k].dim() == 0:
                pass

            # do not reduce metrics that have batch size > num gpus
            elif output[k].size(0) <= num_gpus:
                output[k] = torch.mean(output[k])

        return output
