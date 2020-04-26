from abc import ABC
from typing import Union, Iterable

import torch

from pytorch_lightning.core import memory
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase, LoggerCollection
from pytorch_lightning.utilities.memory import recursive_detach


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

        # ---------------
        # EXTRACT HIDDEN
        # ---------------
        hiddens = output.get('hiddens')

        # use every metric passed in as a candidate for callback
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)

        # detach all metrics for callbacks to prevent memory leaks
        # no .item() because it will slow things down
        callback_metrics = recursive_detach(callback_metrics)

        return loss, progress_bar_metrics, log_metrics, callback_metrics, hiddens

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
