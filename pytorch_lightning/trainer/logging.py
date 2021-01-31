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

import inspect
from abc import ABC
from typing import Mapping, Union

import torch

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import DeviceType, DistributedType
from pytorch_lightning.utilities.distributed import rank_zero_warn
from pytorch_lightning.utilities.memory import recursive_detach


class TrainerLoggingMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    current_epoch: int
    _device_type: DeviceType
    _distrib_type: DistributedType
    log_gpu_memory:...
    logger: Union[LightningLoggerBase, bool]
    global_step: int
    global_rank: int
    default_root_dir: str
    slurm_job_id: int
    num_gpus: int
    logged_metrics:...

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def process_dict_result(self, output, train=False):
        """Reduces output according to the training mode.

        Separates loss from logging and progress bar metrics
        """
        # --------------------
        # WARN DEPRECATED KEYS
        # --------------------
        # TODO: 1.0.0 remove
        if isinstance(output, dict):
            for k, v in output.items():
                if k in ['log', 'progress_bar']:
                    m = inspect.cleandoc(
                        f"The {{{k}:dict keyword}} was deprecated in 0.9.1 and will be removed in 1.0.0\n"
                        " Please use self.log(...) inside the lightningModule instead.\n"
                        " # log on a step or aggregate epoch metric to the logger and/or progress bar"
                        " (inside LightningModule)\n"
                        " self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)"
                    )
                    rank_zero_warn(m)

        # --------------------------
        # handle single scalar only
        # --------------------------
        # single scalar returned from a xx_step
        if isinstance(output, torch.Tensor):
            progress_bar_metrics = {}
            log_metrics = {}
            callback_metrics = {}
            hiddens = None
            return output, progress_bar_metrics, log_metrics, callback_metrics, hiddens

        # ---------------
        # EXTRACT CALLBACK KEYS
        # ---------------
        # all keys not progress_bar or log are candidates for callbacks
        callback_metrics = {}
        if isinstance(output, Mapping):
            for k, v in output.items():
                if k not in ['progress_bar', 'log', 'hiddens']:
                    callback_metrics[k] = v

        if train and self._distrib_type in (DistributedType.DP, DistributedType.DDP2):
            num_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, num_gpus)

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for progress bar when using dp
            if train and self._distrib_type in (DistributedType.DP, DistributedType.DDP2):
                num_gpus = self.num_gpus
                progress_output = self.reduce_distributed_output(progress_output, num_gpus)

            progress_bar_metrics = progress_output
        # todo: specify the possible exception
        except Exception:
            progress_bar_metrics = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for progress bar when using dp
            if train and self._distrib_type in (DistributedType.DP, DistributedType.DDP2):
                num_gpus = self.num_gpus
                log_output = self.reduce_distributed_output(log_output, num_gpus)

            log_metrics = log_output
        # todo: specify the possible exception
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
            # todo: specify the possible exception
            except Exception as exp:
                if isinstance(output, torch.Tensor):
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    ) from exp

            # when using dp need to reduce the loss
            if self._distrib_type in (DistributedType.DP, DistributedType.DDP2):
                loss = self.reduce_distributed_output(loss, self.num_gpus)

        # ---------------
        # EXTRACT HIDDEN
        # ---------------
        hiddens = output.get('hiddens', None) if isinstance(output, Mapping) else None

        # use every metric passed in as a candidate for callback
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)

        # detach all metrics for callbacks to prevent memory leaks
        # no .item() because it will slow things down
        callback_metrics = recursive_detach(callback_metrics)
        progress_bar_metrics = recursive_detach(progress_bar_metrics)
        log_metrics = recursive_detach(log_metrics)

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

            # compute the average of scalars
            elif isinstance(output[k], list):
                output[k] = sum(output[k]) / len(output[k])

            # do nothing when there's a scalar
            elif isinstance(output[k], torch.Tensor) and output[k].dim() == 0:
                pass

            # do not reduce metrics that have batch size > num gpus
            elif output[k].size(0) <= num_gpus:
                output[k] = torch.mean(output[k])

        return output
