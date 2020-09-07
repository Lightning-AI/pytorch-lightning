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
from pytorch_lightning.core import memory


class LoggerConnector:

    def __init__(self, trainer):
        self.trainer = trainer
        self.callback_metrics = {}
        self.logged_metrics = {}

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
        if self.trainer.on_gpu and self.trainer.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.trainer.log_gpu_memory)
            metrics.update(mem_map)

        # add norms
        metrics.update(grad_norm_dic)

        # turn all tensors to scalars
        scalar_metrics = self.trainer.metrics_to_scalars(metrics)

        if "step" in scalar_metrics and step is None:
            step = scalar_metrics.pop("step")

        elif step is None:
            # added metrics by Lightning for convenience
            scalar_metrics['epoch'] = self.trainer.current_epoch
            step = step if step is not None else self.trainer.global_step

        # log actual metrics
        if self.trainer.is_global_zero and self.trainer.logger is not None:
            self.trainer.logger.agg_and_log_metrics(scalar_metrics, step=step)
            self.trainer.logger.save()

            # track the logged metrics
            self.logged_metrics = scalar_metrics
            self.trainer.dev_debugger.track_logged_metrics_history(scalar_metrics)
