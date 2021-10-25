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
from typing import Any, List, Optional
from weakref import proxy

from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class OptimizerConnector:
    def __init__(self, trainer: "pl.Trainer") -> None:
        self.trainer = proxy(trainer)

    def on_trainer_init(self) -> None:
        self.trainer.lr_schedulers = []
        self.trainer.optimizers = []
        self.trainer.optimizer_frequencies = []

    def update_learning_rates(
        self, interval: str, update_plateau_schedulers: bool, opt_indices: Optional[List[int]] = None
    ) -> None:
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            update_plateau_schedulers: control whether ``ReduceLROnPlateau`` or non-plateau schedulers get updated.
                This is used so non-plateau schedulers can be updated before running validation. Checkpoints are
                commonly saved during validation, however, on-plateau schedulers might monitor a validation metric
                so they have to be updated separately.
            opt_indices: indices of the optimizers to update.
        """
        if not self.trainer.lr_schedulers or not self.trainer.lightning_module.automatic_optimization:
            return

        if opt_indices is None:
            opt_indices = []

        for lr_scheduler in self.trainer.lr_schedulers:
            if isinstance(lr_scheduler["opt_idx"], int) and lr_scheduler["opt_idx"] not in opt_indices:
                continue

            if update_plateau_schedulers ^ lr_scheduler["reduce_on_plateau"]:
                continue

            # skip if `optimizer.step()` has never been called
            if (
                not isinstance(lr_scheduler["scheduler"], ReduceLROnPlateau)
                and lr_scheduler["scheduler"].optimizer._step_count == 0
            ):
                continue

            current_idx = self.trainer.fit_loop.batch_idx if interval == "step" else self.trainer.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler["interval"] == interval and current_idx % lr_scheduler["frequency"] == 0:
                monitor_val = None
                if lr_scheduler["reduce_on_plateau"]:
                    # If instance of ReduceLROnPlateau, we need a monitor
                    monitor_key = lr_scheduler["monitor"]
                    monitor_val = self._get_monitor_value(monitor_key)
                    if monitor_val is None:
                        if lr_scheduler.get("strict", True):
                            avail_metrics = list(self.trainer.callback_metrics)
                            raise MisconfigurationException(
                                f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                                f" which is not available. Available metrics are: {avail_metrics}."
                                " Condition can be set using `monitor` key in lr scheduler dict"
                            )
                        rank_zero_warn(
                            f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                            " which is not available but strict is set to `False`."
                            " Skipping learning rate update.",
                            RuntimeWarning,
                        )
                        continue

                self.trainer.fit_loop.epoch_loop.scheduler_progress.increment_ready()

                # update LR
                if lr_scheduler["reduce_on_plateau"]:
                    lr_scheduler["scheduler"].step(monitor_val)
                else:
                    lr_scheduler["scheduler"].step()

                self.trainer.fit_loop.epoch_loop.scheduler_progress.increment_completed()

    def _get_monitor_value(self, key: str) -> Any:
        # this is a separate method to aid in testing
        return self.trainer.callback_metrics.get(key)
