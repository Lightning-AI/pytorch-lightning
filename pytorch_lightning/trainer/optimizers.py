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

from abc import ABC
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers, LightningOptimizer
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import LRSchedulerConfig


class TrainerOptimizersMixin(ABC):
    r"""
    .. deprecated:: v1.6
        The `TrainerOptimizersMixin` was deprecated in v1.6 and will be removed in v1.8.
    """

    _lightning_optimizers: Optional[List[LightningOptimizer]]

    def init_optimizers(self, model: Optional["pl.LightningModule"]) -> Tuple[List, List, List]:
        r"""
        .. deprecated:: v1.6
            `TrainerOptimizersMixin.init_optimizers` was deprecated in v1.6 and will be removed in v1.8.
        """
        rank_zero_deprecation(
            "`TrainerOptimizersMixin.init_optimizers` was deprecated in v1.6 and will be removed in v1.8."
        )
        pl_module = self.lightning_module or model
        return _init_optimizers_and_lr_schedulers(pl_module)

    def convert_to_lightning_optimizers(self):
        r"""
        .. deprecated:: v1.6
            `TrainerOptimizersMixin.convert_to_lightning_optimizers` was deprecated in v1.6 and will be removed in v1.8.
        """
        rank_zero_deprecation(
            "`TrainerOptimizersMixin.convert_to_lightning_optimizers` was deprecated in v1.6 and will be removed in "
            "v1.8."
        )

        def _convert_to_lightning_optimizer(trainer, optimizer):
            if not isinstance(optimizer, LightningOptimizer):
                optimizer = LightningOptimizer(optimizer)
            optimizer._on_trainer_init(trainer)
            return optimizer

        self._lightning_optimizers = {
            opt_idx: _convert_to_lightning_optimizer(self, opt) for opt_idx, opt in enumerate(self.optimizers)
        }
<<<<<<< HEAD
=======

    @staticmethod
    def _configure_schedulers(
        schedulers: list, monitor: Optional[str], is_manual_optimization: bool
    ) -> List[LRSchedulerConfig]:
        """Convert each scheduler into dict structure with relevant information."""
        lr_schedulers = []
        default_config = _get_default_scheduler_config()
        for scheduler in schedulers:
            if is_manual_optimization:
                if isinstance(scheduler, dict):
                    invalid_keys = {"interval", "frequency", "reduce_on_plateau", "monitor", "strict"}
                    keys_to_warn = [k for k in scheduler.keys() if k in invalid_keys]

                    if keys_to_warn:
                        rank_zero_warn(
                            f"The lr scheduler dict contains the key(s) {keys_to_warn}, but the keys will be ignored."
                            " You need to call `lr_scheduler.step()` manually in manual optimization.",
                            category=RuntimeWarning,
                        )

                    scheduler = {key: scheduler[key] for key in scheduler if key not in invalid_keys}
                    lr_schedulers.append({**default_config, **scheduler})
                else:
                    lr_schedulers.append({**default_config, "scheduler": scheduler})
            else:
                if isinstance(scheduler, dict):
                    # check provided keys
                    extra_keys = [k for k in scheduler.keys() if k not in default_config.keys()]
                    if extra_keys:
                        rank_zero_warn(
                            f"Found unsupported keys in the lr scheduler dict: {extra_keys}", category=RuntimeWarning
                        )
                    if "scheduler" not in scheduler:
                        raise MisconfigurationException(
                            'The lr scheduler dict must have the key "scheduler" with its item being an lr scheduler'
                        )
                    if "interval" in scheduler and scheduler["interval"] not in ("step", "epoch"):
                        raise MisconfigurationException(
                            'The "interval" key in lr scheduler dict must be "step" or "epoch"'
                            f' but is "{scheduler["interval"]}"'
                        )
                    scheduler["reduce_on_plateau"] = isinstance(
                        scheduler["scheduler"], optim.lr_scheduler.ReduceLROnPlateau
                    )
                    if scheduler["reduce_on_plateau"] and scheduler.get("monitor", None) is None:
                        raise MisconfigurationException(
                            "The lr scheduler dict must include a monitor when a `ReduceLROnPlateau` scheduler is used."
                            ' For example: {"optimizer": optimizer, "lr_scheduler":'
                            ' {"scheduler": scheduler, "monitor": "your_loss"}}'
                        )
                    is_one_cycle = isinstance(scheduler["scheduler"], optim.lr_scheduler.OneCycleLR)
                    if is_one_cycle and scheduler.get("interval", "epoch") == "epoch":
                        rank_zero_warn(
                            "A `OneCycleLR` scheduler is using 'interval': 'epoch'."
                            " Are you sure you didn't mean 'interval': 'step'?",
                            category=RuntimeWarning,
                        )
                    lr_schedulers.append({**default_config, **scheduler})
                elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if monitor is None:
                        raise MisconfigurationException(
                            "`configure_optimizers` must include a monitor when a `ReduceLROnPlateau`"
                            " scheduler is used. For example:"
                            ' {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}'
                        )
                    lr_schedulers.append(
                        {**default_config, "scheduler": scheduler, "reduce_on_plateau": True, "monitor": monitor}
                    )
                elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                    lr_schedulers.append({**default_config, "scheduler": scheduler})
                else:
                    raise ValueError(f'The provided lr scheduler "{scheduler}" is invalid')
        return lr_schedulers


class _MockOptimizer(Optimizer):
    """The `_MockOptimizer` will be used inplace of an optimizer in the event that `None` is returned from
    `configure_optimizers`."""

    def __init__(self):
        super().__init__([torch.zeros(1)], {})

    def add_param_group(self, param_group):
        pass  # Do Nothing

    def load_state_dict(self, state_dict):
        pass  # Do Nothing

    def state_dict(self):
        return {}  # Return Empty

    def step(self, closure=None):
        if closure is not None:
            closure()

    def zero_grad(self):
        pass  # Do Nothing

    def __repr__(self):
        return "No Optimizer"


def _validate_optim_conf(optim_conf: Dict[str, Any]) -> None:
    valid_keys = {"optimizer", "lr_scheduler", "frequency", "monitor"}
    extra_keys = optim_conf.keys() - valid_keys
    if extra_keys:
        rank_zero_warn(
            f"Found unsupported keys in the optimizer configuration: {set(extra_keys)}", category=RuntimeWarning
        )


def _validate_scheduler_optimizer(optimizers, lr_schedulers):
    if any(sch["scheduler"].optimizer not in optimizers for sch in lr_schedulers):
        raise MisconfigurationException(
            "Some schedulers are attached with an optimizer that wasn't returned from `configure_optimizers`."
        )


def _get_default_scheduler_config() -> Dict[str, Any]:
    return {
        "scheduler": None,
        "name": None,  # no custom name
        "interval": "epoch",  # after epoch is over
        "frequency": 1,  # every epoch/batch
        "reduce_on_plateau": False,  # most often not ReduceLROnPlateau scheduler
        "monitor": None,  # value to monitor for ReduceLROnPlateau
        "strict": True,  # enforce that the monitor exists for ReduceLROnPlateau
        "opt_idx": None,  # necessary to store opt_idx when optimizer frequencies are specified
    }
>>>>>>> eb5b350f9a6bd27a66dfebcb00b3acb33b7bbb89
