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
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import optim
from torch.optim.optimizer import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TrainerOptimizersMixin(ABC):

    def init_optimizers(self, model: LightningModule) -> Tuple[List, List, List]:
        optim_conf = model.configure_optimizers()
        if optim_conf is None:
            rank_zero_warn(
                '`LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer',
                UserWarning,
            )
            optim_conf = _MockOptimizer()

        optimizers, lr_schedulers, optimizer_frequencies = [], [], []
        monitor = None

        # single output, single optimizer
        if isinstance(optim_conf, Optimizer):
            optimizers = [optim_conf]
        # two lists, optimizer + lr schedulers
        elif isinstance(optim_conf, (list, tuple)) and len(optim_conf) == 2 and isinstance(optim_conf[0], list):
            opt, sch = optim_conf
            optimizers = opt
            lr_schedulers = sch if isinstance(sch, list) else [sch]
        # single dictionary
        elif isinstance(optim_conf, dict):
            optimizers = [optim_conf["optimizer"]]
            monitor = optim_conf.get('monitor', None)
            lr_schedulers = [optim_conf["lr_scheduler"]] if "lr_scheduler" in optim_conf else []
        # multiple dictionaries
        elif isinstance(optim_conf, (list, tuple)) and all(isinstance(d, dict) for d in optim_conf):
            optimizers = [opt_dict["optimizer"] for opt_dict in optim_conf]
            lr_schedulers = [opt_dict["lr_scheduler"] for opt_dict in optim_conf if "lr_scheduler" in opt_dict]
            optimizer_frequencies = [
                opt_dict["frequency"] for opt_dict in optim_conf if opt_dict.get("frequency", None) is not None
            ]
            # assert that if frequencies are present, they are given for all optimizers
            if optimizer_frequencies and len(optimizer_frequencies) != len(optimizers):
                raise ValueError("A frequency must be given to each optimizer.")
        # single list or tuple, multiple optimizer
        elif isinstance(optim_conf, (list, tuple)):
            optimizers = list(optim_conf)
        # unknown configuration
        else:
            raise MisconfigurationException(
                'Unknown configuration for model optimizers.'
                ' Output from `model.configure_optimizers()` should either be:\n'
                ' * `torch.optim.Optimizer`\n'
                ' * [`torch.optim.Optimizer`]\n'
                ' * ([`torch.optim.Optimizer`], [`torch.optim.lr_scheduler`])\n'
                ' * {"optimizer": `torch.optim.Optimizer`, (optional) "lr_scheduler": `torch.optim.lr_scheduler`}\n'
                ' * A list of the previously described dict format, with an optional "frequency" key (int)'
            )

        lr_schedulers = self.configure_schedulers(lr_schedulers, monitor=monitor)
        _validate_scheduler_optimizer(optimizers, lr_schedulers)

        return optimizers, lr_schedulers, optimizer_frequencies

    def convert_to_lightning_optimizers(self):

        def _convert_to_lightning_optimizer(trainer, optimizer):
            if not isinstance(optimizer, LightningOptimizer):
                optimizer = LightningOptimizer(optimizer)
            optimizer._on_trainer_init(trainer)
            return optimizer

        self._lightning_optimizers = {
            opt_idx: _convert_to_lightning_optimizer(self, opt)
            for opt_idx, opt in enumerate(self.optimizers)
        }

    def configure_schedulers(self, schedulers: list, monitor: Optional[str] = None):
        # Convert each scheduler into dict structure with relevant information
        lr_schedulers = []
        default_config = _get_default_scheduler_config()
        for scheduler in schedulers:
            if isinstance(scheduler, dict):
                # check provided keys
                extra_keys = [k for k in scheduler.keys() if k not in default_config.keys()]
                if extra_keys:
                    rank_zero_warn(f'Found unsupported keys in the lr scheduler dict: {extra_keys}', RuntimeWarning)
                if 'scheduler' not in scheduler:
                    raise MisconfigurationException(
                        'The lr scheduler dict must have the key "scheduler" with its item being an lr scheduler'
                    )
                if 'interval' in scheduler and scheduler['interval'] not in ('step', 'epoch'):
                    raise MisconfigurationException(
                        f'The "interval" key in lr scheduler dict must be "step" or "epoch"'
                        f' but is "{scheduler["interval"]}"'
                    )

                scheduler['reduce_on_plateau'] = isinstance(
                    scheduler['scheduler'], optim.lr_scheduler.ReduceLROnPlateau
                )
                if scheduler['reduce_on_plateau'] and scheduler.get('monitor', None) is None:
                    raise MisconfigurationException(
                        'The lr scheduler dict must include a monitor when a `ReduceLROnPlateau` scheduler is used.'
                        ' For example: {"optimizer": optimizer, "lr_scheduler":'
                        ' {"scheduler": scheduler, "monitor": "your_loss"}}'
                    )
                lr_schedulers.append({**default_config, **scheduler})
            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if monitor is None:
                    raise MisconfigurationException(
                        '`configure_optimizers` must include a monitor when a `ReduceLROnPlateau` scheduler is used.'
                        ' For example:'
                        ' {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}'
                    )
                lr_schedulers.append({
                    **default_config, 'scheduler': scheduler,
                    'reduce_on_plateau': True,
                    'monitor': monitor
                })
            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                lr_schedulers.append({**default_config, 'scheduler': scheduler})
            else:
                raise ValueError(f'The provided lr scheduler "{scheduler}" is invalid')
        return lr_schedulers


class _MockOptimizer(Optimizer):
    """The `_MockOptimizer` will be used inplace of an optimizer in the event that `None`
    is returned from `configure_optimizers`.
    """

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
        return 'No Optimizer'


def _validate_scheduler_optimizer(optimizers, lr_schedulers):
    if any(sch['scheduler'].optimizer not in optimizers for sch in lr_schedulers):
        raise MisconfigurationException(
            "Some schedulers are attatched with an optimizer that wasn't returned from `configure_optimizers`."
        )


def _get_default_scheduler_config() -> Dict[str, Any]:
    return {
        'scheduler': None,
        'name': None,  # no custom name
        'interval': 'epoch',  # after epoch is over
        'frequency': 1,  # every epoch/batch
        'reduce_on_plateau': False,  # most often not ReduceLROnPlateau scheduler
        'monitor': None,  # value to monitor for ReduceLROnPlateau
        'strict': True,  # enforce that the monitor exists for ReduceLROnPlateau
    }
