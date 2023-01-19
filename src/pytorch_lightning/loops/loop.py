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
from typing import Dict, Optional

from torchmetrics import Metric

import pytorch_lightning as pl
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.progress import BaseProgress
from pytorch_lightning.utilities.imports import _fault_tolerant_training


class Loop:
    """Basic Loops interface."""

    def __init__(self) -> None:
        self._restarting = False
        self._trainer: Optional["pl.Trainer"] = None

    @property
    def trainer(self) -> "pl.Trainer":
        if self._trainer is None:
            raise RuntimeError("The loop is not attached to a Trainer.")
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: "pl.Trainer") -> None:
        """Connects this loop's trainer and its children."""
        self._trainer = trainer
        for v in self.__dict__.values():
            if isinstance(v, Loop):
                v.trainer = trainer

    @property
    def restarting(self) -> bool:
        """Whether the state of this loop was reloaded and it needs to restart."""
        return self._restarting

    @restarting.setter
    def restarting(self, restarting: bool) -> None:
        """Connects this loop's restarting value and its children."""
        self._restarting = restarting
        for loop in vars(self).values():
            if isinstance(loop, Loop):
                loop.restarting = restarting

    def on_save_checkpoint(self) -> Dict:
        """Called when saving a model checkpoint, use to persist loop state.

        Returns:
            The current loop state.
        """
        return {}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        """Called when loading a model checkpoint, use to reload loop state."""

    def state_dict(self, destination: Optional[Dict] = None, prefix: str = "") -> Dict:
        """The state dict is determined by the state and progress of this loop and all its children.

        Args:
            destination: An existing dictionary to update with this loop's state. By default a new dictionary
                is returned.
            prefix: A prefix for each key in the state dictionary
        """
        if destination is None:
            destination = {}

        destination[prefix + "state_dict"] = self.on_save_checkpoint()

        # do not get the mode from `self.trainer` because it might not have been attached yet
        ft_enabled = _fault_tolerant_training()
        for k, v in self.__dict__.items():
            key = prefix + k
            if isinstance(v, BaseProgress):
                destination[key] = v.state_dict()
            elif isinstance(v, Loop):
                v.state_dict(destination, key + ".")
            elif ft_enabled and isinstance(v, _ResultCollection):
                # sync / unsync metrics
                v.sync()
                destination[key] = v.state_dict()
                v.unsync()

        return destination

    def load_state_dict(
        self,
        state_dict: Dict,
        prefix: str = "",
        metrics: Optional[Dict[str, Metric]] = None,
    ) -> None:
        """Loads the state of this loop and all its children."""
        self._load_from_state_dict(state_dict.copy(), prefix, metrics)
        for k, v in self.__dict__.items():
            if isinstance(v, Loop):
                v.load_state_dict(state_dict.copy(), prefix + k + ".")
        self.restarting = True

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, metrics: Optional[Dict[str, Metric]] = None) -> None:
        trainer = self._trainer
        for k, v in self.__dict__.items():
            key = prefix + k
            if key not in state_dict:
                # compatibility with old checkpoints
                continue

            if isinstance(v, BaseProgress):
                v.load_state_dict(state_dict[key])
            elif isinstance(v, _ResultCollection) and trainer is not None and trainer.lightning_module is not None:
                metric_attributes = {
                    name: module
                    for name, module in self.trainer.lightning_module.named_modules()
                    if isinstance(module, Metric)
                }
                if metrics:
                    metric_attributes.update(metrics)

                # The `_ResultCollection` objects have 2 types of metrics: `Tensor` and `torchmetrics.Metric`.
                # When creating a checkpoint, the `Metric`s are dropped from the loop `state_dict` to serialize only
                # Python primitives. However, their states are saved with the model's `state_dict`.
                # On reload, we need to re-attach the `Metric`s back to the `_ResultCollection`.
                # The references are provided through the `metric_attributes` dictionary.
                v.load_state_dict(state_dict[key], metrics=metric_attributes, sync_fn=self.trainer.strategy.reduce)

                if not self.trainer.is_global_zero:
                    v.reset(metrics=False)

        if prefix + "state_dict" in state_dict:  # compatibility with old checkpoints
            self.on_load_checkpoint(state_dict[prefix + "state_dict"])
