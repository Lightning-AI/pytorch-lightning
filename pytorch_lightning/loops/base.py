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

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

from deprecate import void
from torchmetrics import Metric

import pytorch_lightning as pl
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.progress import BaseProgress
from pytorch_lightning.utilities.exceptions import MisconfigurationException

T = TypeVar("T")  # the output type of `run`


class Loop(ABC, Generic[T]):
    """Basic Loops interface. All classes derived from this must implement the following properties and methods:

        * :attr:`done` (property): Condition to break the loop
        * :attr:`reset` (method): Resets the internal state between multiple calls of :attr:`run`
        * :attr:`advance` (method): Implements one step of the loop

    This class implements the following loop structure:

    .. code-block:: python

        on_run_start()

        while not done:
            on_advance_start()
            advance()
            on_advance_end()

        on_run_end()
    """

    def __init__(self) -> None:
        self.restarting = False
        self._trainer: Optional["pl.Trainer"] = None

    @property
    def trainer(self) -> "pl.Trainer":
        if self._trainer is None:
            raise RuntimeError("The loop is not attached to a Trainer.")
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: "pl.Trainer"):
        """Connects this loop's trainer and its children."""
        if not isinstance(trainer, pl.Trainer):
            raise MisconfigurationException(
                f"Loop {self.__class__.__name__} should be connected to a `Trainer`, found: {trainer}."
            )
        self._trainer = trainer
        for v in self.__dict__.values():
            if isinstance(v, Loop):
                v.trainer = trainer

    @property
    @abstractmethod
    def done(self) -> bool:
        """Property indicating when the loop is finished.

        Example::

            @property
            def done(self):
                return self.trainer.global_step >= self.trainer.max_steps
        """

    @property
    def skip(self) -> bool:
        """Determine whether to return immediately from the call to :meth:`run`.

        Example::

            @property
            def skip(self):
                return len(self.trainer.train_dataloader) == 0
        """
        return False

    def connect(self, **kwargs: "Loop") -> None:
        """Optionally connect one or multiple loops to this one.

        Linked loops should form a tree.
        """

    def on_skip(self) -> Optional[Any]:
        """The function to run when :meth:`run` should be skipped, determined by the condition in :attr:`skip`.

        Returns:
            the default output value of :meth:`on_run_end`
        """

    def run(self, *args: Any, **kwargs: Any) -> T:
        """The main entry point to the loop.

        Will frequently check the :attr:`done` condition and calls :attr:`advance`
        until :attr:`done` evaluates to ``True``.

        Override this if you wish to change the default behavior. The default implementation is:

        Example::

            def run(self, *args, **kwargs):
                if self.skip:
                    return self.on_skip()

                self.reset()
                self.on_run_start(*args, **kwargs)

                while not self.done:
                    self.advance(*args, **kwargs)

                output = self.on_run_end()
                return output

        Returns:
            The output of :attr:`on_run_end` (often outputs collected from each step of the loop)
        """
        if self.skip:
            return self.on_skip()

        self.reset()

        self.on_run_start(*args, **kwargs)

        while not self.done:
            try:
                self.on_advance_start(*args, **kwargs)
                self.advance(*args, **kwargs)
                self.on_advance_end()
                self.restarting = False
            except StopIteration:
                break

        output = self.on_run_end()
        return output

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state of the loop at the beginning of each call to :attr:`run`.

        Example::

            def reset(self):
                # reset your internal state or add custom logic
                # if you expect run() to be called multiple times
                self.current_iteration = 0
                self.outputs = []
        """

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Hook to be called as the first thing after entering :attr:`run` (except the state reset).

        Accepts all arguments passed to :attr:`run`.
        """
        void(*args, **kwargs)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Hook to be called each time before :attr:`advance` is called.

        Accepts all arguments passed to :attr`run`.
        """
        void(*args, **kwargs)

    @abstractmethod
    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs a single step.

        Accepts all arguments passed to :attr:`run`.

        Example::

            def advance(self, iterator):
                batch = next(iterator)
                loss = self.trainer.lightning_module.training_step(batch, batch_idx)
                ...
        """

    def on_advance_end(self) -> None:
        """Hook to be called each time after :attr:`advance` is called."""

    def on_run_end(self) -> T:
        """Hook to be called at the end of the run.

        Its return argument is returned from :attr:`run`.
        """

    def teardown(self) -> None:
        """Use to release memory etc."""

    def on_save_checkpoint(self) -> Dict:
        """Called when saving a model checkpoint, use to persist loop state.

        Returns:
            The current loop state.
        """
        return {}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        """Called when loading a model checkpoint, use to reload loop state."""

    def state_dict(self, destination: Optional[Dict] = None, prefix: Optional[str] = "") -> Dict:
        """The state dict is determined by the state and progress of this loop and all its children.

        Args:
            destination: An existing dictionary to update with this loop's state. By default a new dictionary
                is returned.
            prefix: A prefix for each key in the state dictionary
        """
        if destination is None:
            destination = {}

        destination[prefix + "state_dict"] = self.on_save_checkpoint()

        for k, v in self.__dict__.items():
            key = prefix + k
            if isinstance(v, BaseProgress):
                destination[key] = v.state_dict()
            elif isinstance(v, Loop):
                v.state_dict(destination, key + ".")
            elif isinstance(v, ResultCollection):
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

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, metrics: Optional[Dict[str, Metric]] = None) -> None:
        for k, v in self.__dict__.items():
            key = prefix + k
            if isinstance(v, BaseProgress):
                v.load_state_dict(state_dict[key])
            elif (
                isinstance(v, ResultCollection)
                and self.trainer is not None
                and getattr(self.trainer, "lightning_module", None) is not None
            ):
                metric_attributes = {
                    name: module
                    for name, module in self.trainer.lightning_module.named_modules()
                    if isinstance(module, Metric)
                }
                if metrics:
                    metric_attributes.update(metrics)

                # The `ResultCollection` objects have 2 types of metrics: `Tensor` and `torchmetrics.Metric`.
                # When creating a checkpoint, the `Metric`s are dropped from the loop `state_dict` to serialize only
                # Python primitives. However, their states are saved with the model's `state_dict`.
                # On reload, we need to re-attach the `Metric`s back to the `ResultCollection`.
                # The references are provided through the `metric_attributes` dictionary.
                v.load_state_dict(
                    state_dict[prefix + k], metrics=metric_attributes, sync_fn=self.trainer.training_type_plugin.reduce
                )

                if not self.trainer.is_global_zero:
                    v.reset(metrics=False)

        self.on_load_checkpoint(state_dict[prefix + "state_dict"])
        self.restarting = True
