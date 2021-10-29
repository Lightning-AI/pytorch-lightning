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
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional
from weakref import proxy

from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def do_nothing_closure() -> None:
    return


class LightningOptimizer:
    """This class is used to wrap the user optimizers and handle properly the backward and optimizer_step logic
    across accelerators, AMP, accumulate_grad_batches."""

    def __init__(self, optimizer: Optimizer):
        # copy most of the `Optimizer` methods into this instance. `__del__` is skipped in case the optimizer has
        # implemented custom logic which we would not want to call on destruction of the `LightningOptimizer`
        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k not in ("step", "__del__")}

        # For Horovod
        if hasattr(optimizer, "skip_synchronize"):
            self.__class__ = type(
                "Lightning" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__.__bases__[0]), {}
            )
            self.skip_synchronize = optimizer.skip_synchronize
            self.synchronize = optimizer.synchronize
        else:
            self.__class__ = type("Lightning" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})

        self._optimizer = optimizer
        self._trainer: Optional["pl.Trainer"] = None
        self._optimizer_idx = 0

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def _on_trainer_init(self, trainer: "pl.Trainer") -> None:
        self._trainer = proxy(trainer)
        for opt_idx, opt in enumerate(trainer.optimizers):
            if opt == self._optimizer:
                self._optimizer_idx = opt_idx
                break

    @classmethod
    def _to_lightning_optimizer(cls, optimizer: Optimizer, trainer: "pl.Trainer", opt_idx: int) -> "LightningOptimizer":
        # apex overrides .step function and need to be wrapped on each step
        if trainer.amp_backend is not None and trainer.amp_backend == AMPType.APEX:
            lightning_optimizer = cls(optimizer)
            lightning_optimizer._on_trainer_init(trainer)
        else:
            lightning_optimizer = trainer.lightning_optimizers[opt_idx]
        return lightning_optimizer

    @contextmanager
    def toggle_model(self, sync_grad: bool = True) -> Generator[None, None, None]:
        """This function is just a helper for advanced users.

        Considering the current optimizer as A and all other optimizers as B.
        Toggling means all parameters from B exclusive to A will have ``requires_grad`` set to False.

        When performing gradient accumulation, there is no need to perform grad synchronization
        during the accumulation phase.
        Setting `sync_grad` to False will block this synchronization and improve performance.
        """
        # local import here to avoid circular import
        from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior

        assert self._trainer is not None
        lightning_module = self._trainer.lightning_module

        with _block_parallel_sync_behavior(self._trainer, block=(not sync_grad)):
            lightning_module.toggle_optimizer(self, self._optimizer_idx)
            yield
            lightning_module.untoggle_optimizer(self._optimizer_idx)

    def step(self, closure: Optional[Callable[[], Any]] = None, **kwargs: Any) -> None:
        """Performs a single optimization step (parameter update).

        Args:
            closure: An optional optimizer_closure.
            kwargs: Any additional arguments to the ``optimizer.step()`` call.

        Example::

            # Scenario for a GAN using manual optimization
            def training_step(...):
                opt_gen, opt_dis = self.optimizers()

                ...

                # compute generator loss
                loss_gen = self.compute_generator_loss(...)
                # zero_grad needs to be called before backward
                opt_gen.zero_grad()
                self.manual_backward(loss_gen)
                opt_gen.step()

                # compute discriminator loss
                loss_dis = self.compute_discriminator_loss(...)

                # zero_grad needs to be called before backward
                opt_dis.zero_grad()
                self.manual_backward(loss_dis)
                opt_dis.step()


            # A more advanced example
            def training_step(self, batch, batch_idx, ...):
                opt_gen, opt_dis = self.optimizers()

                ...
                accumulated_grad_batches = batch_idx % 2 == 0

                # compute generator loss
                def closure_gen():
                    loss_gen = self.compute_generator_loss(...)
                    self.manual_backward(loss_gen)
                    if accumulated_grad_batches:
                        opt_gen.zero_grad()

                with opt_gen.toggle_model(sync_grad=accumulated_grad_batches):
                    opt_gen.step(closure=closure_gen)

                def closure_dis():
                    loss_dis = self.compute_discriminator_loss(...)
                    self.manual_backward(loss_dis)
                    if accumulated_grad_batches:
                        opt_dis.zero_grad()

                with opt_dis.toggle_model(sync_grad=accumulated_grad_batches):
                    opt_dis.step(closure=closure_dis)
        """
        if closure is None:
            closure = do_nothing_closure
            profiler_action = "optimizer_step_without_closure"
        elif not callable(closure):
            raise MisconfigurationException("When `optimizer.step(closure)` is called, the closure should be callable")
        else:
            profiler_action = "optimizer_step_with_closure"
        profiler_action += f"_{self._optimizer_idx}"

        trainer = self._trainer
        assert trainer is not None
        with trainer.profiler.profile(profiler_action):
            trainer.accelerator.optimizer_step(self._optimizer, self._optimizer_idx, closure, **kwargs)
