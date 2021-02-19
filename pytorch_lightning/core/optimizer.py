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
import types
from contextlib import contextmanager
from typing import Callable, Optional
from weakref import proxy

from torch.optim import Optimizer

from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def is_lightning_optimizer(optimizer):
    return isinstance(optimizer, LightningOptimizer)


def do_nothing_closure():
    return


class LightningOptimizer:
    """
    This class is used to wrap the user optimizers and handle properly
    the backward and optimizer_step logic across accelerators, AMP, accumulate_grad_batches
    """

    def __init__(self, optimizer: Optimizer):

        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k != 'step'}

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
        self._trainer = None
        self._optimizer_idx = None
        self._total_optimizer_step_calls = 0

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def defaults(self):
        return self._optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self._optimizer.defaults = defaults

    @property
    def state(self):
        return self._optimizer.state

    @state.setter
    def state(self, state):
        self._optimizer.state = state

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self._optimizer.param_groups = param_groups

    def _on_trainer_init(self, trainer):
        self._trainer = proxy(trainer)
        for opt_idx, opt in enumerate(trainer.optimizers):
            if opt == self._optimizer:
                self._optimizer_idx = opt_idx
                break

    @classmethod
    def _to_lightning_optimizer(cls, optimizer, trainer, opt_idx):
        # apex overrides .step function and need to be wrapped on each step
        if trainer.amp_backend == AMPType.APEX:
            optimizer = cls(optimizer)
            optimizer._on_trainer_init(trainer)
        else:
            optimizer = trainer.lightning_optimizers[opt_idx]
        return optimizer

    def _toggle_model(self):
        model_ref = self._trainer.lightning_module
        model_ref.toggle_optimizer(self, self._optimizer_idx)

    def _untoggle_model(self):
        model_ref = self._trainer.lightning_module
        model_ref.untoggle_optimizer(self)

    @contextmanager
    def toggle_model(self, sync_grad: bool = True):
        """
        This function is just a helper for advanced users.

        Considering the current optimizer as A and all other optimizers as B.
        Toggling means all parameters from B exclusive to A will have ``requires_grad`` set to False.


        When performing gradient accumulation, there is no need to perform grad synchronization
        during the accumulation phase.
        Setting `sync_grad` to False will block this synchronization and improve performance.
        """
        with self._trainer.train_loop.block_ddp_sync_behaviour(not sync_grad):
            self._toggle_model()
            yield
            self._untoggle_model()

    def __optimizer_step(self, closure: Optional[Callable] = None, profiler_name: str = None, **kwargs):
        trainer = self._trainer
        optimizer = self._optimizer
        model = trainer.lightning_module

        with trainer.profiler.profile(profiler_name):
            trainer.accelerator.optimizer_step(optimizer, self._optimizer_idx, lambda_closure=closure, **kwargs)

        if self._trainer.train_loop.automatic_optimization:
            trainer.train_loop.on_before_zero_grad(optimizer)
            model.optimizer_zero_grad(trainer.current_epoch, trainer.batch_idx, optimizer, self._optimizer_idx)

    def step(self, *args, closure: Optional[Callable] = None, **kwargs):
        """
        Call this directly from your training_step when doing optimizations manually.
        By using this we can ensure that all the proper scaling when using 16-bit, accelerator etc
        is been done properly for you.

        .. note:: In Manual Optimization, the user is expected to know when to call zero_grad,
            perform accumulated_grad_batches, etc ... Lightning will only take care of precision and accelerators

        Args:

            closure: One could provide its own optimizer_closure. Set to None by default.

            args: Any parameters provided to wrapped optimizer.step()

            kwargs: Any parameters provided to wrapped optimizer.step()

        Example::

            # Scenario for a GAN.

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


            # Scenario for a GAN advanced

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
            profiler_name = "closure_{self._optimizer_idx}"
            closure = do_nothing_closure
        else:
            if not isinstance(closure, types.FunctionType):
                raise MisconfigurationException("When closure is provided, it should be a function")
            profiler_name = f"optimizer_step_and_closure_{self._optimizer_idx}"

        self.__optimizer_step(*args, closure=closure, profiler_name=profiler_name, **kwargs)
        self._total_optimizer_step_calls += 1

    def __repr__(self):
        groups = [{k: round(v, 12) if isinstance(v, float) else v
                   for k, v in sorted(group.items()) if k != "params"} for group in self.param_groups]
        return f"{self.__class__.__name__}(groups={groups})"
