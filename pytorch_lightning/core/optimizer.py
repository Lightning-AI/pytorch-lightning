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
from typing import Callable, Optional
from weakref import proxy

from torch.optim.optimizer import Optimizer

from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities import DeviceType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


def is_lightning_optimizer(optimizer):
    return isinstance(optimizer, LightningOptimizer)


def do_nothing_closure():
    return


class LightningOptimizer:
    """
    This class is used to wrap the user optimizers and handle properly
    the backward and optimizer_step logic across accelerators, AMP, accumulate_grad_batches
    """
    def __init__(self,
                 optimizer: Optimizer):

        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k != 'step'}

        # For Horovod
        if hasattr(optimizer, "skip_synchronize"):
            self.__class__ = type("Lightning" + optimizer.__class__.__name__,
                                  (self.__class__, optimizer.__class__.__bases__[0]), {})
            self.skip_synchronize = optimizer.skip_synchronize
            self.synchronize = optimizer.synchronize
        else:
            self.__class__ = type("Lightning" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})

        self._optimizer = optimizer
        self._trainer = None
        self._optimizer_idx = None

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

    def __optimizer_step(self, zero_grad: bool = False, closure: Optional[Callable] = None, profiler_name: str = None, **kwargs):
        trainer = self._trainer
        optimizer = self._optimizer
        model = trainer.get_model()

        with trainer.profiler.profile(profiler_name):
            trainer.accelerator_backend.optimizer_step(optimizer, self._optimizer_idx, lambda_closure=closure, **kwargs)

        if self._trainer.train_loop.automatic_optimization or zero_grad:

            trainer.train_loop.on_before_zero_grad(optimizer)

            model.optimizer_zero_grad(
                trainer.current_epoch,
                trainer.batch_idx,
                optimizer,
                self._optimizer_idx
            )

    def _check_make_optimizer_step(self, make_optimizer_step: Optional[bool]) -> bool:
        if make_optimizer_step is not None and self._trainer.overriden_optimizer_zero_grad:
            raise MisconfigurationException(
                "When overriding LightningModule `optimizer_zero_grad`, make_optimizer_step is not allowed.")

        if self._trainer.train_loop.automatic_optimization:
            if self._trainer.overriden_optimizer_step and self._trainer.overriden_optimizer_zero_grad:
                return True

        if make_optimizer_step is None:
            make_optimizer_step = True

        return make_optimizer_step

    def toggle_optimizer(self):
        model = self.trainer.get_model()
        model.toggle_optimizer(self, self._optimizer_idx)

    def step(
        self,
        *args,
        closure: Optional[Callable] = None,
        make_optimizer_step: Optional[bool] = None,
        zero_grad : bool = False,
        toggle_optimizer : bool = False,
        **kwargs
    ):
        """
        Call this directly from your training_step when doing optimizations manually.
        By using this we can ensure that all the proper scaling when using 16-bit etc has been done for you

        .. tip:: In manual mode we still automatically accumulate grad over batches if
           Trainer(accumulate_grad_batches=x) is set.

        Args:

            closure: One could provide its own optimizer_closure. Set to None by default.

            make_optimizer_step: Whether to force an optimizer step. When nothing is provided,
                we will use perform an optimizer step.

            zero_grad: Whether to perform zero_grad after performing an `optimizer.step()`

            toggle_optimizer: Wheter to toggle the model based the optimizer.

            args: Any parameters provided to wrapped optimizer.step()

            kwargs: Any parameters provided to wrapped optimizer.step()

        Example::

            def training_step(...):
                (opt_a, opt_b) = self.optimizers()
                loss_a = ...

                # automatically applies scaling, etc...

                self.manual_backward(loss_a)
                opt_a.step()
                opt_a.zero_grad()

        Example::

            # Scenario for a gan.

            def training_step(self, batch, batch_idx, optimizer_idx):

                opt_gen, opt_dis = self.optimizers()

                ...
                loss_total = loss_gen + loss_dis

                self.manual_backward(loss_total)

                opt_gen.step()
                opt_gen.zero_grad()

                # perform accumulated gradients for 4 batches
                if batch_idx % 4 == 0 :
                    opt_dis.step()
                    opt_dis.zero_grad()

        Even better when using multi-gpus and complex accumulated gradient strategy

        Example::

            # Scenario for a gan.

                opt_gen, opt_dis = self.optimizers()

                def closure():
                    ...
                    if batch_idx % 4 == 0:
                        loss = loss_gen + loss_dis
                    else:
                        loss = loss_gen
                    self.manual_backward(loss)

                # perform 2 batches gradient accumulation
                # toggle optimizer, so discriminator doesn't receive gradients
                # except on discriminator steps.
                opt_gen.step(
                    closure=closure,
                    make_optimizer_step=batch_idx % 2 == 0,
                    zero_grad=True,
                    toggle_optimizer=batch_idx % 4 != 0)

                if batch_idx % 4 == 0:
                    opt_gen.step(zero_grad=True)

        """
        profiler_name = f"optimizer_step_and_closure_{self._optimizer_idx}"

        if closure is None:
            closure = do_nothing_closure
        else:
            if not isinstance(closure, types.FunctionType):
                raise MisconfigurationException("When closure is provided, it should be a function")

        make_optimizer_step = self._check_make_optimizer_step(make_optimizer_step)

        if toggle_optimizer:
            self.toggle_optimizer()

        if make_optimizer_step:
            self.__optimizer_step(*args, zero_grad=zero_grad, closure=closure, profiler_name=profiler_name, **kwargs)
        else:
            # make sure to call optimizer_closure when accumulating
            with self._trainer.profiler.profile(f"closure_{self._optimizer_idx}"):
                with self._trainer.train_loop.block_ddp_sync_behaviour():
                    closure()

    def __repr__(self):
        groups = [
            {
                k: round(v, 12) if isinstance(v, float) else v
                for k, v in sorted(group.items())
                if k != "params"
            }
            for group in self.param_groups
        ]
        return f"{self.__class__.__name__}(groups={groups})"
