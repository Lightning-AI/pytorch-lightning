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
from typing import Any, Callable, Optional
from weakref import proxy

from torch.optim.optimizer import Optimizer

from pytorch_lightning.utilities import AMPType, TPU_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if TPU_AVAILABLE:
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
                 optimizer: Optimizer,
                 accumulate_grad_batches: Optional[int] = None):

        assert accumulate_grad_batches is None or isinstance(accumulate_grad_batches, int)
        if isinstance(accumulate_grad_batches, int) and accumulate_grad_batches < 1:
            raise MisconfigurationException(
                f"accumulate_grad_batches parameters {accumulate_grad_batches} should be >= 1"
            )

        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k != 'step'}

        # For Horovod
        if hasattr(optimizer, "skip_synchronize"):
            self.__class__ = type("Lightning" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__.__bases__[0]), {})
            self.skip_synchronize = optimizer.skip_synchronize
            self.synchronize = optimizer.synchronize
        else:
            self.__class__ = type("Lightning" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})

        self._optimizer = optimizer
        self._trainer = None
        self._accumulate_grad_batches = accumulate_grad_batches
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

    @property
    def accumulate_grad_batches(self):
        return self._accumulate_grad_batches

    @accumulate_grad_batches.setter
    def accumulate_grad_batches(self, accumulate_grad_batches):
        self._accumulate_grad_batches = accumulate_grad_batches

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

    def _accumulated_batches_reached(self):
        if self.accumulate_grad_batches is None:
            return self._trainer.train_loop._accumulated_batches_reached()
        return (self._trainer.batch_idx + 1) % self.accumulate_grad_batches == 0

    @property
    def _should_accumulate(self):
        # checks if backward or backward + optimizer step (via closure)
        accumulation_done = self._accumulated_batches_reached()
        is_final_batch = self._trainer.train_loop._num_training_batches_reached()
        return not (accumulation_done or is_final_batch)

    def __optimizer_step(self, *args, closure: Optional[Callable] = None, profiler_name: str = None, **kwargs):
        trainer = self._trainer
        optimizer = self._optimizer
        model = trainer.get_model()

        if trainer.on_tpu:
            with trainer.profiler.profile(profiler_name):
                xm.optimizer_step(optimizer, optimizer_args={'closure': closure, **kwargs})

        elif trainer.amp_backend is not None:
            trainer.precision_connector.backend.optimizer_step(trainer, optimizer, closure)

        else:
            with trainer.profiler.profile(profiler_name):
                optimizer.step(closure=closure, *args, **kwargs)

        accelerator_backend = trainer.accelerator_backend
        if accelerator_backend is not None and accelerator_backend.rpc_enabled:
            if accelerator_backend.ddp_plugin.is_main_rpc_process:
                # Initialize optimizer step on main process
                accelerator_backend.ddp_plugin.worker_optimizer_step(
                    model=model,
                    opt_idx=self._optimizer_idx,
                    *args,
                    **kwargs
                )

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
            make_optimizer_step = not self._should_accumulate

        return make_optimizer_step

    def step(self, *args, closure: Optional[Callable] = None, make_optimizer_step: Optional[bool] = None, **kwargs):
        """
        Call this directly from your training_step when doing optimizations manually.
        By using this we can ensure that all the proper scaling when using 16-bit etc has been done for you

        .. tip:: In manual mode we still automatically accumulate grad over batches if
           Trainer(accumulate_grad_batches=x) is set.

        Args:

            closure: One could provide its own optimizer_closure. Set to None by default.

            make_optimizer_step: Whether to force an optimizer step. When nothing is provided,
                we will use `accumulate_grad_batches` for accumulation frequency by default.
                However, one coud provide True and False based on its own scheduling.
                Refer to example 2 and 3

            args: Any parameters provided to wrapped optimizer.step()

            kwargs: Any parameters provided to wrapped optimizer.step()

        Example::

            def training_step(...):
                (opt_a, opt_b) = self.optimizers()
                loss_a = ...
                # automatically applies scaling, etc...
                self.manual_backward(loss_a, opt_a)
                opt_a.step()

        Example::

            def training_step(self, batch, batch_idx):
                # using Boring Model
                opt = self.optimizers() # only 1 optimizer

                def compute_loss():
                    x = batch[0]
                    x = F.dropout(x, 0.1)
                    predictions = self(x)
                    predictions = F.dropout(predictions, 0.1)
                    loss = self.loss(None, predictions)
                    return loss

                def closure():
                    # emulate MC dropout training
                    num_backward = 1
                    losses = []
                    for backward_idx in range(num_backward + 1):
                        loss = compute_loss()
                        losses.append(loss)
                        retain_graph = num_backward!= backward_idx
                        self.manual_backward(loss, opt, retain_graph=retain_graph)
                    loss_mean = torch.stack(losses).mean()
                    loss_std = torch.stack(losses).std()
                    self.log("train_loss_mean", loss_mean, on_step=True, prog_bar=True, on_epoch=True)
                    self.log("train_loss_std", loss_std, on_step=True, prog_bar=True, on_epoch=True)

                opt.step(loss, closure=closure)

        Example::

            # Scenario for a gan.

            def training_step(self, batch, batch_idx, optimizer_idx):

                # emulate gans training
                opt_gen, opt_dis = self.optimizers()

                # Note: Be careful, don't log on the same key in self.log in both closure
                # as they will be aggregated together on epoch_end

                def gen_closure():
                    ... forward and compute loss for generator
                    loss_gen = ...
                    self.log("loss_gen", loss_gen, on_step=True, on_epoch=True)
                    self.manual_backward(loss_gen, opt_gen)

                def dis_closure():
                    ... forward and compute loss for discriminator
                    loss_dis = ...
                    self.log("loss_dis", loss_dis, on_step=True, on_epoch=True)
                    self.manual_backward(loss_dis, opt_dis)

                # this will accumulate gradients for 2 batches and then call opt_gen.step()
                opt_gen.step(closure=gen_closure, make_optimizer_step=batch_idx % 2 == 0)

                # update discriminator every 4 batches
                # therefore, no gradient accumulation for discriminator
                if batch_idx % 4 == 0 :
                    # Note: Set make_optimizer_step to True or it will use by default
                    # Trainer(accumulate_grad_batches=x)
                    opt_dis.step(closure=optimizer_closure, make_optimizer_step=True)
        """
        profiler_name = f"optimizer_step_and_closure_{self._optimizer_idx}"

        if closure is None:
            closure = do_nothing_closure
            profile_name = f"optimizer_step_{self._optimizer_idx}"
        else:
            if not isinstance(closure, types.FunctionType):
                raise MisconfigurationException("When closure is provided, it should be a function")

        make_optimizer_step = self._check_make_optimizer_step(make_optimizer_step)

        if make_optimizer_step:
            self.__optimizer_step(*args, closure=closure, profiler_name=profiler_name, **kwargs)
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
