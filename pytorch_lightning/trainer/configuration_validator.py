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
from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_utils import is_overridden


class ConfigValidator(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def verify_loop_configurations(self, model: LightningModule):
        r"""
        Checks that the model is configured correctly before training or testing is started.

        Args:
            model: The model to check the configuration.

        """
        if not self.trainer.testing:
            self.__verify_train_loop_configuration(model)
            self.__verify_eval_loop_configuration(model, 'validation')
        else:
            # check test loop configuration
            self.__verify_eval_loop_configuration(model, 'test')

    def __verify_train_loop_configuration(self, model):
        # -----------------------------------
        # verify model has a training step
        # -----------------------------------
        has_training_step = is_overridden('training_step', model)
        if not has_training_step:
            raise MisconfigurationException(
                'No `training_step()` method defined. Lightning `Trainer` expects as minimum a'
                ' `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.'
            )

        # -----------------------------------
        # verify model has a train dataloader
        # -----------------------------------
        has_train_dataloader = is_overridden('train_dataloader', model)
        if not has_train_dataloader:
            raise MisconfigurationException(
                'No `train_dataloader()` method defined. Lightning `Trainer` expects as minimum a'
                ' `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.'
            )

        # -----------------------------------
        # verify model has optimizer
        # -----------------------------------
        has_optimizers = is_overridden('configure_optimizers', model)
        if not has_optimizers:
            raise MisconfigurationException(
                'No `configure_optimizers()` method defined. Lightning `Trainer` expects as minimum a'
                ' `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.'
            )

        has_optimizer_step = is_overridden('optimizer_step', model)
        enable_pl_optimizer = self.trainer._enable_pl_optimizer
        automatic_optimization = self.trainer.train_loop.automatic_optimization
        if has_optimizer_step and not enable_pl_optimizer and automatic_optimization:
            log.warn(
                "When overriding `LightningModule` optimizer_step with "
                "Trainer(..., enable_pl_optimizer=False, automatic_optimization=True, ...), "
                "we won t be calling `.zero_grad` we can't assume when you call your optimizer.step(). "
                "For Lightning to take care of it, please use Trainer(enable_pl_optimizer=True)."
            )

        going_to_accumulate_grad_batches = self.trainer.accumulation_scheduler.going_to_accumulate_grad_batches()

        if has_optimizer_step and going_to_accumulate_grad_batches and automatic_optimization:
            raise MisconfigurationException(
                'When overriding `LightningModule` optimizer_step with Trainer(..., automatic_optimization=True, ...), '
                'accumulate_grad_batches should to be 1. It ensures this `optimizer_step` is called on every batch'
            )

    def __verify_eval_loop_configuration(self, model, eval_loop_name):
        step_name = f'{eval_loop_name}_step'

        # map the dataloader name
        loader_name = f'{eval_loop_name}_dataloader'
        if eval_loop_name == 'validation':
            loader_name = 'val_dataloader'

        has_loader = is_overridden(loader_name, model)
        has_step = is_overridden(step_name, model)

        if has_loader and not has_step:
            rank_zero_warn(
                f'you passed in a {loader_name} but have no {step_name}. Skipping {eval_loop_name} loop'
            )
        if has_step and not has_loader:
            rank_zero_warn(
                f'you defined a {step_name} but have no {loader_name}. Skipping {eval_loop_name} loop'
            )
