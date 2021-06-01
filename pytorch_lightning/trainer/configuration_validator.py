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
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden


class ConfigValidator:

    def __init__(self, trainer: 'pl.Trainer') -> None:
        self.trainer = trainer

    def verify_loop_configurations(self, model: 'pl.LightningModule') -> None:
        r"""
        Checks that the model is configured correctly before the run is started.

        Args:
            model: The model to check the configuration.

        """
        if self.trainer.state.fn in (TrainerFn.FITTING, TrainerFn.TUNING):
            self.__verify_train_loop_configuration(model)
            self.__verify_eval_loop_configuration(model, 'val')
            self.__verify_manual_optimization_support(model)
        elif self.trainer.state.fn == TrainerFn.VALIDATING:
            self.__verify_eval_loop_configuration(model, 'val')
        elif self.trainer.state.fn == TrainerFn.TESTING:
            self.__verify_eval_loop_configuration(model, 'test')
        elif self.trainer.state.fn == TrainerFn.PREDICTING:
            self.__verify_predict_loop_configuration(model)
        self.__verify_dp_batch_transfer_support(model)

    def __verify_train_loop_configuration(self, model: 'pl.LightningModule') -> None:
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

        trainer = self.trainer

        trainer.overriden_optimizer_step = is_overridden('optimizer_step', model)
        trainer.overriden_optimizer_zero_grad = is_overridden('optimizer_zero_grad', model)
        automatic_optimization = model.automatic_optimization
        going_to_accumulate_grad_batches = trainer.accumulation_scheduler.going_to_accumulate_grad_batches()

        has_overriden_optimization_functions = trainer.overriden_optimizer_step or trainer.overriden_optimizer_zero_grad
        if has_overriden_optimization_functions and going_to_accumulate_grad_batches and automatic_optimization:
            raise MisconfigurationException(
                'When overriding `LightningModule` optimizer_step or optimizer_zero_grad,'
                ' `accumulate_grad_batches` in `Trainer` should be 1.'
                ' It ensures optimizer_step or optimizer_zero_grad are called on every batch.'
            )

    def __verify_eval_loop_configuration(self, model: 'pl.LightningModule', stage: str) -> None:
        loader_name = f'{stage}_dataloader'
        step_name = 'validation_step' if stage == 'val' else 'test_step'

        has_loader = is_overridden(loader_name, model)
        has_step = is_overridden(step_name, model)

        if has_loader and not has_step:
            rank_zero_warn(f'you passed in a {loader_name} but have no {step_name}. Skipping {stage} loop')
        if has_step and not has_loader:
            rank_zero_warn(f'you defined a {step_name} but have no {loader_name}. Skipping {stage} loop')

    def __verify_predict_loop_configuration(self, model: 'pl.LightningModule') -> None:
        has_predict_dataloader = is_overridden('predict_dataloader', model)
        if not has_predict_dataloader:
            raise MisconfigurationException('Dataloader not found for `Trainer.predict`')

    def __verify_dp_batch_transfer_support(self, model: 'pl.LightningModule') -> None:
        """Raise Misconfiguration exception since these hooks are not supported in DP mode"""
        # TODO: Remove this blocker once batch transfer to device is integrated in Lightning for DP mode.
        batch_transfer_hooks = ('on_before_batch_transfer', 'transfer_batch_to_device', 'on_after_batch_transfer')
        for hook in batch_transfer_hooks:
            if self.trainer.accelerator_connector.use_dp and is_overridden(hook, model):
                raise MisconfigurationException(f'Overriding `{hook}` is not supported in DP mode.')

    def __verify_manual_optimization_support(self, model: 'pl.LightningModule') -> None:
        if model.automatic_optimization:
            return
        if self.trainer.gradient_clip_val > 0:
            raise MisconfigurationException(
                f"Automatic gradient clipping is not supported for manual optimization."
                f" Remove `Trainer(gradient_clip_val={self.trainer.gradient_clip_val})`"
                f" or switch to automatic optimization."
            )
        if self.trainer.accumulate_grad_batches != 1:
            raise MisconfigurationException(
                f"Automatic gradient accumulation is not supported for manual optimization."
                f" Remove `Trainer(accumulate_grad_batches={self.trainer.accumulate_grad_batches})`"
                f" or switch to automatic optimization."
            )
