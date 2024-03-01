# Copyright The Lightning AI team.
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
"""Test logging in the training loop."""

import inspect
from unittest import mock
from unittest.mock import ANY

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.connectors.logger_connector.fx_validator import _FxValidator
from lightning.pytorch.trainer.connectors.logger_connector.result import _ResultCollection
from lightning.pytorch.trainer.states import RunningStage, TrainerFn


def test_default_level_for_hooks_that_support_logging():
    def _make_assertion(model, hooks, result_mock, on_step, on_epoch, extra_kwargs):
        for hook in hooks:
            model._current_fx_name = hook
            model.log(hook, 1)
            result_mock.assert_called_with(
                hook, hook, torch.tensor(1), on_step=on_step, on_epoch=on_epoch, **extra_kwargs
            )

    trainer = Trainer()
    model = BoringModel()
    model.trainer = trainer
    extra_kwargs = {
        k: ANY
        for k in inspect.signature(_ResultCollection.log).parameters
        if k not in ["self", "fx", "name", "value", "on_step", "on_epoch"]
    }
    all_logging_hooks = {k for k in _FxValidator.functions if _FxValidator.functions[k]}

    with mock.patch(
        "lightning.pytorch.trainer.connectors.logger_connector.result._ResultCollection.log", return_value=None
    ) as result_mock:
        trainer.state.stage = RunningStage.TRAINING
        hooks = [
            "on_before_backward",
            "backward",
            "on_after_backward",
            "on_before_optimizer_step",
            "optimizer_step",
            "configure_gradient_clipping",
            "clip_gradients",
            "on_before_zero_grad",
            "optimizer_zero_grad",
            "training_step",
            "on_train_batch_start",
            "on_train_batch_end",
        ]
        all_logging_hooks = all_logging_hooks - set(hooks)
        _make_assertion(model, hooks, result_mock, on_step=True, on_epoch=False, extra_kwargs=extra_kwargs)

        hooks = [
            "on_train_start",
            "on_train_epoch_start",
            "on_train_epoch_end",
        ]
        all_logging_hooks = all_logging_hooks - set(hooks)
        _make_assertion(model, hooks, result_mock, on_step=False, on_epoch=True, extra_kwargs=extra_kwargs)

        trainer.state.stage = RunningStage.VALIDATING
        trainer.state.fn = TrainerFn.VALIDATING
        hooks = [
            "on_validation_start",
            "on_validation_epoch_start",
            "on_validation_epoch_end",
            "on_validation_batch_start",
            "on_validation_batch_end",
            "validation_step",
        ]
        all_logging_hooks = all_logging_hooks - set(hooks)
        _make_assertion(model, hooks, result_mock, on_step=False, on_epoch=True, extra_kwargs=extra_kwargs)

        trainer.state.stage = RunningStage.TESTING
        trainer.state.fn = TrainerFn.TESTING
        hooks = [
            "on_test_start",
            "on_test_epoch_start",
            "on_test_epoch_end",
            "on_test_batch_start",
            "on_test_batch_end",
            "test_step",
        ]
        all_logging_hooks = all_logging_hooks - set(hooks)
        _make_assertion(model, hooks, result_mock, on_step=False, on_epoch=True, extra_kwargs=extra_kwargs)

    # just to ensure we checked all possible logging hooks here
    assert len(all_logging_hooks) == 0
