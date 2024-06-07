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
import logging
import math
import os
from copy import deepcopy
from typing import Any
from unittest import mock

import pytest
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.tuner.lr_finder import _LRFinder
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.test.warning import no_warning_call

from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel
from tests_pytorch.helpers.utils import getattr_recursive


def test_error_with_multiple_optimizers(tmp_path):
    """Check that error is thrown when more than 1 optimizer is passed."""

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.save_hyperparameters()
            self.automatic_optimization = False

        def configure_optimizers(self):
            optimizer1 = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
            optimizer2 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            return [optimizer1, optimizer2]

    model = CustomBoringModel(lr=1e-2)

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    with pytest.raises(MisconfigurationException, match="only works with single optimizer"):
        tuner.lr_find(model)


def test_model_reset_correctly(tmp_path):
    """Check that model weights are correctly reset after _lr_find()"""
    model = BoringModel()
    model.lr = 0.1

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)
    before_state_dict = deepcopy(model.state_dict())

    tuner.lr_find(model, num_training=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict:
        assert torch.all(
            torch.eq(before_state_dict[key], after_state_dict[key])
        ), "Model was not reset correctly after learning rate finder"

    assert not any(f for f in os.listdir(tmp_path) if f.startswith(".lr_find"))


def test_trainer_reset_correctly(tmp_path):
    """Check that all trainer parameters are reset correctly after lr_find()"""
    model = BoringModel()
    model.lr = 0.1

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    changed_attributes = [
        "accumulate_grad_batches",
        "callbacks",
        "checkpoint_callback",
        "current_epoch",
        "loggers",
        "global_step",
        "max_steps",
        "fit_loop.max_steps",
        "strategy.setup_optimizers",
        "should_stop",
    ]
    expected = {ca: getattr_recursive(trainer, ca) for ca in changed_attributes}

    with no_warning_call(UserWarning, match="Please add the following callbacks"):
        tuner.lr_find(model, num_training=5)

    actual = {ca: getattr_recursive(trainer, ca) for ca in changed_attributes}
    assert actual == expected
    assert model.trainer == trainer


@pytest.mark.parametrize("use_hparams", [False, True])
def test_tuner_lr_find(tmp_path, use_hparams):
    """Test that lr_find updates the learning rate attribute."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.save_hyperparameters()
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr if use_hparams else self.lr)

    before_lr = 1e-2
    model = CustomBoringModel(lr=before_lr)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)
    tuner = Tuner(trainer)
    tuner.lr_find(model, update_attr=True)

    after_lr = model.hparams.lr if use_hparams else model.lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


@pytest.mark.parametrize("use_hparams", [False, True])
def test_trainer_arg_str(tmp_path, use_hparams):
    """Test that setting trainer arg to string works."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, my_fancy_lr):
            super().__init__()
            self.save_hyperparameters()
            self.my_fancy_lr = my_fancy_lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.hparams.my_fancy_lr if use_hparams else self.my_fancy_lr)

    before_lr = 1e-2
    model = CustomBoringModel(my_fancy_lr=before_lr)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)
    tuner = Tuner(trainer)
    tuner.lr_find(model, update_attr=True, attr_name="my_fancy_lr")
    after_lr = model.hparams.my_fancy_lr if use_hparams else model.my_fancy_lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


@pytest.mark.parametrize("opt", ["Adam", "Adagrad"])
def test_call_to_trainer_method(tmp_path, opt):
    """Test that directly calling the trainer method works."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.save_hyperparameters()

        def configure_optimizers(self):
            return (
                torch.optim.Adagrad(self.parameters(), lr=self.hparams.lr)
                if opt == "Adagrad"
                else torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            )

    before_lr = 1e-2
    model = CustomBoringModel(1e-2)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, mode="linear")
    after_lr = lr_finder.suggestion()
    assert after_lr is not None
    model.hparams.lr = after_lr
    tuner.lr_find(model, update_attr=True)

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


@RunIf(sklearn=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_datamodule_parameter(tmp_path):
    """Test that the datamodule parameter works."""
    seed_everything(1)

    dm = ClassifDataModule()
    model = ClassificationModel(lr=1e-3)

    before_lr = model.lr
    # logger file to get meta
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=dm)
    after_lr = lr_finder.suggestion()
    model.lr = after_lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


def test_accumulation_and_early_stopping(tmp_path):
    """Test that early stopping of learning rate finder works, and that accumulation also works for this feature."""
    seed_everything(1)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.lr = 1e-3

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, accumulate_grad_batches=2)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, early_stop_threshold=None)

    assert lr_finder.suggestion() != 1e-3
    assert len(lr_finder.results["lr"]) == len(lr_finder.results["loss"]) == 100
    assert lr_finder._total_batch_idx == 199


def test_suggestion_parameters_work(tmp_path):
    """Test that default skipping does not alter results in basic case."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.lr)

    # logger file to get meta
    model = CustomBoringModel(lr=1e-2)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=3)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)
    lr1 = lr_finder.suggestion(skip_begin=10)  # default
    lr2 = lr_finder.suggestion(skip_begin=70)  # way too high, should have an impact

    assert lr1 is not None
    assert lr2 is not None
    assert lr1 != lr2, "Skipping parameter did not influence learning rate"


def test_suggestion_with_non_finite_values(tmp_path):
    """Test that non-finite values does not alter results."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.lr)

    model = CustomBoringModel(lr=1e-2)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=3)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)

    before_lr = lr_finder.suggestion()
    lr_finder.results["loss"][-1] = float("nan")
    after_lr = lr_finder.suggestion()

    assert before_lr is not None
    assert after_lr is not None
    assert before_lr == after_lr, "Learning rate was altered because of non-finite loss values"


def test_lr_finder_fails_fast_on_bad_config(tmp_path):
    """Test that tune fails if the model does not have a lr BEFORE running lr find."""
    trainer = Trainer(default_root_dir=tmp_path, max_steps=2)
    tuner = Tuner(trainer)
    with pytest.raises(AttributeError, match="should have one of these fields"):
        tuner.lr_find(BoringModel(), update_attr=True)


def test_lr_candidates_between_min_and_max(tmp_path):
    """Test that learning rate candidates are between min_lr and max_lr."""
    seed_everything(1)

    class TestModel(BoringModel):
        def __init__(self, learning_rate=0.1):
            super().__init__()
            self.save_hyperparameters()

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path)

    lr_min = 1e-8
    lr_max = 1.0
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, max_lr=lr_min, min_lr=lr_max, num_training=3)
    lr_candidates = lr_finder.results["lr"]
    assert all(lr_min <= lr <= lr_max for lr in lr_candidates)


def test_lr_finder_ends_before_num_training(tmp_path):
    """Tests learning rate finder ends before `num_training` steps."""

    class TestModel(BoringModel):
        def __init__(self, learning_rate=0.1):
            super().__init__()
            self.save_hyperparameters()

        def on_before_optimizer_step(self, optimizer):
            assert self.global_step < num_training

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path)
    tuner = Tuner(trainer)
    num_training = 3
    tuner.lr_find(model=model, num_training=num_training)


def test_multiple_lr_find_calls_gives_same_results(tmp_path):
    """Tests that lr_finder gives same results if called multiple times."""
    seed_everything(1)
    model = BoringModel()
    model.lr = 0.1

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        limit_train_batches=10,
        limit_val_batches=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    tuner = Tuner(trainer)
    all_res = [tuner.lr_find(model).results for _ in range(3)]

    assert all(
        all_res[0][k] == curr_lr_finder[k] and len(curr_lr_finder[k]) > 10
        for curr_lr_finder in all_res[1:]
        for k in all_res[0]
    )


@pytest.mark.parametrize(
    ("skip_begin", "skip_end", "losses", "expected_error"),
    [
        (0, 0, [], True),
        (10, 1, [], True),
        (0, 2, [0, 1, 2], True),
        (0, 1, [0, 1, 2], False),
        (1, 1, [0, 1, 2], True),
        (1, 1, [0, 1, 2, 3], False),
        (0, 1, [float("nan"), float("nan"), 0, float("inf"), 1, 2, 3, float("inf"), 2, float("nan"), 1], False),
        (4, 1, [float("nan"), float("nan"), 0, float("inf"), 1, 2, 3, float("inf"), 2, float("nan"), 1], False),
    ],
)
def test_suggestion_not_enough_finite_points(losses, skip_begin, skip_end, expected_error, caplog):
    """Tests the error handling when not enough finite points are available to make a suggestion."""
    caplog.clear()
    lr_finder = _LRFinder(
        mode="exponential",
        lr_min=1e-8,
        lr_max=1,
        num_training=100,
    )
    lrs = list(torch.arange(len(losses)))
    lr_finder.results = {
        "lr": lrs,
        "loss": losses,
    }
    with caplog.at_level(logging.ERROR, logger="root.tuner.lr_finder"):
        lr = lr_finder.suggestion(skip_begin=skip_begin, skip_end=skip_end)

        if expected_error:
            assert lr is None
            assert "Failed to compute suggestion for learning rate" in caplog.text
        else:
            assert lr is not None


def test_lr_attribute_when_suggestion_invalid(tmp_path):
    """Tests learning rate finder ends before `num_training` steps."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.learning_rate = 0.123

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model=model, update_attr=True, num_training=1)  # force insufficient data points
    assert lr_finder.suggestion() is None
    assert model.learning_rate == 0.123  # must remain unchanged because suggestion is not possible


def test_lr_finder_callback_restarting(tmp_path):
    """Test that `LearningRateFinder` does not set restarting=True when loading checkpoint."""
    num_lr_steps = 100

    class MyBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.learning_rate = 0.123

        def on_train_batch_start(self, batch, batch_idx):
            if getattr(self, "_expected_max_steps", None) is not None:
                assert self.trainer.fit_loop.max_steps == self._expected_max_steps

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    class CustomLearningRateFinder(LearningRateFinder):
        milestones = (1,)

        def lr_find(self, trainer, pl_module) -> None:
            pl_module._expected_max_steps = trainer.global_step + self._num_training_steps
            super().lr_find(trainer, pl_module)
            pl_module._expected_max_steps = None
            assert not trainer.fit_loop.restarting
            assert not trainer.fit_loop.epoch_loop.restarting

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                self.lr_find(trainer, pl_module)

    model = MyBoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        callbacks=[
            CustomLearningRateFinder(early_stop_threshold=None, update_attr=True, num_training_steps=num_lr_steps)
        ],
        limit_train_batches=10,
        limit_val_batches=0,
        limit_test_batches=0,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )

    trainer.fit(model)


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@RunIf(standalone=True)
def test_lr_finder_with_ddp(tmp_path):
    seed_everything(7)

    init_lr = 1e-4
    dm = ClassifDataModule()
    model = ClassificationModel(lr=init_lr)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        strategy="ddp",
        devices=2,
        accelerator="cpu",
    )

    tuner = Tuner(trainer)
    tuner.lr_find(model, datamodule=dm, update_attr=True, num_training=20)
    lr = trainer.lightning_module.lr
    lr = trainer.strategy.broadcast(lr)
    assert trainer.lightning_module.lr == lr
    assert lr != init_lr


def test_lr_finder_callback_val_batches(tmp_path):
    """Test that `LearningRateFinder` does not limit the number of val batches during training."""

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.lr)

    num_lr_tuner_training_steps = 5
    model = CustomBoringModel(0.1)
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=False,
        callbacks=[LearningRateFinder(num_training_steps=num_lr_tuner_training_steps)],
    )
    trainer.fit(model)

    assert trainer.num_val_batches[0] == len(trainer.val_dataloaders)
    assert trainer.num_val_batches[0] != num_lr_tuner_training_steps


def test_lr_finder_training_step_none_output(tmp_path):
    # add some nans into the skipped steps (first 10) but also into the steps used to compute the lr
    none_steps = [5, 12, 17]

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.lr = 0.123

        def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            if self.trainer.global_step in none_steps:
                return None

            return super().training_step(batch, batch_idx)

    seed_everything(1)
    model = CustomBoringModel()

    trainer = Trainer(default_root_dir=tmp_path)

    tuner = Tuner(trainer)
    # restrict number of steps for faster test execution
    # and disable early stopping to easily check expected number of lrs and losses
    lr_finder = tuner.lr_find(model=model, update_attr=True, num_training=20, early_stop_threshold=None)
    assert len(lr_finder.results["lr"]) == len(lr_finder.results["loss"]) == 20
    assert torch.isnan(torch.tensor(lr_finder.results["loss"])[none_steps]).all()

    suggested_lr = lr_finder.suggestion()
    assert math.isfinite(suggested_lr)
    assert math.isclose(model.lr, suggested_lr)
