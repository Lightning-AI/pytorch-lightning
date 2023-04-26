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
import os
from copy import deepcopy
from unittest import mock

import pytest
import torch
from lightning_utilities.test.warning import no_warning_call

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.tuner.lr_finder import _LRFinder
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel
from tests_pytorch.helpers.utils import getattr_recursive


def test_error_with_multiple_optimizers(tmpdir):
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

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    tuner = Tuner(trainer)

    with pytest.raises(MisconfigurationException, match="only works with single optimizer"):
        tuner.lr_find(model)


def test_model_reset_correctly(tmpdir):
    """Check that model weights are correctly reset after _lr_find()"""
    model = BoringModel()
    model.lr = 0.1

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    tuner = Tuner(trainer)
    before_state_dict = deepcopy(model.state_dict())

    tuner.lr_find(model, num_training=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(
            torch.eq(before_state_dict[key], after_state_dict[key])
        ), "Model was not reset correctly after learning rate finder"

    assert not any(f for f in os.listdir(tmpdir) if f.startswith(".lr_find"))


def test_trainer_reset_correctly(tmpdir):
    """Check that all trainer parameters are reset correctly after lr_find()"""
    model = BoringModel()
    model.lr = 0.1

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
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
def test_tuner_lr_find(tmpdir, use_hparams):
    """Test that lr_find updates the learning rate attribute."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.save_hyperparameters()
            self.lr = lr

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr if use_hparams else self.lr)
            return optimizer

    before_lr = 1e-2
    model = CustomBoringModel(lr=before_lr)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2)
    tuner = Tuner(trainer)
    tuner.lr_find(model, update_attr=True)

    if use_hparams:
        after_lr = model.hparams.lr
    else:
        after_lr = model.lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


@pytest.mark.parametrize("use_hparams", [False, True])
def test_trainer_arg_str(tmpdir, use_hparams):
    """Test that setting trainer arg to string works."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, my_fancy_lr):
            super().__init__()
            self.save_hyperparameters()
            self.my_fancy_lr = my_fancy_lr

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.my_fancy_lr if use_hparams else self.my_fancy_lr
            )
            return optimizer

    before_lr = 1e-2
    model = CustomBoringModel(my_fancy_lr=before_lr)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2)
    tuner = Tuner(trainer)
    tuner.lr_find(model, update_attr=True, attr_name="my_fancy_lr")
    if use_hparams:
        after_lr = model.hparams.my_fancy_lr
    else:
        after_lr = model.my_fancy_lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


@pytest.mark.parametrize("opt", ["Adam", "Adagrad"])
def test_call_to_trainer_method(tmpdir, opt):
    """Test that directly calling the trainer method works."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.save_hyperparameters()

        def configure_optimizers(self):
            optimizer = (
                torch.optim.Adagrad(self.parameters(), lr=self.hparams.lr)
                if opt == "Adagrad"
                else torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            )
            return optimizer

    before_lr = 1e-2
    model = CustomBoringModel(1e-2)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2)

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
def test_datamodule_parameter(tmpdir):
    """Test that the datamodule parameter works."""
    seed_everything(1)

    dm = ClassifDataModule()
    model = ClassificationModel(lr=1e-3)

    before_lr = model.lr
    # logger file to get meta
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2)

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=dm)
    after_lr = lr_finder.suggestion()
    model.lr = after_lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


def test_accumulation_and_early_stopping(tmpdir):
    """Test that early stopping of learning rate finder works, and that accumulation also works for this
    feature."""
    seed_everything(1)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.lr = 1e-3

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, accumulate_grad_batches=2)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, early_stop_threshold=None)

    assert lr_finder.suggestion() != 1e-3
    assert len(lr_finder.results["lr"]) == 100
    assert lr_finder._total_batch_idx == 199


def test_suggestion_parameters_work(tmpdir):
    """Test that default skipping does not alter results in basic case."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            return optimizer

    # logger file to get meta
    model = CustomBoringModel(lr=1e-2)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=3)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)
    lr1 = lr_finder.suggestion(skip_begin=10)  # default
    lr2 = lr_finder.suggestion(skip_begin=70)  # way too high, should have an impact

    assert lr1 is not None
    assert lr2 is not None
    assert lr1 != lr2, "Skipping parameter did not influence learning rate"


def test_suggestion_with_non_finite_values(tmpdir):
    """Test that non-finite values does not alter results."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            return optimizer

    model = CustomBoringModel(lr=1e-2)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=3)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)

    before_lr = lr_finder.suggestion()
    lr_finder.results["loss"][-1] = float("nan")
    after_lr = lr_finder.suggestion()

    assert before_lr is not None
    assert after_lr is not None
    assert before_lr == after_lr, "Learning rate was altered because of non-finite loss values"


def test_lr_finder_fails_fast_on_bad_config(tmpdir):
    """Test that tune fails if the model does not have a lr BEFORE running lr find."""
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2)
    tuner = Tuner(trainer)
    with pytest.raises(AttributeError, match="should have one of these fields"):
        tuner.lr_find(BoringModel(), update_attr=True)


def test_lr_candidates_between_min_and_max(tmpdir):
    """Test that learning rate candidates are between min_lr and max_lr."""
    seed_everything(1)

    class TestModel(BoringModel):
        def __init__(self, learning_rate=0.1):
            super().__init__()
            self.save_hyperparameters()

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir)

    lr_min = 1e-8
    lr_max = 1.0
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, max_lr=lr_min, min_lr=lr_max, num_training=3)
    lr_candidates = lr_finder.results["lr"]
    assert all(lr_min <= lr <= lr_max for lr in lr_candidates)


def test_lr_finder_ends_before_num_training(tmpdir):
    """Tests learning rate finder ends before `num_training` steps."""

    class TestModel(BoringModel):
        def __init__(self, learning_rate=0.1):
            super().__init__()
            self.save_hyperparameters()

        def on_before_optimizer_step(self, optimizer):
            assert self.global_step < num_training

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir)
    tuner = Tuner(trainer)
    num_training = 3
    tuner.lr_find(model=model, num_training=num_training)


def test_multiple_lr_find_calls_gives_same_results(tmpdir):
    """Tests that lr_finder gives same results if called multiple times."""
    seed_everything(1)
    model = BoringModel()
    model.lr = 0.1

    trainer = Trainer(
        default_root_dir=tmpdir,
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
        for k in all_res[0].keys()
    )


@pytest.mark.parametrize(
    "skip_begin,skip_end,losses,expected_error",
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


def test_lr_attribute_when_suggestion_invalid(tmpdir):
    """Tests learning rate finder ends before `num_training` steps."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.learning_rate = 0.123

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model=model, update_attr=True, num_training=1)  # force insufficient data points
    assert lr_finder.suggestion() is None
    assert model.learning_rate == 0.123  # must remain unchanged because suggestion is not possible


def test_lr_finder_callback_restarting(tmpdir):
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

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                self.lr_find(trainer, pl_module)

    model = MyBoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
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
def test_lr_finder_with_ddp(tmpdir):
    seed_everything(7)

    init_lr = 1e-4
    dm = ClassifDataModule()
    model = ClassificationModel(lr=init_lr)

    trainer = Trainer(
        default_root_dir=tmpdir,
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
