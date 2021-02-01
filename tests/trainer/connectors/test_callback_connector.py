import logging
from unittest.mock import Mock

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from tests.base import BoringModel


def test_attach_model_callbacks():
    """ Test that the callbacks defined in the model and through Trainer get merged correctly. """

    def assert_composition(trainer_callbacks, model_callbacks, expected):
        model = Mock()
        model.configure_callbacks.return_value = model_callbacks
        trainer = Trainer(
            checkpoint_callback=False,
            progress_bar_refresh_rate=0,
            callbacks=trainer_callbacks
        )
        cb_connector = CallbackConnector(trainer)
        cb_connector._attach_model_callbacks(model)
        assert trainer.callbacks == expected

    early_stopping = EarlyStopping()
    progress_bar = ProgressBar()
    lr_monitor = LearningRateMonitor()
    grad_accumulation = GradientAccumulationScheduler({1: 1})

    # no callbacks
    assert_composition(
        trainer_callbacks=[],
        model_callbacks=[],
        expected=[]
    )

    # callbacks of different types
    assert_composition(
        trainer_callbacks=[early_stopping],
        model_callbacks=[progress_bar],
        expected=[early_stopping, progress_bar]
    )

    # same callback type twice, different instance
    assert_composition(
        trainer_callbacks=[progress_bar, EarlyStopping()],
        model_callbacks=[early_stopping],
        expected=[progress_bar, early_stopping]
    )

    # multiple callbacks of the same type in trainer
    assert_composition(
        trainer_callbacks=[LearningRateMonitor(), EarlyStopping(), LearningRateMonitor(), EarlyStopping()],
        model_callbacks=[early_stopping, lr_monitor],
        expected=[early_stopping, lr_monitor]
    )

    # multiple callbacks of the same type, in both trainer and model
    assert_composition(
        trainer_callbacks=[
            LearningRateMonitor(), progress_bar, EarlyStopping(), LearningRateMonitor(), EarlyStopping()
        ],
        model_callbacks=[early_stopping, lr_monitor, grad_accumulation, early_stopping],
        expected=[progress_bar, early_stopping, lr_monitor, grad_accumulation, early_stopping]
    )


def test_attach_model_callbacks_override_info(caplog):
    """ Test that the logs contain the info about overriding callbacks returned by configure_callbacks. """
    model = Mock()
    model.configure_callbacks.return_value = [LearningRateMonitor(), EarlyStopping()]
    trainer = Trainer(
        checkpoint_callback=False,
        callbacks=[EarlyStopping(), LearningRateMonitor(), ProgressBar()]
    )
    cb_connector = CallbackConnector(trainer)
    with caplog.at_level(logging.INFO):
        cb_connector._attach_model_callbacks(model)

    assert "existing callbacks passed to Trainer: EarlyStopping, LearningRateMonitor" in caplog.text


def test_checkpoint_callbacks_are_last(tmpdir):
    """ Test that checkpoint callbacks always get moved to the end of the list, with preserved order. """
    checkpoint1 = ModelCheckpoint(tmpdir)
    checkpoint2 = ModelCheckpoint(tmpdir)
    early_stopping = EarlyStopping()
    lr_monitor = LearningRateMonitor()
    progress_bar = ProgressBar()

    # no model callbacks
    model = Mock()
    model.configure_callbacks.return_value = []
    trainer = Trainer(callbacks=[checkpoint1, progress_bar, lr_monitor, checkpoint2])
    cb_connector = CallbackConnector(trainer)
    cb_connector._attach_model_callbacks(model)
    assert trainer.callbacks == [progress_bar, lr_monitor, checkpoint1, checkpoint2]

    # with model-specific callbacks that substitute ones in Trainer
    model = Mock()
    model.configure_callbacks.return_value = [checkpoint1, early_stopping, checkpoint2]
    trainer = Trainer(callbacks=[progress_bar, lr_monitor, ModelCheckpoint(tmpdir)])
    cb_connector = CallbackConnector(trainer)
    cb_connector._attach_model_callbacks(model)
    assert trainer.callbacks == [progress_bar, lr_monitor, early_stopping, checkpoint1, checkpoint2]
