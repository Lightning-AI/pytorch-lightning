from unittest.mock import Mock

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBar, EarlyStopping, LearningRateMonitor, \
    GradientAccumulationScheduler
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector


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
        cb_connector.attach_model_callbacks(model)
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
