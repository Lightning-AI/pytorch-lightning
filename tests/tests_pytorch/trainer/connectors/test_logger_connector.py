from unittest.mock import MagicMock, patch

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer.connectors.logger_connector import _LoggerConnector


@patch("lightning.pytorch.trainer.connectors.logger_connector.logger_connector.convert_tensors_to_scalars")
def test_uses_provided_step(mock_convert):
    """Test that the LoggerConnector uses explicitly provided step to log metrics."""

    trainer = MagicMock(spec=Trainer)
    trainer.loggers = [logger := MagicMock(spec=Logger)]
    connector = _LoggerConnector(trainer)
    mock_convert.return_value.pop.return_value = step = 42

    connector.log_metrics((metrics := {"some_metric": 123}), step=step)

    assert connector._logged_metrics == metrics
    mock_convert.assert_called_once_with(metrics)
    logger.log_metrics.assert_called_once_with(metrics=mock_convert.return_value, step=step)
    logger.save.assert_called_once_with()


@patch("lightning.pytorch.trainer.connectors.logger_connector.logger_connector.convert_tensors_to_scalars")
def test_uses_step_metric(mock_convert):
    """Test that the LoggerConnector uses explicitly provided step metric to log metrics."""

    trainer = MagicMock(spec=Trainer)
    trainer.loggers = [logger := MagicMock(spec=Logger)]
    connector = _LoggerConnector(trainer)
    mock_convert.return_value.pop.return_value = step = 42.0

    metrics = {"some_metric": 123}
    connector.log_metrics(logged_metrics := {**metrics, "step": step})

    assert connector._logged_metrics == logged_metrics
    mock_convert.assert_called_once_with(logged_metrics)
    logger.log_metrics.assert_called_once_with(metrics=mock_convert.return_value, step=int(step))
    logger.save.assert_called_once_with()


@patch("lightning.pytorch.trainer.connectors.logger_connector.logger_connector.convert_tensors_to_scalars")
def test_uses_batches_that_stepped(mock_convert):
    """Test that the LoggerConnector uses implicitly provided batches_that_stepped to log metrics."""

    trainer = MagicMock(spec=Trainer)
    trainer.fit_loop = MagicMock()
    trainer.loggers = [logger := MagicMock(spec=Logger)]
    connector = _LoggerConnector(trainer)
    mock_convert.return_value.pop.return_value = None

    connector.log_metrics(metrics := {"some_metric": 123})

    assert connector._logged_metrics == metrics
    mock_convert.assert_called_once_with(metrics)
    logger.log_metrics.assert_called_once_with(
        metrics=mock_convert.return_value, step=trainer.fit_loop.epoch_loop._batches_that_stepped
    )
    logger.save.assert_called_once_with()
    mock_convert.return_value.setdefault.assert_called_once_with("epoch", trainer.current_epoch)
