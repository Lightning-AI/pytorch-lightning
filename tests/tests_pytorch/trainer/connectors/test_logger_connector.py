from unittest.mock import MagicMock, patch

import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer.connectors.logger_connector import _ListMap, _LoggerConnector


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


@pytest.mark.parametrize(
    "args",
    [
        (1, 2),
        [1, 2],
        {1, 2},
        range(2),
    ],
)
def test_listmap_init(args):
    """Test initialization with different iterable types."""
    lm = _ListMap(args)
    assert len(lm) == len(args)
    assert isinstance(lm, list)
    if isinstance(args, _ListMap):
        assert lm == args


def test_listmap_init_wrong_type():
    with pytest.raises(TypeError):
        _ListMap({1: 2, 3: 4})


def test_listmap_append():
    """Test appending loggers to the collection."""
    lm = _ListMap()
    lm.append(1)
    assert len(lm) == 1
    lm.append(2)
    assert len(lm) == 2


def test_listmap_extend():
    # extent
    lm = _ListMap([1, 2])
    lm.extend([1, 2, 3])
    assert len(lm) == 5
    assert lm == [1, 2, 1, 2, 3]

    lm2 = _ListMap({"a": 1, "b": 2})
    lm.extend(lm2)
    assert isinstance(lm, _ListMap)
    assert len(lm) == 7
    assert len(lm.keys()) == 0

    lm2.extend([5, 6])
    assert isinstance(lm2, _ListMap)
    assert len(lm2) == 4
    assert len(lm2.keys()) == 2
    assert lm2["a"] == 1
    assert lm2[0] == 1
    assert lm2["b"] == 2
    assert lm2[1] == 2
    assert lm2[2] == 5
    assert lm2[3] == 6


def test_listmap_insert():
    lm = _ListMap({"a": 1, "b": 2})
    lm.insert(1, 3)
    assert len(lm) == 3
    assert lm == [1, 3, 2]
    assert lm["a"] == 1
    assert lm["b"] == 2

    lm.insert(-1, 5)
    assert len(lm) == 4
    assert lm == [1, 3, 5, 2]
    assert lm["a"] == 1
    assert lm["b"] == 2

    lm.insert(-2, 10)
    assert len(lm) == 5
    assert lm == [1, 3, 10, 5, 2]
    assert lm["a"] == 1
    assert lm["b"] == 2


def test_listmap_pop():
    lm = _ListMap({"1": 1, "2": 2})
    lm.extend([3, 4])
    item = lm.pop()
    assert item == 4
    assert len(lm) == 3
    item = lm.pop(1)
    assert item == 2
    assert len(lm) == 2

    target = _ListMap({"1": 1})
    target.append(3)
    assert lm == target


def test_listmap_getitem():
    """Test getting items from the collection."""
    lm = _ListMap({"1": 1})
    lm.append(2)
    assert lm[0] == 1
    assert lm[1] == 2
    assert lm[-1] == 2
    assert lm[0:2] == [1, 2]


def test_listmap_setitem():
    """Test setting items in the collection."""
    lm = _ListMap({"1": 1, "2": 2, "3": 3})
    lm[0] = 10
    assert lm == [10, 2, 3]
    assert lm["1"] == 10
    lm[1:3] = [20, 30]
    assert lm == [10, 20, 30]
    lm[0:2] = [20, 30]
    assert lm == [20, 30, 30]
    assert lm["1"] == 20


def test_listmap_add():
    """Test adding two collections together."""
    lm1 = _ListMap([1, 2])
    lm2 = _ListMap({"3": 3, "5": 5})
    combined = lm1 + lm2
    assert len(combined) == 4
    # assert the output type is a list
    assert type(combined) is list
    assert combined == [1, 2, 3, 5]


def test_listmap_clear():
    """Test clearing the collection."""
    lm = _ListMap({"1": 1, "2": 2, "3": 3})
    lm.clear()
    assert len(lm) == 0
    assert len(lm.keys()) == 0


# Dict type properties tests
def test_listmap_keys():
    lm = _ListMap({
        "a": 1,
        "b": 2,
        "c": 3,
    })
    keys = lm.keys()
    assert set(keys) == {"a", "b", "c"}
    assert "a" in lm
    assert "d" not in lm


def test_listmap_repr():
    lm = _ListMap({"a": 1, "b": 2})
    lm.append(3)
    repr_str = repr(lm)
    assert repr_str == "_ListMap([1, 2, 3], keys=['a', 'b'])"
