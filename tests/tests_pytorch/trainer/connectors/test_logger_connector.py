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
        _ListMap({"a": 1, "b": 2}),
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
    assert len(lm) == 7
    assert lm == [1, 2, 1, 2, 3, 1, 2]
    assert lm["a"] == 1
    assert lm["b"] == 2


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
    lm = _ListMap([1, 2, 3, 4])
    item = lm.pop()
    assert item == 4
    assert len(lm) == 3
    item = lm.pop(1)
    assert item == 2
    assert len(lm) == 2
    assert lm == [1, 3]


def test_listmap_getitem():
    """Test getting items from the collection."""
    lm = _ListMap([1, 2])
    assert lm[0] == 1
    assert lm[1] == 2
    assert lm[-1] == 2
    assert lm[0:2] == [1, 2]


def test_listmap_setitem():
    """Test setting items in the collection."""
    lm = _ListMap([1, 2, 3])
    lm[0] = 10
    assert lm == [10, 2, 3]
    lm[1:3] = [20, 30]
    assert lm == [10, 20, 30]


def test_listmap_add():
    """Test adding two collections together."""
    lm1 = _ListMap([1, 2])
    lm2 = _ListMap({"3": 3, "5": 5})
    combined = lm1 + lm2
    assert isinstance(combined, _ListMap)
    assert len(combined) == 4
    assert combined is not lm1
    assert combined == [1, 2, 3, 5]
    assert combined["3"] == 3
    assert combined["5"] == 5

    ori_lm1_id = id(lm1)

    lm1 += lm2
    assert ori_lm1_id == id(lm1)
    assert isinstance(lm1, _ListMap)
    assert len(lm1) == 4
    assert lm1 == [1, 2, 3, 5]
    assert lm1["3"] == 3
    assert lm1["5"] == 5

    lm3 = _ListMap({"3": 3, "5": 5})
    lm4 = lm2 + lm3
    assert len(lm4) == 4
    assert lm4 == [3, 5, 3, 5]
    assert lm4["3"] == 3
    assert lm4["5"] == 5


def test_listmap_remove():
    """Test removing items from the collection."""
    lm = _ListMap([1, 2, 3])
    lm.remove(2)
    assert len(lm) == 2
    assert 2 not in lm


def test_listmap_reverse():
    """Test reversing the collection."""
    lm = _ListMap({"1": 1, "2": 2, "3": 3})
    lm.reverse()
    assert lm == [3, 2, 1]
    for (key, value), expected in zip(lm.items(), [("1", 1), ("2", 2), ("3", 3)]):
        assert (key, value) == expected


def test_listmap_reversed():
    """Test reversed iterator of the collection."""
    lm = _ListMap({"1": 1, "2": 2, "3": 3})
    rev_lm = list(reversed(lm))
    assert rev_lm == [3, 2, 1]


def test_listmap_clear():
    """Test clearing the collection."""
    lm = _ListMap({"1": 1, "2": 2, "3": 3})
    lm.clear()
    assert len(lm) == 0
    assert len(lm.keys()) == 0


def test_listmap_delitem():
    """Test deleting items from the collection."""
    lm = _ListMap({"a": 1, "b": 2, "c": 3})
    lm.extend([3, 4, 5])
    del lm["b"]
    assert len(lm) == 5
    assert "b" not in lm
    del lm[0]
    assert len(lm) == 4
    assert "a" not in lm
    assert lm == [3, 3, 4, 5]

    del lm[-1]
    assert len(lm) == 3
    assert lm == [3, 3, 4]

    del lm[-2:]
    assert len(lm) == 1
    assert lm == [3]


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


def test_listmap_values():
    lm = _ListMap({
        "a": 1,
        "b": 2,
        "c": 3,
    })
    values = lm.values()
    assert set(values) == {1, 2, 3}


def test_listmap_dict_items():
    lm = _ListMap({
        "a": 1,
        "b": 2,
        "c": 3,
    })
    items = lm.items()
    assert set(items) == {("a", 1), ("b", 2), ("c", 3)}


def test_listmap_dict_pop():
    lm = _ListMap({
        "a": 1,
        "b": 2,
        "c": 3,
    })
    value = lm.pop("b")
    assert value == 2
    assert "b" not in lm
    assert len(lm) == 2

    value = lm.pop(0)
    assert value == 1
    assert lm["c"] == 3  # still accessible by key
    assert len(lm) == 1
    with pytest.raises(KeyError):
        lm["a"]  # "a" was removed


def test_listmap_dict_setitem():
    lm = _ListMap({
        "a": 1,
        "b": 2,
    })
    lm["b"] = 20
    assert lm["b"] == 20
    lm["c"] = 3
    assert lm["c"] == 3
    assert len(lm) == 3


def test_listmap_sort():
    lm = _ListMap({"b": 1, "c": 3, "a": 2, "z": -7})

    lm.extend([-1, -2, 5])
    lm.sort(key=lambda x: abs(x))
    assert lm == [1, -1, 2, -2, 3, 5, -7]
    assert lm["a"] == 2
    assert lm["b"] == 1
    assert lm["c"] == 3
    assert lm["z"] == -7

    lm = _ListMap({"b": 1, "c": 3, "a": 2, "z": -7})
    lm.sort(reverse=True)
    assert lm == [3, 2, 1, -7]
    assert lm["a"] == 2
    assert lm["b"] == 1
    assert lm["c"] == 3
    assert lm["z"] == -7


def test_listmap_get():
    lm = _ListMap({"a": 1, "b": 2, "c": 3})
    assert lm.get("b") == 2
    assert lm.get("d") is None
    assert lm.get("d", 10) == 10


def test_listmap_setitem_append():
    lm = _ListMap({"a": 1, "b": 2})
    lm.append(3)
    lm["c"] = 3

    assert lm == [1, 2, 3, 3]
    assert lm["c"] == 3

    lm.remove(3)
    assert lm == [1, 2, 3]
    assert lm["c"] == 3

    lm.remove(3)
    assert lm == [1, 2]
    with pytest.raises(KeyError):
        lm["c"]  # "c" was removed


def test_listmap_repr():
    lm = _ListMap({"a": 1, "b": 2})
    lm.append(3)
    repr_str = repr(lm)
    assert repr_str == "_ListMap([1, 2, 3], keys=['a', 'b'])"
