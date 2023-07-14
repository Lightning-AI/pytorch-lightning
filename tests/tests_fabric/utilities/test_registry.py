import contextlib
from unittest import mock
from unittest.mock import Mock

from lightning.fabric.utilities.imports import _PYTHON_GREATER_EQUAL_3_8_0, _PYTHON_GREATER_EQUAL_3_10_0
from lightning.fabric.utilities.registry import _load_external_callbacks


class ExternalCallback:
    """A callback in another library that gets registered through entry points."""

    pass


def test_load_external_callbacks():
    """Test that the connector collects Callback instances from factories registered through entry points."""

    def factory_no_callback():
        return []

    def factory_one_callback():
        return ExternalCallback()

    def factory_one_callback_list():
        return [ExternalCallback()]

    def factory_multiple_callbacks_list():
        return [ExternalCallback(), ExternalCallback()]

    with _make_entry_point_query_mock(factory_no_callback):
        callbacks = _load_external_callbacks("lightning.pytorch.callbacks_factory")
    assert callbacks == []

    with _make_entry_point_query_mock(factory_one_callback):
        callbacks = _load_external_callbacks("lightning.pytorch.callbacks_factory")
    assert isinstance(callbacks[0], ExternalCallback)

    with _make_entry_point_query_mock(factory_one_callback_list):
        callbacks = _load_external_callbacks("lightning.pytorch.callbacks_factory")
    assert isinstance(callbacks[0], ExternalCallback)

    with _make_entry_point_query_mock(factory_multiple_callbacks_list):
        callbacks = _load_external_callbacks("lightning.pytorch.callbacks_factory")
    assert isinstance(callbacks[0], ExternalCallback)
    assert isinstance(callbacks[1], ExternalCallback)


@contextlib.contextmanager
def _make_entry_point_query_mock(callback_factory):
    query_mock = Mock()
    entry_point = Mock()
    entry_point.name = "mocked"
    entry_point.load.return_value = callback_factory
    if _PYTHON_GREATER_EQUAL_3_10_0:
        query_mock.return_value = [entry_point]
        import_path = "importlib.metadata.entry_points"
    elif _PYTHON_GREATER_EQUAL_3_8_0:
        query_mock().get.return_value = [entry_point]
        import_path = "importlib.metadata.entry_points"
    else:
        query_mock.return_value = [entry_point]
        import_path = "pkg_resources.iter_entry_points"
    with mock.patch(import_path, query_mock):
        yield
