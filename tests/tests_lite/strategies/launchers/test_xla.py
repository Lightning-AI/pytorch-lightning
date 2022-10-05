from unittest import mock
from unittest.mock import Mock

from tests_lite.helpers.runif import RunIf

from lightning_lite.strategies.launchers.xla import _XLALauncher


@RunIf(skip_windows=True)
def test_xla_launcher_default_start_method(xla_available):
    launcher = _XLALauncher(strategy=Mock())
    assert launcher._start_method == "fork"


@RunIf(skip_windows=True)
def test_xla_launcher_interactive_compatible(xla_available):
    launcher = _XLALauncher(strategy=Mock())
    assert launcher.is_interactive_compatible


@RunIf(skip_windows=True, tpu=True)
@mock.patch("torch_xla.distributed.xla_multiprocessing")
@mock.patch("lightning_lite.strategies.launchers.xla.get_context")
def test_xla_launcher_xmp_spawn(get_context_mock, xmp_mock):
    strategy = Mock()
    launcher = _XLALauncher(strategy=strategy)
    function = Mock()
    launcher.launch(function, "positional-arg", keyword_arg=0)
    queue = get_context_mock.return_value.SimpleQueue.return_value
    get_context_mock.assert_called_with("fork")
    xmp_mock.spawn.assert_called_with(
        launcher._wrapping_function,
        args=(function, ("positional-arg",), {"keyword_arg": 0}, queue),
        nprocs=strategy.num_processes,
        start_method="fork",
    )
    queue.get.assert_called_once_with()
