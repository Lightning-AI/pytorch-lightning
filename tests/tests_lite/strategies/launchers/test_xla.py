from unittest import mock
from unittest.mock import ANY, Mock

from tests_lite.helpers.runif import RunIf

from lightning_lite.strategies.launchers.xla import _XLALauncher


@RunIf(skip_windows=True)
def test_xla_launcher_default_start_method():
    launcher = _XLALauncher(strategy=Mock())
    assert launcher._start_method == "fork"


@RunIf(skip_windows=True)
def test_xla_launcher_interactive_compatible():
    launcher = _XLALauncher(strategy=Mock())
    assert launcher.is_interactive_compatible


@RunIf(skip_windows=True, tpu=True)
@mock.patch("torch_xla.distributed.xla_multiprocessing")
def test_xla_launcher_xmp_spawn(xmp_mock):
    strategy = Mock()
    nprocs = 8
    strategy.parallel_devices = list(range(nprocs))
    launcher = _XLALauncher(strategy=strategy)
    function = Mock()
    launcher.launch(function, "positional-arg", keyword_arg=0)
    xmp_mock.spawn.assert_called_with(
        ANY,
        args=(function, ("positional-arg",), {"keyword_arg": 0}, ANY),
        nprocs=nprocs,
        join=True,
        daemon=False,
        start_method="fork",
    )
