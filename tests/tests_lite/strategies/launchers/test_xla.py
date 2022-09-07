from unittest import mock
from unittest.mock import ANY, Mock

from lightning_lite.strategies.launchers.xla import _XLALauncher


def test_xla_launcher_default_start_method():
    launcher = _XLALauncher(strategy=Mock())
    assert launcher._start_method == "fork"


def test_xla_launcher_interactive_compatible():
    launcher = _XLALauncher(strategy=Mock())
    assert launcher.is_interactive_compatible


@mock.patch("lightning_lite.strategies.launchers.xla.mp")
@mock.patch("lightning_lite.strategies.launchers.xla.xm")
@mock.patch("lightning_lite.strategies.launchers.xla.xmp")
def test_xla_launcher_xmp_spawn(xmp_mock, xm_mock, mp_mock):
    strategy = Mock()
    strategy.parallel_devices = [0, 1, 2, 3]
    launcher = _XLALauncher(strategy=strategy)
    function = Mock()
    launcher.launch(function, "positional-arg", keyword_arg=0)
    # mp_mock.get_context.assert_called_with(start_method)
    xmp_mock.spawn.assert_called_with(
        ANY,
        args=(function, ("positional-arg",), {"keyword_arg": 0}, ANY),
        nprocs=4,
        join=True,
        daemon=False,
        start_method="fork",
    )
