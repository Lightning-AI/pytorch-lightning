import pytest
from tests_lite.helpers.runif import RunIf

from lightning_lite.strategies import XLAStrategy
from lightning_lite.strategies.launchers.xla import _XLALauncher
from lightning_lite.utilities.distributed import ReduceOp


def xla_launch(function):
    strategy = XLAStrategy(parallel_devices=list(range(8)))
    launcher = _XLALauncher(strategy=strategy)
    return launcher.launch(function, strategy)


def broadcast_on_tpu_fn(strategy):
    obj = ("ver_0.5", "logger_name", strategy.local_rank)
    result = strategy.broadcast(obj)
    assert result == ("ver_0.5", "logger_name", 0)


@RunIf(tpu=True)
def test_broadcast_on_tpu():
    """Checks if an object from the main process is broadcasted to other processes correctly."""
    xla_launch(broadcast_on_tpu_fn)


def tpu_reduce_fn(strategy):
    with pytest.raises(ValueError, match="XLAStrategy only supports"):
        strategy.reduce(1, reduce_op="undefined")

    with pytest.raises(ValueError, match="XLAStrategy only supports"):
        strategy.reduce(1, reduce_op=ReduceOp.MAX)

        # it is faster to loop over here than to parameterize the test
        for reduce_op in ("mean", "AVG", "sum", ReduceOp.SUM):
            result = strategy.reduce(1, reduce_op=reduce_op)
            if isinstance(reduce_op, str) and reduce_op.lower() in ("mean", "avg"):
                assert result.item() == 1
            else:
                assert result.item() == 8


@RunIf(tpu=True)
def test_tpu_reduce():
    """Test tpu spawn reduce operation."""
    xla_launch(tpu_reduce_fn)
