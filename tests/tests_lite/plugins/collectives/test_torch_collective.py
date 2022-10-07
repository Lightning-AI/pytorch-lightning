import datetime
from unittest import mock

import pytest
import torch
from tests_lite.helpers.runif import RunIf

from lightning_lite.accelerators import CPUAccelerator
from lightning_lite.plugins.collectives import TorchCollective
from lightning_lite.plugins.environments import LightningEnvironment
from lightning_lite.strategies import DDPStrategy
from lightning_lite.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning_lite.utilities.imports import _TORCH_GREATER_EQUAL_1_11

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:
    ReduceOp = mock.Mock()

PASSED_TENSOR = mock.Mock()
PASSED_OBJECT = mock.Mock()


@pytest.fixture(autouse=True)
def check_destroy_group():
    if torch.distributed.is_available():
        assert not torch.distributed.is_initialized()
    with mock.patch(
        "lightning_lite.plugins.collectives.torch_collective.TorchCollective.new_group",
        wraps=TorchCollective.new_group,
    ) as mock_new, mock.patch(
        "lightning_lite.plugins.collectives.torch_collective.TorchCollective.destroy_group",
        wraps=TorchCollective.destroy_group,
    ) as mock_destroy:
        yield
        assert (
            mock_new.call_count == mock_destroy.call_count
        ), "new_group and destroy_group should be called the same number of times"


@pytest.mark.parametrize(
    ["fn_name", "kwargs", "return_key"],
    [
        ("send", {"tensor": PASSED_TENSOR, "dst": 0, "tag": 0}, None),
        ("recv", {"tensor": PASSED_TENSOR, "src": 0, "tag": 0}, "tensor"),
        ("broadcast", {"tensor": PASSED_TENSOR, "src": 0}, "tensor"),
        ("all_reduce", {"tensor": PASSED_TENSOR, "op": ReduceOp.SUM}, "tensor"),
        ("reduce", {"tensor": PASSED_TENSOR, "dst": 0, "op": ReduceOp.SUM}, "tensor"),
        ("all_gather", {"tensor_list": [PASSED_TENSOR], "tensor": PASSED_TENSOR}, "tensor_list"),
        ("gather", {"tensor": PASSED_TENSOR, "gather_list": [PASSED_TENSOR], "dst": 0}, "gather_list"),
        ("scatter", {"tensor": PASSED_TENSOR, "scatter_list": [PASSED_TENSOR], "src": 0}, "tensor"),
        ("reduce_scatter", {"output": PASSED_TENSOR, "input_list": [PASSED_TENSOR], "op": ReduceOp.SUM}, "output"),
        (
            "all_to_all",
            {"output_tensor_list": [PASSED_TENSOR], "input_tensor_list": [PASSED_TENSOR]},
            "output_tensor_list",
        ),
        ("barrier", {"device_ids": [0]}, None),
        ("all_gather_object", {"object_list": [PASSED_OBJECT], "obj": PASSED_OBJECT}, "object_list"),
        pytest.param(
            "broadcast_object_list",
            {"object_list": [PASSED_OBJECT], "src": 0},
            "object_list",
            marks=RunIf(max_torch="1.10"),
        ),
        pytest.param(
            "broadcast_object_list",
            {"object_list": [PASSED_OBJECT], "src": 0, "device": torch.device("cpu")},
            "object_list",
            marks=RunIf(min_torch="1.10"),
        ),
        (
            "gather_object",
            {"obj": PASSED_OBJECT, "object_gather_list": [PASSED_OBJECT], "dst": 0},
            "object_gather_list",
        ),
        (
            "scatter_object_list",
            {"scatter_object_output_list": [PASSED_OBJECT], "scatter_object_input_list": [PASSED_OBJECT], "src": 0},
            "scatter_object_output_list",
        ),
        ("monitored_barrier", {"timeout": datetime.timedelta(seconds=1), "wait_all_ranks": False}, None),
    ],
)
@RunIf(distributed=True)
def test_collective_calls_with_created_group(fn_name, kwargs, return_key):
    with mock.patch("torch.distributed.is_available", return_value=True), mock.patch(
        "torch.distributed.init_process_group"
    ), mock.patch("torch.distributed.new_group"):
        collective = TorchCollective(instantiate_group=True)
    fn = getattr(collective, fn_name)
    with mock.patch(f"torch.distributed.{fn_name}", autospec=True) as mock_call:
        result = fn(**kwargs)
    mock_call.assert_called_once_with(**kwargs, group=collective.group)
    if return_key is not None:
        assert result == kwargs[return_key]

    with mock.patch("torch.distributed.destroy_process_group"):
        collective.teardown()


@RunIf(distributed=True)
def test_convert_ops():
    # Test regular names
    assert TorchCollective._convert_to_native_op("band") == ReduceOp.BAND
    assert TorchCollective._convert_to_native_op("bor") == ReduceOp.BOR
    assert TorchCollective._convert_to_native_op("bxor") == ReduceOp.BXOR
    assert TorchCollective._convert_to_native_op("max") == ReduceOp.MAX
    assert TorchCollective._convert_to_native_op("min") == ReduceOp.MIN
    assert TorchCollective._convert_to_native_op("product") == ReduceOp.PRODUCT
    assert TorchCollective._convert_to_native_op("sum") == ReduceOp.SUM
    # Test we are passing through native ops without change
    assert TorchCollective._convert_to_native_op(ReduceOp.BAND) == ReduceOp.BAND
    assert TorchCollective._convert_to_native_op(ReduceOp.BOR) == ReduceOp.BOR
    assert TorchCollective._convert_to_native_op(ReduceOp.BXOR) == ReduceOp.BXOR
    assert TorchCollective._convert_to_native_op(ReduceOp.MAX) == ReduceOp.MAX
    assert TorchCollective._convert_to_native_op(ReduceOp.MIN) == ReduceOp.MIN
    assert TorchCollective._convert_to_native_op(ReduceOp.PRODUCT) == ReduceOp.PRODUCT
    assert TorchCollective._convert_to_native_op(ReduceOp.SUM) == ReduceOp.SUM
    # Test we are handling different casing properly
    assert TorchCollective._convert_to_native_op("BOR") == ReduceOp.BOR
    assert TorchCollective._convert_to_native_op("BoR") == ReduceOp.BOR

    # AVG is very recent!
    if _TORCH_GREATER_EQUAL_1_11:
        assert TorchCollective._convert_to_native_op("avg") == ReduceOp.AVG

    # Test invalid type
    with pytest.raises(ValueError, match="op 1 should be a `str` or `ReduceOp`"):
        TorchCollective._convert_to_native_op(1)

    # Test invalid string
    with pytest.raises(ValueError, match="op 'INVALID' is not a member of `ReduceOp`"):
        TorchCollective._convert_to_native_op("invalid")


@RunIf(distributed=True)
def test_repeated_create_and_destroy():
    with mock.patch("torch.distributed.init_process_group") as init_mock, mock.patch(
        "torch.distributed.new_group"
    ) as new_mock:
        collective = TorchCollective(instantiate_group=True)
    init_mock.assert_called_once()
    new_mock.assert_called_once()

    with pytest.raises(RuntimeError, match="TorchCollective` already owns a group"):
        collective.create_group()

    with mock.patch("torch.distributed.destroy_process_group") as destroy_mock:
        collective.teardown()
    with pytest.raises(RuntimeError, match="TorchCollective` does not own a group to destroy"):
        collective.teardown()

    destroy_mock.assert_called_once_with(new_mock.return_value)
    assert collective._group is None


@RunIf(distributed=True)
def test_init_and_new_group():
    with mock.patch("torch.distributed.init_process_group"), mock.patch(
        "torch.distributed.new_group"
    ) as new_mock, mock.patch("torch.distributed.destroy_process_group") as destroy_mock:
        collective = TorchCollective(instantiate_group=True)
        collective.teardown()
        collective.create_group()
        collective.teardown()
    assert new_mock.call_count == 2
    assert destroy_mock.call_count == 2


def collective_launch(fn, parallel_devices):
    strategy = DDPStrategy(
        accelerator=CPUAccelerator(), parallel_devices=parallel_devices, cluster_environment=LightningEnvironment()
    )
    launcher = _MultiProcessingLauncher(strategy=strategy)
    collective = TorchCollective(
        init_kwargs={
            "rank": strategy.global_rank,
            "world_size": strategy.num_processes,
            "main_address": "localhost",
            "backend": "gloo",
        },
    )
    launcher.launch(fn, strategy, collective)


def _test_distributed_collectives_fn(strategy, collective):
    rank = strategy._local_rank
    collective.create_group(init_kwargs={"rank": rank})

    # all_gather
    tensor_list = [torch.zeros(2, dtype=torch.long) for _ in range(strategy.num_processes)]
    this = torch.arange(2, dtype=torch.long) + 2 * rank
    out = collective.all_gather(tensor_list, this)
    expected = torch.arange(2 * strategy.num_processes).split(2)
    torch.testing.assert_close(tuple(out), expected)

    # reduce
    this = torch.tensor(rank + 1)
    out = collective.reduce(this, dst=0, op="max")
    expected = torch.tensor(strategy.num_processes) if rank == 0 else this
    torch.testing.assert_close(out, expected)

    # all_reduce
    this = torch.tensor(rank + 1)
    out = collective.all_reduce(this, op="min")
    expected = torch.tensor(1)
    torch.testing.assert_close(out, expected)

    collective.teardown()


@RunIf(distributed=True)
@pytest.mark.parametrize("n", (1, 2))
def test_collectives_distributed(n):
    collective_launch(_test_distributed_collectives_fn, [torch.device("cpu")] * n)
