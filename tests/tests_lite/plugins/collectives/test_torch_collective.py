import datetime
from unittest import mock

import pytest
import torch
from tests_lite.helpers.runif import RunIf

from lightning_lite.plugins.collectives import TorchCollective
from lightning_lite.utilities.imports import _TORCH_GREATER_EQUAL_1_11

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:
    ReduceOp = mock.Mock()


PASSED_TENSOR = mock.Mock()
PASSED_OBJECT = mock.Mock()


@pytest.fixture(autouse=True)
def check_destroy_group():
    with mock.patch(
        "lightning_lite.plugins.collectives.torch_collective.TorchCollective.init_group",
        wraps=TorchCollective.init_group,
    ) as mock_create, mock.patch(
        "lightning_lite.plugins.collectives.torch_collective.TorchCollective.destroy_group",
        wraps=TorchCollective.destroy_group,
    ) as mock_destroy:
        yield
        assert (
            mock_create.call_count == mock_destroy.call_count
        ), "init_group and destroy_group should be called the same number of times"


@pytest.mark.parametrize(
    ["fn_name", "kwargs", "return_key"],
    [
        ("send", {"tensor": PASSED_TENSOR, "dst": 0, "tag": 0}, None),
        ("recv", {"tensor": PASSED_TENSOR, "src": 0, "tag": 0}, "tensor"),
        ("broadcast", {"tensor": PASSED_TENSOR, "src": 0}, "tensor"),
        ("all_reduce", {"tensor": PASSED_TENSOR, "op": ReduceOp.SUM}, "tensor"),
        ("reduce", {"tensor": PASSED_TENSOR, "dst": 0, "op": ReduceOp.SUM}, "tensor"),
        (
            "all_gather",
            {"tensor_list": [PASSED_TENSOR], "tensor": PASSED_TENSOR},
            "tensor_list",
        ),
        (
            "gather",
            {"tensor": PASSED_TENSOR, "gather_list": [PASSED_TENSOR], "dst": 0},
            "gather_list",
        ),
        (
            "scatter",
            {"tensor": PASSED_TENSOR, "scatter_list": [PASSED_TENSOR], "src": 0},
            "tensor",
        ),
        (
            "reduce_scatter",
            {"output": PASSED_TENSOR, "input_list": [PASSED_TENSOR], "op": ReduceOp.SUM},
            "output",
        ),
        (
            "all_to_all",
            {"output_tensor_list": [PASSED_TENSOR], "input_tensor_list": [PASSED_TENSOR]},
            "output_tensor_list",
        ),
        ("barrier", {"device_ids": [0]}, None),
        (
            "all_gather_object",
            {"object_list": [PASSED_OBJECT], "obj": PASSED_OBJECT},
            "object_list",
        ),
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
        (
            "monitored_barrier",
            {"timeout": datetime.timedelta(seconds=1), "wait_all_ranks": False},
            None,
        ),
    ],
)
@RunIf(distributed=True)
def test_collective_calls_with_created_group(fn_name, kwargs, return_key):
    with mock.patch("torch.distributed.is_available", return_value=True), mock.patch(
        "torch.distributed.init_process_group"
    ):
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
        "torch.distributed.destroy_process_group"
    ) as destroy_mock:
        collective = TorchCollective(instantiate_group=True)
        with pytest.raises(RuntimeError, match="TorchCollective already owns a group."):
            collective.create_group()

        collective.teardown()
        with pytest.raises(RuntimeError, match="TorchCollective does not own a group to destroy."):
            collective.teardown()

        destroy_mock.assert_called_once_with(init_mock.return_value)


@RunIf(distributed=True)
def test_create_group_pass_params():
    collective = TorchCollective(arg1=None, arg2=10)
    with mock.patch("torch.distributed.init_process_group") as init_mock:
        collective.create_group(arg2=2, arg3=3)
    init_mock.assert_called_once_with(arg1=None, arg2=2, arg3=3)
    assert collective._group_kwargs == {"arg1": None, "arg2": 2, "arg3": 3}

    with mock.patch("torch.distributed.destroy_process_group"):
        collective.teardown()
