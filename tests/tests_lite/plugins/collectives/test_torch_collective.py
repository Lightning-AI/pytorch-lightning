import datetime
from unittest import mock

import pytest
import torch

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:
    ReduceOp = mock.Mock()

from lightning_lite.plugins.collectives import TorchCollective

PASSED_TENSOR = mock.Mock()
PASSED_OBJECT = mock.Mock()


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
        (
            "broadcast_object_list",
            {"object_list": [PASSED_OBJECT], "src": 0, "device": torch.device("cpu")},
            "object_list",
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
