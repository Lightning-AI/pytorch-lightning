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
    ["fn_name", "orig_call", "kwargs", "return_key"],
    [
        ("send", "torch.distributed.send", dict(tensor=PASSED_TENSOR, dst=0, tag=0), None),
        ("recv", "torch.distributed.recv", dict(tensor=PASSED_TENSOR, src=0, tag=0), "tensor"),
        ("broadcast", "torch.distributed.broadcast", dict(tensor=PASSED_TENSOR, src=0), "tensor"),
        ("all_reduce", "torch.distributed.all_reduce", dict(tensor=PASSED_TENSOR, op=ReduceOp.SUM), "tensor"),
        ("reduce", "torch.distributed.reduce", dict(tensor=PASSED_TENSOR, dst=0, op=ReduceOp.SUM), "tensor"),
        (
            "all_gather",
            "torch.distributed.all_gather",
            dict(tensor_list=[PASSED_TENSOR], tensor=PASSED_TENSOR),
            "tensor_list",
        ),
        (
            "gather",
            "torch.distributed.gather",
            dict(tensor=PASSED_TENSOR, gather_list=[PASSED_TENSOR], dst=0),
            "gather_list",
        ),
        (
            "scatter",
            "torch.distributed.scatter",
            dict(tensor=PASSED_TENSOR, scatter_list=[PASSED_TENSOR], src=0),
            "tensor",
        ),
        (
            "reduce_scatter",
            "torch.distributed.reduce_scatter",
            dict(output=PASSED_TENSOR, input_list=[PASSED_TENSOR], op=ReduceOp.SUM),
            "output",
        ),
        (
            "all_to_all",
            "torch.distributed.all_to_all",
            dict(output_tensor_list=[PASSED_TENSOR], input_tensor_list=[PASSED_TENSOR]),
            "output_tensor_list",
        ),
        ("barrier", "torch.distributed.barrier", dict(device_ids=[0]), None),
        (
            "all_gather_object",
            "torch.distributed.all_gather_object",
            dict(object_list=[PASSED_OBJECT], obj=PASSED_OBJECT),
            "object_list",
        ),
        (
            "broadcast_object_list",
            "torch.distributed.broadcast_object_list",
            dict(object_list=[PASSED_OBJECT], src=0, device=torch.device("cpu")),
            "object_list",
        ),
        (
            "gather_object",
            "torch.distributed.gather_object",
            dict(obj=PASSED_OBJECT, object_gather_list=[PASSED_OBJECT], dst=0),
            "object_gather_list",
        ),
        (
            "scatter_object_list",
            "torch.distributed.scatter_object_list",
            dict(scatter_object_output_list=[PASSED_OBJECT], scatter_object_input_list=[PASSED_OBJECT], src=0),
            "scatter_object_output_list",
        ),
        (
            "monitored_barrier",
            "torch.distributed.monitored_barrier",
            dict(timeout=datetime.timedelta(seconds=1), wait_all_ranks=False),
            None,
        ),
    ],
)
def test_collective_calls_with_created_group(fn_name, orig_call, kwargs, return_key):
    with mock.patch("torch.distributed.is_available", return_value=True), mock.patch(
        "torch.distributed.init_process_group"
    ):
        collective = TorchCollective(instantiate_group=True)
    fn = getattr(collective, fn_name)
    with mock.patch(orig_call, autospec=True) as mock_call:
        result = fn(**kwargs)
    mock_call.assert_called_once_with(**kwargs, group=collective.group)
    if return_key is not None:
        assert result == kwargs[return_key]
