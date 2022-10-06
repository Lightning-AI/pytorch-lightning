import datetime
from unittest import mock

import pytest
import torch
from torch.distributed import ReduceOp

from lightning_lite.plugins.collectives import TorchCollective

PASSED_TENSOR = mock.Mock()


@pytest.mark.parametrize(
    ["fn_name", "orig_call", "args", "return_index"],
    [
        ("send", "torch.distributed.send", [PASSED_TENSOR, 0, 0], -1),
        ("recv", "torch.distributed.recv", [PASSED_TENSOR, 0, 0], 0),
        ("broadcast", "torch.distributed.broadcast", [PASSED_TENSOR, 0], 0),
        ("all_reduce", "torch.distributed.all_reduce", [PASSED_TENSOR, ReduceOp.SUM], 0),
        ("reduce", "torch.distributed.reduce", [PASSED_TENSOR, 0, ReduceOp.SUM], 0),
        ("all_gather", "torch.distributed.all_gather", [[PASSED_TENSOR], PASSED_TENSOR], 0),
        ("gather", "torch.distributed.gather", [PASSED_TENSOR, [PASSED_TENSOR], 0], 1),
        ("scatter", "torch.distributed.scatter", [PASSED_TENSOR, [PASSED_TENSOR], 0], 0),
        ("reduce_scatter", "torch.distributed.reduce_scatter", [PASSED_TENSOR, [PASSED_TENSOR], ReduceOp.SUM], 0),
        ("all_to_all", "torch.distributed.all_to_all", [[PASSED_TENSOR], [PASSED_TENSOR]], 0),
        ("barrier", "torch.distributed.barrier", [[0]], -1),
        ("all_gather_object", "torch.distributed.all_gather_object", [[object()], object()], 0),
        ("broadcast_object_list", "torch.distributed.broadcast_object_list", [[object()], 0, torch.device("cpu")], 0),
        ("gather_object", "torch.distributed.gather_object", [object(), [object()], 0], 1),
        ("scatter_object_list", "torch.distributed.scatter_object_list", [[object()], [object()], 0], 1),
        ("monitored_barrier", "torch.distributed.monitored_barrier", [datetime.timedelta(seconds=1), False], -1),
    ],
)
def test_collective_calls_with_created_group(fn_name, orig_call, args, return_index):
    with mock.patch(orig_call) as mock_call, mock.patch("torch.distributed.init_process_group"):
        collective = TorchCollective().create_group()
        result = collective.__getattribute__(fn_name)(*args)
        mock_call.assert_called_once_with(*args, group=collective.group)
        if return_index != -1:
            assert result == args[return_index]
