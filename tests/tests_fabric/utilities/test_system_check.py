import os
from unittest import mock

import torch
from lightning.fabric.utilities.system_check import _run_all_reduce_test


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("lightning.fabric.utilities.system_check.torch.device", return_value=torch.device("cpu"))
@mock.patch("lightning.fabric.utilities.system_check.torch.cuda.set_device")
@mock.patch("lightning.fabric.utilities.system_check.torch.distributed")
def test_run_all_reduce_test(dist_mock, set_device_mock, __):
    _run_all_reduce_test(local_rank=1, world_size=4)
    set_device_mock.assert_called_once()
    dist_mock.init_process_group.assert_called_once()
    dist_mock.barrier.assert_called_once()
    dist_mock.all_reduce.assert_called_once()
