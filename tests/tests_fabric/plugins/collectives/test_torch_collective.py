import contextlib
import datetime
import os
from functools import partial
from unittest import mock

import pytest
import torch
from lightning.fabric.accelerators import CPUAccelerator, CUDAAccelerator
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.launchers.multiprocessing import _MultiProcessingLauncher

from tests_fabric.helpers.runif import RunIf

if TorchCollective.is_available():
    from torch.distributed import ReduceOp
else:
    ReduceOp = mock.Mock()

skip_distributed_unavailable = pytest.mark.skipif(
    not TorchCollective.is_available(), reason="torch.distributed unavailable"
)

PASSED_TENSOR = mock.Mock()
PASSED_OBJECT = mock.Mock()


@contextlib.contextmanager
def check_destroy_group():
    with mock.patch(
        "lightning.fabric.plugins.collectives.torch_collective.TorchCollective.new_group",
        wraps=TorchCollective.new_group,
    ) as mock_new, mock.patch(
        "lightning.fabric.plugins.collectives.torch_collective.TorchCollective.destroy_group",
        wraps=TorchCollective.destroy_group,
    ) as mock_destroy:
        yield
    # 0 to account for tests that mock distributed
    # -1 to account for destroying the default process group
    expected = 0 if mock_new.call_count == 0 else mock_destroy.call_count - 1
    assert mock_new.call_count == expected, f"new_group={mock_new.call_count}, destroy_group={mock_destroy.call_count}"
    if TorchCollective.is_available():
        assert not torch.distributed.distributed_c10d._pg_map
        assert not TorchCollective.is_initialized()


@pytest.mark.parametrize(
    ("fn_name", "kwargs", "return_key"),
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
        ("monitored_barrier", {"timeout": datetime.timedelta(seconds=1), "wait_all_ranks": False}, None),
    ],
)
@skip_distributed_unavailable
def test_collective_calls_with_created_group(fn_name, kwargs, return_key):
    collective = TorchCollective()
    with mock.patch("torch.distributed.init_process_group"):
        collective.setup()
    with mock.patch("torch.distributed.new_group"):
        collective.create_group()
    fn = getattr(collective, fn_name)
    with mock.patch(f"torch.distributed.{fn_name}", autospec=True) as mock_call:
        result = fn(**kwargs)
    mock_call.assert_called_once_with(**kwargs, group=collective.group)
    if return_key is not None:
        assert result == kwargs[return_key]

    with mock.patch("torch.distributed.destroy_process_group"):
        collective.teardown()


@skip_distributed_unavailable
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
    assert TorchCollective._convert_to_native_op("avg") == ReduceOp.AVG

    # Test invalid type
    with pytest.raises(ValueError, match="Unsupported op 1 of type int"):
        TorchCollective._convert_to_native_op(1)

    # Test invalid string
    with pytest.raises(ValueError, match="op 'INVALID' is not a member of `Red"):
        TorchCollective._convert_to_native_op("invalid")

    # Test RedOpType
    assert TorchCollective._convert_to_native_op(ReduceOp.RedOpType.AVG) == ReduceOp.RedOpType.AVG
    op = torch.distributed._make_nccl_premul_sum(2.0)  # this returns a ReduceOp
    assert TorchCollective._convert_to_native_op(op) == ReduceOp.PREMUL_SUM
    assert TorchCollective._convert_to_native_op("premul_sum") == ReduceOp.PREMUL_SUM


@skip_distributed_unavailable
@mock.patch.dict(os.environ, {}, clear=True)
def test_repeated_create_and_destroy():
    collective = TorchCollective()
    with mock.patch("torch.distributed.init_process_group"):
        collective.setup(main_address="foo", main_port="123")

    assert not os.environ

    with mock.patch("torch.distributed.new_group") as new_mock:
        collective.create_group()
    new_mock.assert_called_once()

    with pytest.raises(RuntimeError, match="TorchCollective` already owns a group"):
        collective.create_group()

    with mock.patch.dict("torch.distributed.distributed_c10d._pg_map", {collective.group: ("", None)}), mock.patch(
        "torch.distributed.destroy_process_group"
    ) as destroy_mock:
        collective.teardown()
    # this would be called twice if `init_process_group` wasn't patched. once for the group and once for the default
    # group
    destroy_mock.assert_called_once()

    assert not os.environ

    with pytest.raises(RuntimeError, match="TorchCollective` does not own a group"):
        collective.teardown()
    destroy_mock.assert_called_once_with(new_mock.return_value)
    assert collective._group is None

    with mock.patch("torch.distributed.new_group"), mock.patch("torch.distributed.destroy_process_group"):
        # check we can create_group again. also chaining
        collective.create_group().teardown()


def collective_launch(fn, parallel_devices, num_groups=1):
    device_to_accelerator = {"cuda": CUDAAccelerator, "cpu": CPUAccelerator}
    accelerator_cls = device_to_accelerator[parallel_devices[0].type]
    strategy = DDPStrategy(
        accelerator=accelerator_cls(),
        parallel_devices=parallel_devices,
        cluster_environment=LightningEnvironment(),
        start_method="spawn",
    )
    launcher = _MultiProcessingLauncher(strategy=strategy)
    collectives = [TorchCollective() for _ in range(num_groups)]
    wrapped = partial(wrap_launch_function, fn, strategy, collectives)
    return launcher.launch(wrapped, strategy, *collectives)


def wrap_launch_function(fn, strategy, collectives, *args, **kwargs):
    strategy._set_world_ranks()
    collectives[0].setup(  # only one needs to setup
        world_size=strategy.num_processes,
        main_address="localhost",
        backend=strategy._get_process_group_backend(),
        rank=strategy.global_rank,
    )
    with check_destroy_group():  # manually use the fixture for the assertions
        fn(*args, **kwargs)
        # not necessary since they will be destroyed on process destruction, only added to fulfill the assertions
        for c in collectives:
            c.teardown()


def _test_distributed_collectives_fn(strategy, collective):
    collective.create_group()

    # all_gather
    tensor_list = [torch.zeros(2, dtype=torch.long) for _ in range(strategy.num_processes)]
    this = torch.arange(2, dtype=torch.long) + 2 * strategy.global_rank
    out = collective.all_gather(tensor_list, this)
    expected = torch.arange(2 * strategy.num_processes).split(2)
    torch.testing.assert_close(tuple(out), expected)

    # reduce
    this = torch.tensor(strategy.global_rank + 1)
    out = collective.reduce(this, dst=0, op="max")
    expected = torch.tensor(strategy.num_processes) if strategy.global_rank == 0 else this
    torch.testing.assert_close(out, expected)

    # all_reduce
    this = torch.tensor(strategy.global_rank + 1)
    out = collective.all_reduce(this, op=ReduceOp.MIN)
    expected = torch.tensor(1)
    torch.testing.assert_close(out, expected)


@pytest.mark.skip(reason="test hangs too often")
@skip_distributed_unavailable
@pytest.mark.parametrize(
    "n", [1, pytest.param(2, marks=[RunIf(skip_windows=True), pytest.mark.xfail(raises=TimeoutError, strict=False)])]
)
def test_collectives_distributed(n):
    collective_launch(_test_distributed_collectives_fn, [torch.device("cpu")] * n)


def _test_distributed_collectives_cuda_fn(strategy, collective):
    collective.create_group()

    this = torch.tensor(1.5, device=strategy.root_device)
    premul_sum = torch.distributed._make_nccl_premul_sum(2.0)
    out = collective.all_reduce(this, op=premul_sum)
    assert out == 3


@skip_distributed_unavailable
@RunIf(min_cuda_gpus=1)
def test_collectives_distributed_cuda():
    collective_launch(_test_distributed_collectives_cuda_fn, [torch.device("cuda")])


def _test_two_groups(strategy, left_collective, right_collective):
    left_collective.create_group(ranks=[0, 1])
    right_collective.create_group(ranks=[1, 2])

    tensor = torch.tensor(strategy.global_rank)
    if strategy.global_rank in (0, 1):
        tensor = left_collective.all_reduce(tensor)
        assert tensor == 1
    right_collective.barrier()  # avoids deadlock for global rank 1
    if strategy.global_rank in (1, 2):
        tensor = right_collective.all_reduce(tensor)
        assert tensor == 3


@skip_distributed_unavailable
@pytest.mark.flaky(reruns=5)
@RunIf(skip_windows=True)  # unhandled timeouts
@pytest.mark.xfail(raises=TimeoutError, strict=False)
def test_two_groups():
    collective_launch(_test_two_groups, [torch.device("cpu")] * 3, num_groups=2)


def _test_default_process_group(strategy, *collectives):
    for collective in collectives:
        assert collective.group == torch.distributed.group.WORLD
    world_size = strategy.world_size
    for c in collectives:
        tensor = torch.tensor(world_size)
        r = c.all_reduce(tensor)
        assert world_size**2 == r


@skip_distributed_unavailable
@pytest.mark.flaky(reruns=5)
@RunIf(skip_windows=True)  # unhandled timeouts
def test_default_process_group():
    collective_launch(_test_default_process_group, [torch.device("cpu")] * 3, num_groups=2)


@skip_distributed_unavailable
@mock.patch.dict(os.environ, {}, clear=True)
def test_collective_manages_default_group():
    collective = TorchCollective()
    with mock.patch("torch.distributed.init_process_group"):
        collective.setup(main_address="foo", main_port="123")

    assert TorchCollective.manages_default_group

    with mock.patch.object(collective, "_group") as mock_group, mock.patch.dict(
        "torch.distributed.distributed_c10d._pg_map", {mock_group: ("", None)}
    ), mock.patch("torch.distributed.destroy_process_group") as destroy_mock:
        collective.teardown()
    destroy_mock.assert_called_once_with(mock_group)

    assert not TorchCollective.manages_default_group
