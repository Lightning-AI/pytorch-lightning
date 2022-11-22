import os
from copy import deepcopy
from functools import partial
from unittest import mock

import pytest
from lightning_utilities.core.imports import module_available
from tests_app.helpers.utils import no_warning_call

import lightning as L
from lightning_app.components.multi_node.trainer import _LightningTrainerRunExecutor


def dummy_callable(**kwargs):
    t = L.pytorch.Trainer(**kwargs)
    return t._all_passed_kwargs


def dummy_init(self, **kwargs):
    self._all_passed_kwargs = kwargs


def _get_args_after_tracer_injection(**kwargs):
    with mock.patch.object(L.pytorch.Trainer, "__init__", dummy_init):
        ret_val = _LightningTrainerRunExecutor.run(
            local_rank=0,
            work_run=partial(dummy_callable, **kwargs),
            main_address="1.2.3.4",
            main_port=5,
            node_rank=6,
            num_nodes=7,
            nprocs=8,
        )
        env_vars = deepcopy(os.environ)
    return ret_val, env_vars


@pytest.mark.skipif(not module_available("lightning.pytorch"), reason="lightning.pytorch not available")
@pytest.mark.skipif(
    not L.pytorch.accelerators.MPSAccelerator.is_available(), reason="MPS not available but required for this test"
)
@pytest.mark.parametrize("accelerator_given,accelerator_expected", [("cpu", "cpu"), ("auto", "cpu"), ("gpu", "cpu")])
def test_trainer_run_executor_mps_forced_cpu(accelerator_given, accelerator_expected):
    warning_str = (
        r"Forcing accelerator=cpu as other accelerators \(specifically MPS\) are not supported "
        + "by PyTorch for distributed training on mps capable devices"
    )
    if accelerator_expected != accelerator_given:
        warning_context = pytest.warns(UserWarning, match=warning_str)
    else:
        warning_context = no_warning_call(match=warning_str + "*")

    with warning_context:
        ret_val, env_vars = _get_args_after_tracer_injection(accelerator=accelerator_given)
    assert ret_val["accelerator"] == accelerator_expected


@pytest.mark.parametrize(
    "args_given,args_expected",
    [
        (
            {
                "devices": 1,
                "num_nodes": 1,
                "accelerator": "gpu",
            },
            {"devices": 8, "num_nodes": 7, "accelerator": "auto"},
        ),
        ({"strategy": "ddp_spawn"}, {"strategy": "ddp"}),
        ({"strategy": "ddp_sharded_spawn"}, {"strategy": "ddp_sharded"}),
    ],
)
@pytest.mark.skipif(not module_available("lightning.pytorch"), reason="lightning.pytorch not available")
def test_trainer_run_executor_arguments_choices(args_given: dict, args_expected: dict):

    # ddp with mps devices not available (tested separately, just patching here for cross-os testing of other args)
    if L.pytorch.accelerators.MPSAccelerator.is_available():
        args_expected["accelerator"] = "cpu"

    ret_val, env_vars = _get_args_after_tracer_injection(**args_given)

    for k, v in args_expected.items():
        assert ret_val[k] == v

    assert env_vars["MASTER_ADDR"] == "1.2.3.4"
    assert env_vars["MASTER_PORT"] == "5"
    assert env_vars["GROUP_RANK"] == "6"
    assert env_vars["RANK"] == str(0 + 6 * 8)
    assert env_vars["LOCAL_RANK"] == "0"
    assert env_vars["WORLD_SIZE"] == str(7 * 8)
    assert env_vars["LOCAL_WORLD_SIZE"] == "8"
    assert env_vars["TORCHELASTIC_RUN_ID"] == "1"


def test_trainer_run_executor_invalid_strategy_instances():
    with pytest.raises(ValueError, match="DDP Spawned strategies aren't supported yet."):
        _, _ = _get_args_after_tracer_injection(strategy=L.pytorch.strategies.DDPSpawnStrategy())

    with pytest.raises(ValueError, match="DDP Spawned strategies aren't supported yet."):
        _, _ = _get_args_after_tracer_injection(strategy=L.pytorch.strategies.DDPSpawnShardedStrategy())
