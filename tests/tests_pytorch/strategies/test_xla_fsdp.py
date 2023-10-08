# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Optional
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from lightning.fabric.accelerators.xla import _using_pjrt
from lightning.fabric.strategies.xla_fsdp import _activation_checkpointing_auto_wrapper
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins.precision import XLAPrecisionPlugin
from lightning.pytorch.strategies import XLAFSDPStrategy

from tests_pytorch.helpers.runif import RunIf


class BoringModelTPU(BoringModel):
    def on_train_start(self) -> None:
        index = 0 if _using_pjrt() else 1
        # assert strategy attributes for device setting
        assert self.device == torch.device("xla", index=index)
        assert os.environ.get("PT_XLA_DEBUG") == "1"


class TestFSDPModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer: Optional[torch.nn.Module] = None

    def _init_model(self) -> None:
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def configure_model(self) -> None:
        if self.layer is None:
            self._init_model()
        # the model is already wrapped with FSDP: no need to wrap again!
        from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel

        if isinstance(self.layer, XlaFullyShardedDataParallel):
            return
        for i, layer in enumerate(self.layer):
            if i % 2 == 0:
                self.layer[i] = XlaFullyShardedDataParallel(layer)
        self.layer = XlaFullyShardedDataParallel(self.layer)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.layer.parameters(), lr=0.1)

    def on_train_batch_start(self, batch, batch_idx):
        assert batch.dtype == torch.float32


def _run_multiple_stages(trainer, model, model_path: Optional[str] = None):
    trainer.fit(model)

    model_path = trainer.strategy.broadcast(model_path)
    model_path = model_path if model_path else trainer.checkpoint_callback.last_model_path

    trainer.save_checkpoint(model_path, weights_only=False)


# TODO (gkroiz): can uncomment after full checkpointing works
# _assert_save_equality(trainer, model_path, cls=model.__class__)

# with torch.inference_mode():
#     # Test entry point
#     trainer.test(model)  # model is wrapped, will not call `configure_model`

#     # provide model path, will create a new unwrapped model and load and then call `configure_shared_model` to wrap
#     trainer.test(ckpt_path=model_path)

#     # Predict entry point
#     trainer.predict(model)  # model is wrapped, will not call `configure_model`

#     # provide model path, will create a new unwrapped model and load and then call `configure_shared_model` to wrap
#     trainer.predict(ckpt_path=model_path)


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_rank_properties_access(xla_available):
    """Test that the strategy returns the expected values depending on whether we're in the main process or not."""
    strategy = XLAFSDPStrategy()
    strategy.cluster_environment = Mock()

    # we're in the main process, no processes have been launched yet
    assert not strategy._launched
    assert strategy.global_rank == 0
    assert strategy.local_rank == 0
    assert strategy.node_rank == 0
    assert strategy.world_size == 1

    # simulate we're in a worker process
    strategy._launched = True
    assert strategy.global_rank == strategy.cluster_environment.global_rank()
    assert strategy.local_rank == strategy.cluster_environment.local_rank()
    assert strategy.node_rank == strategy.cluster_environment.node_rank()
    assert strategy.world_size == strategy.cluster_environment.world_size()


@RunIf(min_torch="2.0", tpu=True, standalone=True)
def test_xla_fsdp_strategy_full_checkpoint(tmpdir):
    """Test to ensure that checkpoint is saved correctly when using TPUs, and all stages can be run."""
    model = TestFSDPModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="tpu",
        # TODO (gkroiz): switch to full when full checkpointing works
        strategy=XLAFSDPStrategy(state_dict_type="sharded"),
        precision="bf16-true",
        max_epochs=1,
    )
    _run_multiple_stages(trainer, model, tmpdir + "last_checkpoint")


@RunIf(min_torch="2.0", tpu=True)
def test_xla_fsdp_activation_checkpointing_setup():
    """Test XLAFSDP activation checkpointing setup."""
    from torch_xla.distributed.fsdp import checkpoint_module
    from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel

    auto_wrapper_callable = lambda m, *args, **kwargs: XlaFullyShardedDataParallel(
        checkpoint_module(m), *args, **kwargs
    )

    strategy = XLAFSDPStrategy(auto_wrapper_callable=auto_wrapper_callable)

    assert auto_wrapper_callable in strategy._fsdp_kwargs.values()


def test_xla_fsdp_policy(xla_available):
    strategy = XLAFSDPStrategy(foo=1)
    assert strategy._fsdp_kwargs == {"foo": 1}

    strategy = XLAFSDPStrategy(auto_wrap_policy={torch.nn.Linear})
    kwargs = strategy._parse_fsdp_kwargs()

    assert kwargs["auto_wrap_policy"].func._mock_name == "transformer_auto_wrap_policy"

    strategy = XLAFSDPStrategy(activation_checkpointing_policy={torch.nn.Linear})
    kwargs = strategy._parse_fsdp_kwargs()
    kwargs = strategy._parse_fsdp_kwargs()  # ensure it's idempotent
    assert set(kwargs) == {"auto_wrapper_callable"}
    assert kwargs["auto_wrapper_callable"].func is _activation_checkpointing_auto_wrapper

    strategy = XLAFSDPStrategy(
        accelerator=Mock(),
        auto_wrap_policy={torch.nn.Linear},
        activation_checkpointing_policy={torch.nn.Linear},
        precision_plugin=XLAPrecisionPlugin("bf16-true"),
    )
    kwargs = strategy._parse_fsdp_kwargs()
    assert set(kwargs) == {"auto_wrap_policy", "auto_wrapper_callable", "compute_dtype"}
    assert kwargs["auto_wrap_policy"].func._mock_name == "transformer_auto_wrap_policy"
    assert kwargs["auto_wrapper_callable"].func is _activation_checkpointing_auto_wrapper
    assert kwargs["compute_dtype"] is torch.bfloat16
    strategy.teardown()

    strategy = XLAFSDPStrategy(activation_checkpointing_policy={torch.nn.Linear}, auto_wrapper_callable="foo")
    with pytest.raises(ValueError, match="cannot set both"):
        strategy._parse_fsdp_kwargs()

    strategy = XLAFSDPStrategy(activation_checkpointing_policy="foo")
    with pytest.raises(TypeError, match="must be a set"):
        strategy._parse_fsdp_kwargs()
