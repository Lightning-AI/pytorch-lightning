import os
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from typing import Any, Dict, Optional
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
import torch
import torch.nn as nn

from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_1_12,
    _TORCH_GREATER_EQUAL_2_0,
    _TORCH_GREATER_EQUAL_2_1,
)
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins.precision.fsdp import FSDPMixedPrecisionPlugin
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision
    from torch.distributed.fsdp.wrap import always_wrap_policy, size_based_auto_wrap_policy, wrap
else:
    size_based_auto_wrap_policy = lambda *_, **__: False
if _TORCH_GREATER_EQUAL_2_0:
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
else:
    ModuleWrapPolicy = object


class TestFSDPModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer: Optional[nn.Module] = None

    def _init_model(self) -> None:
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def setup(self, stage: str) -> None:
        if self.layer is None:
            self._init_model()

    def configure_model(self) -> None:
        # the model is already wrapped with FSDP: no need to wrap again!
        if isinstance(self.layer, FullyShardedDataParallel):
            return
        for i, layer in enumerate(self.layer):
            if i % 2 == 0:
                self.layer[i] = wrap(layer)
        self.layer = wrap(self.layer)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # when loading full state dict, we first need to create a new unwrapped model
        self._init_model()

    def configure_optimizers(self):
        # There is some issue with SGD optimizer state in FSDP
        return torch.optim.AdamW(self.layer.parameters(), lr=0.1)

    def on_train_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_test_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_validation_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_predict_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        assert isinstance(self.layer, FullyShardedDataParallel)
        assert isinstance(self.trainer.strategy.precision_plugin, FSDPMixedPrecisionPlugin)

        if self.trainer.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif self.trainer.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif self.trainer.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif self.trainer.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision {self.trainer.precision}")

        assert self.layer.mixed_precision.param_dtype == param_dtype
        assert self.layer.mixed_precision.reduce_dtype == reduce_dtype
        assert self.layer.mixed_precision.buffer_dtype == buffer_dtype

        for layer_num in [0, 2]:
            assert isinstance(self.layer.module[layer_num], FullyShardedDataParallel)
            assert self.layer[layer_num].mixed_precision.param_dtype == param_dtype
            assert self.layer[layer_num].mixed_precision.reduce_dtype == reduce_dtype
            assert self.layer[layer_num].mixed_precision.buffer_dtype == buffer_dtype


class TestBoringModel(BoringModel):
    def __init__(self, wrap_min_params: int = 2):
        super().__init__()

        self.save_hyperparameters()
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        self.should_be_wrapped = [(32 * 32 + 32) > wrap_min_params, None, (32 * 2 + 2) > wrap_min_params]

    def configure_optimizers(self):
        parameters = self.parameters() if _TORCH_GREATER_EQUAL_2_0 else self.trainer.model.parameters()

        # SGD's FSDP optimier state is fixed in https://github.com/pytorch/pytorch/pull/99214
        return torch.optim.AdamW(parameters, lr=0.1)


class TestFSDPModelAutoWrapped(TestBoringModel):
    def on_train_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_test_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_validation_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def on_predict_batch_end(self, *_) -> None:
        self._assert_layer_fsdp_instance()

    def _assert_layer_fsdp_instance(self) -> None:
        assert isinstance(self.layer, torch.nn.Sequential)
        assert isinstance(self.trainer.strategy.precision_plugin, FSDPMixedPrecisionPlugin)

        if self.trainer.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif self.trainer.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif self.trainer.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif self.trainer.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision {self.trainer.precision}")

        for layer_num in [0, 2]:
            if not self.should_be_wrapped[layer_num]:
                # this layer is not wrapped
                assert not isinstance(self.layer[layer_num], FullyShardedDataParallel)
                continue
            assert isinstance(self.layer[layer_num], FullyShardedDataParallel)
            assert self.layer[layer_num].mixed_precision.param_dtype == param_dtype
            assert self.layer[layer_num].mixed_precision.reduce_dtype == reduce_dtype
            assert self.layer[layer_num].mixed_precision.buffer_dtype == buffer_dtype


def _run_multiple_stages(trainer, model, model_path: Optional[str] = None):
    trainer.fit(model)
    model_path = trainer.strategy.broadcast(model_path)
    model_path = model_path if model_path else trainer.checkpoint_callback.last_model_path

    trainer.save_checkpoint(model_path, weights_only=True)

    _assert_save_equality(trainer, model_path, cls=model.__class__)

    with torch.inference_mode():
        # Test entry point
        trainer.test(model)  # model is wrapped, will not call `configure_model`

        # provide model path, will create a new unwrapped model and load and then call `configure_shared_model` to wrap
        trainer.test(ckpt_path=model_path)

        # Predict entry point
        trainer.predict(model)  # model is wrapped, will not call `configure_model`

        # provide model path, will create a new unwrapped model and load and then call `configure_shared_model` to wrap
        trainer.predict(ckpt_path=model_path)


def _assert_save_equality(trainer, ckpt_path, cls=TestFSDPModel):
    # Use FullySharded to get the state dict for the sake of comparison
    model_state_dict = trainer.strategy.lightning_module_state_dict()

    if trainer.is_global_zero:
        saved_model = cls.load_from_checkpoint(ckpt_path)

        # Assert model parameters are identical after loading
        for ddp_param, shard_param in zip(model_state_dict.values(), saved_model.state_dict().values()):
            assert torch.equal(ddp_param, shard_param)


@RunIf(min_torch="1.12")
def test_invalid_on_cpu(tmpdir, cuda_count_0):
    """Test to ensure that we raise Misconfiguration for FSDP on CPU."""
    with pytest.raises(
        MisconfigurationException,
        match=f"You selected strategy to be `{FSDPStrategy.strategy_name}`, but GPU accelerator is not used.",
    ):
        trainer = Trainer(accelerator="cpu", default_root_dir=tmpdir, fast_dev_run=True, strategy="fsdp")
        assert isinstance(trainer.strategy, FSDPStrategy)
        trainer.strategy.setup_environment()


@RunIf(min_torch="1.12", min_cuda_gpus=1)
@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("16-mixed", (torch.float32, torch.float16, torch.float16)),
        ("bf16-mixed", (torch.float32, torch.bfloat16, torch.bfloat16)),
        # TODO: add 16-true and bf16-true once supported
    ],
)
def test_precision_plugin_config(precision, expected):
    plugin = FSDPMixedPrecisionPlugin(precision=precision, device="cuda")
    config = plugin.mixed_precision_config

    assert config.param_dtype == expected[0]
    assert config.buffer_dtype == expected[1]
    assert config.reduce_dtype == expected[2]


@RunIf(min_torch="1.12")
def test_fsdp_custom_mixed_precision(tmpdir):
    """Test to ensure that passing a custom mixed precision config works."""
    config = MixedPrecision()
    strategy = FSDPStrategy(mixed_precision=config)
    assert strategy.mixed_precision_config == config


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
def test_fsdp_strategy_sync_batchnorm(tmpdir):
    """Test to ensure that sync_batchnorm works when using FSDP and GPU, and all stages can be run."""
    model = TestFSDPModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy="fsdp",
        precision="16-mixed",
        max_epochs=1,
        sync_batchnorm=True,
    )
    _run_multiple_stages(trainer, model, os.path.join(tmpdir, "last.ckpt"))


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("precision", ["16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))])
def test_fsdp_strategy_checkpoint(tmpdir, precision):
    """Test to ensure that checkpoint is saved correctly when using a single GPU, and all stages can be run."""
    model = TestFSDPModel()
    trainer = Trainer(
        default_root_dir=tmpdir, accelerator="gpu", devices=1, strategy="fsdp", precision=precision, max_epochs=1
    )
    _run_multiple_stages(trainer, model, os.path.join(tmpdir, "last.ckpt"))


if _TORCH_GREATER_EQUAL_2_0:

    def custom_auto_wrap_policy(
        module,
        recurse,
        nonwrapped_numel: int,
    ) -> bool:
        return nonwrapped_numel >= 2

else:

    def custom_auto_wrap_policy(
        module,
        recurse,
        unwrapped_params: int,
    ) -> bool:
        return unwrapped_params >= 2


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("wrap_min_params", [2, 1024, 100000000])
def test_fsdp_strategy_full_state_dict(tmpdir, wrap_min_params):
    """Test to ensure that the full state dict is extracted when using FSDP strategy.

    Based on `wrap_min_params`, the model will be fully wrapped, half wrapped, and not wrapped at all.

    """
    model = TestFSDPModelAutoWrapped(wrap_min_params=wrap_min_params)
    correct_state_dict = model.state_dict()  # State dict before wrapping

    strategy = FSDPStrategy(auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=wrap_min_params))
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        precision="16-mixed",
        max_epochs=1,
        barebones=True,
    )
    trainer.fit(model)

    full_state_dict = trainer.strategy.lightning_module_state_dict()

    if trainer.global_rank != 0:
        assert len(full_state_dict) == 0
        return

    # State dict should contain same number of keys
    assert len(correct_state_dict) == len(full_state_dict)
    # OrderedDict should return the same keys in the same order
    assert all(_ex == _co for _ex, _co in zip(full_state_dict.keys(), correct_state_dict.keys()))


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize(
    ("model", "strategy", "strategy_cfg"),
    [
        pytest.param(TestFSDPModel(), "fsdp", None, id="manually_wrapped"),
        pytest.param(
            TestFSDPModelAutoWrapped(),
            FSDPStrategy,
            {"auto_wrap_policy": custom_auto_wrap_policy},
            marks=RunIf(max_torch="2.0.0"),
            id="autowrap_1x",
        ),
        pytest.param(
            TestFSDPModelAutoWrapped(),
            FSDPStrategy,
            {"auto_wrap_policy": custom_auto_wrap_policy},
            marks=RunIf(min_torch="2.0.0"),
            id="autowrap_2x",
        ),
        pytest.param(
            TestFSDPModelAutoWrapped(),
            FSDPStrategy,
            {
                "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}) if _TORCH_GREATER_EQUAL_2_1 else None,
                "use_orig_params": True,
            },
            marks=RunIf(min_torch="2.1.0"),
            id="autowrap_use_orig_params",
        ),
    ],
)
def test_fsdp_checkpoint_multi_gpus(tmpdir, model, strategy, strategy_cfg):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run."""
    ck = ModelCheckpoint(save_last=True)

    strategy_cfg = strategy_cfg or {}
    if not isinstance(strategy, str):
        strategy = strategy(**strategy_cfg)

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        precision="16-mixed",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        limit_predict_batches=2,
        callbacks=[ck],
    )
    _run_multiple_stages(trainer, model)


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, min_torch="1.12")
def test_invalid_parameters_in_optimizer():
    trainer = Trainer(
        strategy="fsdp",
        accelerator="cuda",
        devices=1,
        fast_dev_run=1,
    )
    error_context = (
        nullcontext()
        if _TORCH_GREATER_EQUAL_2_0
        else pytest.raises(ValueError, match="The optimizer does not seem to reference any FSDP parameters")
    )

    class EmptyParametersModel(BoringModel):
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-2)

    model = EmptyParametersModel()
    with error_context:
        trainer.fit(model)

    class NoFlatParametersModel(BoringModel):
        def configure_optimizers(self):
            layer = torch.nn.Linear(4, 5)
            return torch.optim.Adam(layer.parameters(), lr=1e-2)

    model = NoFlatParametersModel()
    with error_context:
        trainer.fit(model)


@RunIf(min_torch="1.12")
@mock.patch("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_1_13", False)
def test_fsdp_activation_checkpointing_support():
    """Test that we error out if activation checkpointing requires a newer PyTorch version."""
    with pytest.raises(ValueError, match="activation_checkpointing` requires torch >= 1.13.0"):
        FSDPStrategy(activation_checkpointing=Mock())


@RunIf(min_torch="1.13")
def test_fsdp_activation_checkpointing():
    """Test that the FSDP strategy can apply activation checkpointing to the given layers."""

    class Block1(nn.Linear):
        pass

    class Block2(nn.Linear):
        pass

    class Model(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Sequential(Block1(4, 4), Block1(5, 5))
            self.layer1 = Block2(2, 2)
            self.layer2 = nn.Linear(3, 3)

    if _TORCH_GREATER_EQUAL_2_1:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        strategy = FSDPStrategy(activation_checkpointing_policy={Block1})
        assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
        assert isinstance(strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)

        strategy = FSDPStrategy(activation_checkpointing_policy=ModuleWrapPolicy({Block1, Block2}))
        assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
        assert isinstance(strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)
    else:
        strategy = FSDPStrategy(activation_checkpointing=Block1)
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

        strategy = FSDPStrategy(activation_checkpointing=[Block1, Block2])
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

        strategy = FSDPStrategy(activation_checkpointing_policy={Block1})
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

        strategy = FSDPStrategy(activation_checkpointing_policy={Block1, Block2})
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

    model = Model()
    strategy._parallel_devices = [torch.device("cuda", 0)]
    strategy._lightning_module = model
    strategy._process_group = Mock()
    with mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", new=MagicMock), mock.patch(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
    ) as apply_mock:
        wrapped = strategy._setup_model(model)
    apply_mock.assert_called_with(wrapped, checkpoint_wrapper_fn=ANY, **strategy._activation_checkpointing_kwargs)


@RunIf(min_torch="1.12")
def test_fsdp_strategy_cpu_offload():
    """Test the different ways cpu offloading can be enabled."""
    # bool
    strategy = FSDPStrategy(cpu_offload=True)
    assert strategy.cpu_offload == CPUOffload(offload_params=True)

    # dataclass
    config = CPUOffload()
    strategy = FSDPStrategy(cpu_offload=config)
    assert strategy.cpu_offload == config


@RunIf(min_torch="1.12")
def test_fsdp_sharding_strategy():
    """Test the different ways the sharding strategy can be set."""
    from torch.distributed.fsdp import ShardingStrategy

    # default
    strategy = FSDPStrategy()
    assert strategy.sharding_strategy == ShardingStrategy.FULL_SHARD

    # enum
    strategy = FSDPStrategy(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
    assert strategy.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP

    # string
    strategy = FSDPStrategy(sharding_strategy="NO_SHARD")
    assert strategy.sharding_strategy == ShardingStrategy.NO_SHARD
    strategy = FSDPStrategy(sharding_strategy="no_shard")
    assert strategy.sharding_strategy == ShardingStrategy.NO_SHARD


@RunIf(min_torch="1.12")
def test_fsdp_use_orig_params():
    """Test that Lightning enables `use_orig_params` in PyTorch >= 2.0."""
    with mock.patch("lightning.pytorch.strategies.fsdp._TORCH_GREATER_EQUAL_2_0", False):
        strategy = FSDPStrategy()
        assert "use_orig_params" not in strategy.kwargs

    with mock.patch("lightning.pytorch.strategies.fsdp._TORCH_GREATER_EQUAL_2_0", True):
        strategy = FSDPStrategy()
        assert strategy.kwargs["use_orig_params"]
        strategy = FSDPStrategy(use_orig_params=False)
        assert not strategy.kwargs["use_orig_params"]


@RunIf(min_torch="1.12")
@mock.patch("torch.distributed.init_process_group")
def test_set_timeout(init_process_group_mock):
    """Test that the timeout gets passed to the ``torch.distributed.init_process_group`` function."""
    test_timedelta = timedelta(seconds=30)
    strategy = FSDPStrategy(timeout=test_timedelta, parallel_devices=[torch.device("cpu")])
    strategy.cluster_environment = LightningEnvironment()
    strategy.accelerator = Mock()
    strategy.setup_environment()
    process_group_backend = strategy._get_process_group_backend()
    global_rank = strategy.cluster_environment.global_rank()
    world_size = strategy.cluster_environment.world_size()
    init_process_group_mock.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta
    )


@RunIf(min_torch="1.12")
def test_fsdp_strategy_load_optimizer_states_multiple():
    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")])

    # More states than optimizers configured
    strategy.optimizers = [Mock()]
    checkpoint = {"optimizer_states": [Mock(), Mock()]}
    with pytest.raises(RuntimeError, match="1 optimizers but the checkpoint contains 2 optimizers to load"):
        strategy.load_optimizer_state_dict(checkpoint)

    # Fewer states than optimizers configured
    strategy.optimizers = [Mock(), Mock()]
    checkpoint = {"optimizer_states": [Mock()]}
    with pytest.raises(RuntimeError, match="2 optimizers but the checkpoint contains 1 optimizers to load"):
        strategy.load_optimizer_state_dict(checkpoint)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("wrap_min_params", [2, 1024, 100000000])
def test_fsdp_strategy_save_optimizer_states(tmpdir, wrap_min_params):
    """Test to ensure that the full state dict and optimizer states is saved when using FSDP strategy.

    Based on `wrap_min_params`, the model will be fully wrapped, half wrapped, and not wrapped at all. If the model can
    be restored to DDP, it means that the optimizer states were saved correctly.

    """
    model = TestFSDPModelAutoWrapped(wrap_min_params=wrap_min_params)

    strategy = FSDPStrategy(auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=wrap_min_params))
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        precision="16-mixed",
        max_epochs=1,
        barebones=True,
    )

    trainer.fit(model)
    model_path = os.path.join(tmpdir, "last.ckpt")
    model_path = trainer.strategy.broadcast(model_path)
    trainer.save_checkpoint(model_path)

    model_state_dict = trainer.strategy.lightning_module_state_dict()
    optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    if trainer.global_rank != 0:
        assert len(model_state_dict) == 0

        if _TORCH_GREATER_EQUAL_2_1:
            assert len(optimizer_state_dict) == 0

    # restore model to ddp
    model = TestBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, accelerator="gpu", devices=2, strategy="ddp", max_epochs=1)

    # This step will restore the model and optimizer states
    trainer.fit(model, ckpt_path=model_path)

    # Get the model and optimizer states from the restored ddp model
    restored_model_state_dict = trainer.strategy.lightning_module_state_dict()
    restored_optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    if trainer.global_rank == 0:
        # assert everything is the same
        assert len(model_state_dict) == len(restored_model_state_dict)
        assert len(optimizer_state_dict) == len(restored_optimizer_state_dict)

        torch.testing.assert_close(model_state_dict, restored_model_state_dict, atol=0, rtol=0)
        torch.testing.assert_close(optimizer_state_dict, restored_optimizer_state_dict, atol=0, rtol=0)

    trainer.strategy.barrier()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("wrap_min_params", [2, 1024, 100000000])
def test_fsdp_strategy_load_optimizer_states(tmpdir, wrap_min_params):
    """Test to ensure that the full state dict and optimizer states can be load when using FSDP strategy.

    Based on `wrap_min_params`, the model will be fully wrapped, half wrapped, and not wrapped at all. If the DDP model
    can be restored to FSDP, it means that the optimizer states were restored correctly.

    """

    # restore model to ddp
    model = TestBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, accelerator="gpu", devices=2, strategy="ddp", max_epochs=1)

    # This step will restore the model and optimizer states
    trainer.fit(model)
    model_path = os.path.join(tmpdir, "last.ckpt")
    model_path = trainer.strategy.broadcast(model_path)
    trainer.save_checkpoint(model_path)

    # Get the model and optimizer states from the restored ddp model
    model_state_dict = trainer.strategy.lightning_module_state_dict()
    optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    # Build a new FSDP model
    model = TestFSDPModelAutoWrapped(wrap_min_params=wrap_min_params)

    strategy = FSDPStrategy(auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=wrap_min_params))
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        precision="16-mixed",
        max_epochs=1,
        barebones=True,
    )

    trainer.fit(model, ckpt_path=model_path)

    restored_model_state_dict = trainer.strategy.lightning_module_state_dict()
    restored_optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    if trainer.global_rank != 0:
        assert len(restored_model_state_dict) == 0

        if _TORCH_GREATER_EQUAL_2_1:
            assert len(restored_optimizer_state_dict) == 0

    if trainer.global_rank == 0:
        # assert everything is the same
        assert len(model_state_dict) == len(restored_model_state_dict)
        assert len(optimizer_state_dict) == len(restored_optimizer_state_dict)
        torch.testing.assert_close(model_state_dict, restored_model_state_dict, atol=0, rtol=0)
        torch.testing.assert_close(optimizer_state_dict, restored_optimizer_state_dict, atol=0, rtol=0)

    trainer.strategy.barrier()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
    ],
)
def test_configure_model(precision, expected_dtype):
    """Test that the module under configure_model gets moved to the right device and dtype."""
    trainer = Trainer(
        accelerator="cuda",
        devices=2,
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        precision=precision,
        fast_dev_run=1,
    )

    class MyModel(BoringModel):
        def configure_model(self):
            self.layer = torch.nn.Linear(32, 2)
            # The model is on the CPU until after `.setup()``
            # TODO: Support initialization on meta device
            expected_device = torch.device("cpu")
            assert self.layer.weight.device == expected_device
            assert self.layer.weight.dtype == expected_dtype

        def on_fit_start(self):
            # Parameters get sharded in `.setup()` and moved to the target device
            assert self.layer.weight.device == torch.device("cuda", self.local_rank)
            assert self.layer.weight.dtype == expected_dtype

    model = MyModel()
    trainer.fit(model)
