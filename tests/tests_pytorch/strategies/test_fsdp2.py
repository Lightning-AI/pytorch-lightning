import os
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from functools import partial
from pathlib import Path
from re import escape
from typing import Optional
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
from torchmetrics import Accuracy

from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.fabric.utilities.init import _has_all_dtensor_params_or_buffers
from lightning.fabric.utilities.load import _load_distributed_checkpoint
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins import HalfPrecision
from lightning.pytorch.strategies import FSDP2Strategy, FSDPStrategy
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.consolidate_checkpoint import _format_checkpoint
from tests_pytorch.helpers.runif import RunIf


class TestFSDP2Model(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer: Optional[nn.Module] = None

    def _init_model(self) -> None:
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def configure_optimizers(self):
        # There is some issue with SGD optimizer state in FSDP
        return torch.optim.AdamW(self.layer.parameters(), lr=0.1)

    def on_train_batch_start(self, batch, batch_idx):
        assert batch.dtype == torch.float32

    def on_train_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp2_instance()

    def on_test_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp2_instance()

    def on_validation_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp2_instance()

    def on_predict_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp2_instance()

    def _assert_layer_fsdp2_instance(self):
        # FSDP2 does not wrap modules in a distinct class (like FSDP1’s FullyShardedDataParallel).
        # Instead, it injects an internal `_fsdp_state` attribute and replaces all parameters/buffers
        # with DTensors in place. These two checks together confirm that `self.layer` is FSDP2-wrapped.
        assert hasattr(self.layer, "_fsdp_state")
        assert _has_all_dtensor_params_or_buffers(self.layer)


class TestBoringModel(BoringModel):
    def __init__(self):
        super().__init__(self)

        self.save_hyperparameters()
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def configure_optimizers(self):
        # SGD's FSDP optimizer, state is fixed in https://github.com/pytorch/pytorch/pull/99214
        return torch.optim.AdamW(self.parameters(), lr=0.1)


class TestFSDP2ModelAutoWrapped(TestBoringModel):
    def on_train_batch_start(self, batch, batch_idx):
        assert batch.dtype == torch.float32

    def on_train_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp2_instance()

    def on_test_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp2_instance()

    def on_validation_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp2_instance()

    def on_predict_batch_end(self, _, batch, batch_idx):
        assert batch.dtype == torch.float32
        self._assert_layer_fsdp2_instance()

    def _assert_layer_fsdp2_instance(self):
        # FSDP2 does not wrap modules in a distinct class (like FSDP1’s FullyShardedDataParallel).
        # Instead, it injects an internal `_fsdp_state` attribute and replaces all parameters/buffers
        # with DTensors in place. These two checks together confirm that `self.layer` is FSDP2-wrapped.
        assert hasattr(self.layer, "_fsdp_state")
        assert _has_all_dtensor_params_or_buffers(self.layer)


def _run_multiple_stages(trainer, model, model_path: Optional[str] = None):
    trainer.fit(model)
    trainer.test(model)

    model_path = trainer.strategy.broadcast(model_path)
    model_path = Path(model_path if model_path else trainer.checkpoint_callback.last_model_path)

    # Save another checkpoint after testing, without optimizer states
    trainer.save_checkpoint(model_path.with_name("after-test"))
    trainer.save_checkpoint(model_path, weights_only=True)

    if not model_path.is_dir():  # TODO (@awaelchli): Add support for asserting equality of sharded checkpoints
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


def _assert_save_equality(trainer, ckpt_path, cls=TestFSDP2Model):
    # Use FullySharded to get the state dict for the sake of comparison
    model_state_dict = trainer.strategy.lightning_module_state_dict()

    if trainer.is_global_zero:
        saved_model = cls.load_from_checkpoint(ckpt_path)

        # Assert model parameters are identical after loading
        for ddp_param, shard_param in zip(model_state_dict.values(), saved_model.state_dict().values()):
            assert torch.equal(ddp_param, shard_param)


@pytest.mark.parametrize("strategy", ["fsdp2", "fsdp2_cpu_offload"])
def test_invalid_on_cpu(tmp_path, cuda_count_0, strategy):
    """Test to ensure that we raise Misconfiguration for FSDP on CPU."""
    with pytest.raises(ValueError, match="The strategy `fsdp2` requires a GPU accelerator"):
        trainer = Trainer(accelerator="cpu", default_root_dir=tmp_path, fast_dev_run=True, strategy=strategy)
        assert isinstance(trainer.strategy, FSDP2Strategy)
        trainer.strategy.setup_environment()


def test_custom_mixed_precision():
    """Test to ensure that passing a custom mixed precision config works."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    # custom mp policy
    mp_policy = MixedPrecisionPolicy(
        compute_dtype=torch.float16, param_dtype=torch.bfloat16, buffer_dtype=torch.float16, cast_forward_inputs=True
    )
    strategy = FSDP2Strategy(mp_policy=mp_policy)
    assert strategy.mp_policy == mp_policy

    # default mp policy
    strategy = FSDP2Strategy(mp_policy=None)
    assert isinstance(strategy.mp_policy, MixedPrecisionPolicy)
    assert strategy.mp_policy.param_dtype is None
    assert strategy.mp_policy.reduce_dtype is None
    assert strategy.mp_policy.output_dtype is None
    assert strategy.mp_policy.cast_forward_inputs is True

    # invalid mp policy
    class InvalidMPPolicy:
        pass

    with pytest.raises(TypeError, match="`mp_policy` should be of type `MixedPrecisionPolicy`"):
        FSDP2Strategy(mp_policy=InvalidMPPolicy())


@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_strategy_sync_batchnorm(tmp_path):
    """Test to ensure that sync_batchnorm works when using FSDP and GPU, and all stages can be run."""
    model = TestFSDP2Model()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=2,
        strategy="fsdp2",
        precision="16-true",
        max_epochs=1,
        sync_batchnorm=True,
    )
    _run_multiple_stages(trainer, model, os.path.join(tmp_path, "last.ckpt"))


@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=1, skip_windows=True)
def test_modules_without_parameters(tmp_path):
    """Test that TorchMetrics get moved to the device despite not having any parameters."""

    class MetricsModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric = Accuracy("multiclass", num_classes=10)
            assert self.metric.device == self.metric.tp.device == torch.device("cpu")

        def setup(self, stage) -> None:
            assert self.metric.device == self.metric.tp.device == torch.device("cpu")

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            assert self.metric.device == self.metric.tp.device == torch.device("cuda", 0)
            self.metric(torch.rand(2, 10, device=self.device), torch.randint(0, 10, size=(2,), device=self.device))
            return loss

    model = MetricsModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cuda",
        devices=1,
        strategy="fsdp2",
        max_steps=1,
    )
    trainer.fit(model)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize("precision", ["16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))])
def test_strategy_checkpoint(state_dict_type, precision, tmp_path):
    """Test to ensure that checkpoint is saved correctly when using a single GPU, and all stages can be run."""
    model = TestFSDP2Model()
    strategy = FSDP2Strategy()
    trainer = Trainer(
        default_root_dir=tmp_path, accelerator="gpu", devices=2, strategy=strategy, precision=precision, max_epochs=1
    )
    _run_multiple_stages(trainer, model, os.path.join(tmp_path, "last.ckpt"))


def custom_auto_wrap_policy(
    module,
    recurse,
    nonwrapped_numel: int,
) -> bool:
    return nonwrapped_numel >= 2


@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize(
    ("model", "strategy", "strategy_cfg"),
    [
        pytest.param(TestFSDP2Model(), "fsdp", None, id="manually_wrapped"),
        pytest.param(
            TestFSDP2ModelAutoWrapped(),
            FSDPStrategy,
            {"auto_wrap_policy": custom_auto_wrap_policy},
            id="autowrap_2x",
        ),
        pytest.param(
            TestFSDP2ModelAutoWrapped(),
            FSDPStrategy,
            {
                "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
                "use_orig_params": True,
            },
            id="autowrap_use_orig_params",
        ),
    ],
)
def test_checkpoint_multi_gpus(tmp_path, model, strategy, strategy_cfg):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs, and all stages can be run."""
    ck = ModelCheckpoint(save_last=True)

    strategy_cfg = strategy_cfg or {}
    if not isinstance(strategy, str):
        strategy = strategy(**strategy_cfg)

    trainer = Trainer(
        default_root_dir=tmp_path,
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


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True)
@pytest.mark.parametrize("use_orig_params", [None, False, True])
def test_invalid_parameters_in_optimizer(use_orig_params):
    fsdp_kwargs = {}
    if use_orig_params is not None:
        fsdp_kwargs = {"use_orig_params": use_orig_params}

    trainer = Trainer(
        strategy=FSDPStrategy(**fsdp_kwargs),
        accelerator="cuda",
        devices=1,
        fast_dev_run=1,
    )

    class EmptyParametersModel(BoringModel):
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-2)

    model = EmptyParametersModel()
    trainer.fit(model)

    class NoFlatParametersModel(BoringModel):
        def configure_optimizers(self):
            layer = torch.nn.Linear(4, 5)
            return torch.optim.Adam(layer.parameters(), lr=1e-2)

    error_context = (
        nullcontext()
        if use_orig_params is not False
        else pytest.raises(ValueError, match="The optimizer does not seem to reference any FSDP parameters")
    )

    model = NoFlatParametersModel()
    with error_context:
        trainer.fit(model)


def test_forbidden_precision_raises():
    with pytest.raises(TypeError, match="can only work with the `FSDPPrecision"):
        FSDPStrategy(precision_plugin=HalfPrecision())

    strategy = FSDPStrategy()
    with pytest.raises(TypeError, match="can only work with the `FSDPPrecision"):
        strategy.precision_plugin = HalfPrecision()


def test_activation_checkpointing():
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

    strategy = FSDPStrategy(activation_checkpointing_policy={Block1})
    assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
    assert isinstance(strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)

    strategy = FSDPStrategy(activation_checkpointing_policy=ModuleWrapPolicy({Block1, Block2}))
    assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
    assert isinstance(strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)

    model = Model()
    strategy._parallel_devices = [torch.device("cuda", 0)]
    strategy._lightning_module = model
    strategy._process_group = Mock()
    with (
        mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", new=MagicMock),
        mock.patch(
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
        ) as apply_mock,
    ):
        wrapped = strategy._setup_model(model)
    apply_mock.assert_called_with(wrapped, checkpoint_wrapper_fn=ANY, **strategy._activation_checkpointing_kwargs)


def test_strategy_cpu_offload():
    """Test the different ways cpu offloading can be enabled."""
    # bool
    strategy = FSDPStrategy(cpu_offload=True)
    assert strategy.cpu_offload == CPUOffload(offload_params=True)

    # dataclass
    config = CPUOffload()
    strategy = FSDPStrategy(cpu_offload=config)
    assert strategy.cpu_offload == config


def test_sharding_strategy():
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


@pytest.mark.parametrize("sharding_strategy", ["HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"])
def test_hybrid_shard_configuration(sharding_strategy, monkeypatch):
    """Test that the hybrid sharding strategies can only be used with automatic wrapping or a manually specified pg."""
    with pytest.raises(RuntimeError, match="The hybrid sharding strategy requires you to pass at least one of"):
        FSDPStrategy(sharding_strategy=sharding_strategy)

    strategy = FSDPStrategy(auto_wrap_policy={nn.Linear}, sharding_strategy=sharding_strategy)
    assert strategy.sharding_strategy.name == sharding_strategy

    process_group = (Mock(), Mock())
    strategy = FSDPStrategy(sharding_strategy=sharding_strategy, process_group=process_group)
    assert strategy.sharding_strategy.name == sharding_strategy
    assert strategy.kwargs["process_group"] is process_group

    monkeypatch.setattr("lightning.pytorch.strategies.fsdp._TORCH_GREATER_EQUAL_2_2", False)
    with pytest.raises(ValueError, match="`device_mesh` argument is only supported in torch >= 2.2."):
        FSDPStrategy(device_mesh=Mock())

    monkeypatch.setattr("lightning.pytorch.strategies.fsdp._TORCH_GREATER_EQUAL_2_2", True)
    device_mesh = Mock()
    strategy = FSDPStrategy(sharding_strategy=sharding_strategy, device_mesh=device_mesh)
    assert strategy.sharding_strategy.name == sharding_strategy
    assert strategy.kwargs["device_mesh"] is device_mesh

    with pytest.raises(ValueError, match="process_group.* device_mesh=.* are mutually exclusive"):
        FSDPStrategy(sharding_strategy=sharding_strategy, process_group=process_group, device_mesh=device_mesh)


def test_use_orig_params():
    """Test that Lightning enables `use_orig_params` automatically."""
    strategy = FSDPStrategy()
    assert strategy.kwargs["use_orig_params"]
    strategy = FSDPStrategy(use_orig_params=False)
    assert not strategy.kwargs["use_orig_params"]


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
    kwargs = {}
    if _TORCH_GREATER_EQUAL_2_3:
        kwargs["device_id"] = strategy.root_device if strategy.root_device.type != "cpu" else None
    init_process_group_mock.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta, **kwargs
    )


@mock.patch("lightning.pytorch.strategies.fsdp._load_raw_module_state")
def test_strategy_load_optimizer_states_multiple(_, tmp_path):
    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")], state_dict_type="full")
    trainer = Trainer()
    trainer.state.fn = TrainerFn.FITTING
    strategy._lightning_module = Mock(trainer=trainer)
    spec = torch.optim.Optimizer

    # More states than optimizers configured
    strategy.optimizers = [Mock(spec=spec)]
    checkpoint = {"state_dict": {}, "optimizer_states": [{"state": {}}, {"state": {}}]}
    torch.save(checkpoint, tmp_path / "two-states.ckpt")
    with pytest.raises(RuntimeError, match="1 optimizers but the checkpoint contains 2 optimizers to load"):
        strategy.load_checkpoint(tmp_path / "two-states.ckpt")

    # Fewer states than optimizers configured
    strategy.optimizers = [Mock(spec=spec), Mock(spec=spec)]
    checkpoint = {"state_dict": {}, "optimizer_states": [{"state": {}}]}
    torch.save(checkpoint, tmp_path / "one-state.ckpt")
    with pytest.raises(RuntimeError, match="2 optimizers but the checkpoint contains 1 optimizers to load"):
        strategy.load_checkpoint(tmp_path / "one-state.ckpt")


@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize("wrap_min_params", [2, 1024, 100000000])
def test_strategy_save_optimizer_states(tmp_path, wrap_min_params):
    """Test to ensure that the full state dict and optimizer states is saved when using FSDP strategy.

    Based on `wrap_min_params`, the model will be fully wrapped, half wrapped, and not wrapped at all. If the model can
    be restored to DDP, it means that the optimizer states were saved correctly.

    """
    model = TestFSDP2ModelAutoWrapped(wrap_min_params=wrap_min_params)

    strategy = FSDPStrategy(auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=wrap_min_params))
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        precision="16-mixed",
        max_epochs=1,
        barebones=True,
    )

    trainer.fit(model)
    model_path = os.path.join(tmp_path, "last.ckpt")
    model_path = trainer.strategy.broadcast(model_path)
    trainer.save_checkpoint(model_path)

    model_state_dict = trainer.strategy.lightning_module_state_dict()
    optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    if trainer.global_rank != 0:
        assert len(model_state_dict) == 0

    if trainer.global_rank != 0:
        assert len(optimizer_state_dict) == 0

    # restore model to ddp
    model = TestBoringModel()
    trainer = Trainer(default_root_dir=tmp_path, accelerator="gpu", devices=2, strategy="ddp", max_epochs=1)

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


@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize("wrap_min_params", [2, 1024, 100000000])
def test_strategy_load_optimizer_states(wrap_min_params, tmp_path):
    """Test to ensure that the full state dict and optimizer states can be load when using FSDP strategy.

    Based on `wrap_min_params`, the model will be fully wrapped, half wrapped, and not wrapped at all. If the DDP model
    can be restored to FSDP, it means that the optimizer states were restored correctly.

    """

    # restore model to ddp
    model = TestBoringModel()
    trainer = Trainer(default_root_dir=tmp_path, accelerator="gpu", devices=2, strategy="ddp", max_epochs=1)

    # This step will restore the model and optimizer states
    trainer.fit(model)
    model_path = os.path.join(tmp_path, "last.ckpt")
    model_path = trainer.strategy.broadcast(model_path)
    trainer.save_checkpoint(model_path)

    # Get the model and optimizer states from the restored ddp model
    model_state_dict = trainer.strategy.lightning_module_state_dict()
    optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    # Build a new FSDP model
    model = TestFSDP2ModelAutoWrapped(wrap_min_params=wrap_min_params)

    strategy = FSDPStrategy(auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=wrap_min_params))
    trainer = Trainer(
        default_root_dir=tmp_path,
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

    if trainer.global_rank != 0:
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
def test_configure_model(precision, expected_dtype, tmp_path):
    """Test that the module under configure_model gets moved to the right device and dtype."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cuda",
        devices=2,
        strategy=FSDP2Strategy(),
        precision=precision,
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )

    class MyModel(BoringModel):
        def configure_model(self):
            self.layer = torch.nn.Linear(32, 2)
            # The model is on the CPU until after `.setup()``
            # TODO: Support initialization on meta device
            expected_device = torch.device("cpu")
            assert self.layer.weight.device == expected_device
            assert self.layer.weight.dtype == expected_dtype

        def configure_optimizers(self):
            # There is some issue with SGD optimizer state in FSDP
            return torch.optim.AdamW(self.layer.parameters(), lr=0.1)

        def on_fit_start(self):
            # Parameters get sharded in `.setup()` and moved to the target device
            assert self.layer.weight.device == torch.device("cuda", self.local_rank)
            assert self.layer.weight.dtype == expected_dtype

    model = MyModel()
    trainer.fit(model)


def test_save_checkpoint_storage_options(tmp_path):
    """Test that the FSDP strategy does not accept storage options for saving checkpoints."""
    strategy = FSDP2Strategy()
    with pytest.raises(TypeError, match=escape("FSDP2Strategy.save_checkpoint(..., storage_options=...)` is not")):
        strategy.save_checkpoint(filepath=tmp_path, checkpoint=Mock(), storage_options=Mock())


def test_load_unknown_checkpoint_type(tmp_path):
    """Test that the strategy validates the contents at the checkpoint path."""
    strategy = FSDP2Strategy()
    strategy.model = Mock()
    strategy._lightning_module = Mock()
    path = tmp_path / "empty_dir"  # neither a single file nor a directory with meta file
    path.mkdir()
    with pytest.raises(ValueError, match="does not point to a valid checkpoint"):
        strategy.load_checkpoint(checkpoint_path=path)


class TestFSDP2CheckpointModel(BoringModel):
    def __init__(self, params_to_compare=None):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        self.params_to_compare = params_to_compare

    def configure_optimizers(self):
        # SGD's FSDP optimizer, state is fixed in https://github.com/pytorch/pytorch/pull/99214
        return torch.optim.AdamW(self.parameters(), lr=0.1)

    def on_train_start(self):
        if self.params_to_compare is None:
            return
        for p0, p1 in zip(self.params_to_compare, self.trainer.model.parameters()):
            torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=2, standalone=True)
def test_save_load_sharded_state_dict(tmp_path):
    """Test FSDP saving and loading with the sharded state dict format."""
    strategy = FSDP2Strategy()
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "accelerator": "cuda",
        "devices": 2,
        "max_epochs": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }

    # Initial training
    model = TestFSDP2CheckpointModel()
    trainer = Trainer(**trainer_kwargs, strategy=strategy)
    trainer.fit(model)
    params_before = deepcopy(list(trainer.model.parameters()))

    checkpoint_path = Path(trainer.strategy.broadcast(trainer.checkpoint_callback.best_model_path))
    assert set(os.listdir(checkpoint_path)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    metadata = torch.load(checkpoint_path / "meta.pt", weights_only=True)
    assert "pytorch-lightning_version" in metadata
    assert len(metadata["callbacks"]) == 1  # model checkpoint callback
    assert "state_dict" not in metadata
    assert "optimizer_states" not in metadata

    # Load checkpoint and continue training
    trainer_kwargs.update(max_epochs=2)
    model = TestFSDP2CheckpointModel(params_to_compare=params_before)
    strategy = FSDP2Strategy()
    trainer = Trainer(**trainer_kwargs, strategy=strategy)
    trainer.fit(model, ckpt_path=checkpoint_path)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("16-true", torch.float16),
    ],
)
def test_module_init_context(precision, expected_dtype, tmp_path):
    """Test that the module under the init-context gets moved to the right device and dtype."""

    class Model(BoringModel):
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-2)

        def on_train_start(self):
            # Parameters get sharded in `FSDPStrategy.setup()` and moved to the target device
            assert self.layer.weight.device == torch.device("cuda", self.local_rank)
            assert self.layer.weight.dtype == expected_dtype
            optimizer = self.optimizers(use_pl_optimizer=False)
            assert optimizer.param_groups[0]["params"][0].device.type == "cuda"

    def _run_setup_assertions(empty_init, expected_device):
        trainer = Trainer(
            default_root_dir=tmp_path,
            accelerator="cuda",
            devices=2,
            strategy=FSDP2Strategy(),
            precision=precision,
            max_steps=1,
            barebones=True,
            enable_checkpointing=False,
            logger=False,
        )
        with trainer.init_module(empty_init=empty_init):
            model = Model()

        # The model is on the CPU/meta-device until after `FSDPStrategy.setup()`
        assert model.layer.weight.device == expected_device
        assert model.layer.weight.dtype == expected_dtype
        trainer.fit(model)

    # Case 1: No empty init
    _run_setup_assertions(empty_init=False, expected_device=torch.device("cpu"))

    # Case 2: Empty-init with meta device
    _run_setup_assertions(empty_init=True, expected_device=torch.device("meta"))


@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.3.0")
def test_save_sharded_and_consolidate_and_load(tmp_path):
    """Test the consolidation of a FSDP2-sharded checkpoint into a single file."""

    class CustomModel(BoringModel):
        def configure_optimizers(self):
            # Use Adam instead of SGD for this test because it has state
            # In PyTorch >= 2.4, saving an optimizer with empty state would result in a `KeyError: 'state'`
            # when loading the optimizer state-dict back.
            # TODO: To resolve this, switch to the new `torch.distributed.checkpoint` APIs in FSDPStrategy
            return torch.optim.Adam(self.parameters(), lr=0.1)

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cuda",
        devices=2,
        strategy=FSDP2Strategy(),
        max_steps=3,
    )
    trainer.fit(model)

    checkpoint_path_sharded = trainer.strategy.broadcast(str(trainer.checkpoint_callback.best_model_path))
    assert set(os.listdir(checkpoint_path_sharded)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    # consolidate the checkpoint to a single file
    checkpoint_path_full = trainer.strategy.broadcast(str(tmp_path / "checkpoint_full.ckpt"))
    if trainer.global_rank == 0:
        checkpoint = _load_distributed_checkpoint(Path(checkpoint_path_sharded))
        checkpoint = _format_checkpoint(checkpoint)
        torch.save(checkpoint, checkpoint_path_full)
    trainer.strategy.barrier()

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cuda",
        devices=2,
        strategy="ddp",
        max_steps=4,
    )
    trainer.fit(model, ckpt_path=checkpoint_path_full)
