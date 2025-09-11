import os
from copy import deepcopy
from pathlib import Path
from re import escape
from typing import Optional
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from lightning.fabric.utilities.init import _has_all_dtensor_params_or_buffers
from lightning.fabric.utilities.load import _load_distributed_checkpoint
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import FSDP2Strategy
from lightning.pytorch.utilities.consolidate_checkpoint import _format_checkpoint
from tests_pytorch.helpers.runif import RunIf


# Minimal boring model for FSDP2 tests (used for DDP/FSDP2 checkpoint compatibility)
class TestBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.1)


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
        # FSDP2 injects an internal `_fsdp_state` attribute and replaces all parameters/buffers with DTensors.
        assert hasattr(self.layer, "_fsdp_state")
        assert _has_all_dtensor_params_or_buffers(self.layer)


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


@RunIf(min_torch="2.6.0")
@pytest.mark.parametrize("strategy", ["fsdp2", "fsdp2_cpu_offload"])
def test_invalid_on_cpu(tmp_path, cuda_count_0, strategy):
    """Test to ensure that we raise Misconfiguration for FSDP on CPU."""
    with pytest.raises(ValueError, match="The strategy `fsdp2` requires a GPU accelerator"):
        trainer = Trainer(accelerator="cpu", default_root_dir=tmp_path, fast_dev_run=True, strategy=strategy)
        assert isinstance(trainer.strategy, FSDP2Strategy)
        trainer.strategy.setup_environment()


@RunIf(min_torch="2.6.0")
def test_custom_mixed_precision():
    """Test to ensure that passing a custom mixed precision config works."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    # custom mp policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float16, output_dtype=torch.float16, cast_forward_inputs=True
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
@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="2.6.0")
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
@RunIf(min_cuda_gpus=1, skip_windows=True, min_torch="2.6.0")
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
@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="2.6.0")
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


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="2.6.0")
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


@RunIf(min_torch="2.6.0")
def test_save_checkpoint_storage_options(tmp_path):
    """Test that the FSDP strategy does not accept storage options for saving checkpoints."""
    strategy = FSDP2Strategy()
    with pytest.raises(TypeError, match=escape("FSDP2Strategy.save_checkpoint(..., storage_options=...)` is not")):
        strategy.save_checkpoint(filepath=tmp_path, checkpoint=Mock(), storage_options=Mock())


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
@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.6.0")
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


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="2.6.0")
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
@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.6.0")
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


@RunIf(max_torch="2.5", min_cuda_gpus=1)
@pytest.mark.parametrize("strategy", ["fsdp2", "fsdp2_cpu_offload"])
def test_fsdp2_requires_torch_2_6_or_newer(tmp_path, strategy):
    """FSDP2 strategies should error on torch < 2.6."""
    with pytest.raises(ValueError, match="FSDP2Strategy requires torch>=2.6.0."):
        Trainer(default_root_dir=tmp_path, fast_dev_run=True, strategy=strategy)
