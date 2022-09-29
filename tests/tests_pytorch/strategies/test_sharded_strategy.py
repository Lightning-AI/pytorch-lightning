import os
from unittest import mock
from unittest.mock import Mock

import pytest
import torch

from lightning_lite.strategies.fairscale import _FAIRSCALE_AVAILABLE
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
from pytorch_lightning.strategies import DDPShardedStrategy, DDPSpawnShardedStrategy
from pytorch_lightning.trainer.states import TrainerFn
from tests_pytorch.helpers.runif import RunIf

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS


@pytest.mark.parametrize("clip_val", [0, 10])
@RunIf(min_cuda_gpus=1, fairscale=True)
@mock.patch("fairscale.optim.oss.OSS.clip_grad_norm")
def test_ddp_sharded_precision_16_clip_gradients(mock_oss_clip_grad_norm, clip_val, tmpdir):
    """Ensure that clip gradients is only called if the value is greater than 0."""
    model = BoringModel()
    trainer = Trainer(
        strategy="ddp_sharded",
        accelerator="gpu",
        devices=1,
        precision=16,
        fast_dev_run=True,
        gradient_clip_val=clip_val,
    )
    trainer.fit(model)
    if clip_val > 0:
        mock_oss_clip_grad_norm.assert_called()
    else:
        mock_oss_clip_grad_norm.assert_not_called()


@RunIf(fairscale=True)
@pytest.mark.parametrize(
    "strategy,expected", [("ddp_sharded", DDPShardedStrategy), ("ddp_sharded_spawn", DDPSpawnShardedStrategy)]
)
def test_sharded_ddp_choice(strategy, expected):
    """Test to ensure that strategy is correctly chosen."""
    trainer = Trainer(fast_dev_run=True, strategy=strategy)
    assert isinstance(trainer.strategy, expected)


@RunIf(min_cuda_gpus=1, fairscale=True)
@pytest.mark.parametrize(
    "strategy,expected", [("ddp_sharded", DDPShardedStrategy), ("ddp_sharded_spawn", DDPSpawnShardedStrategy)]
)
def test_ddp_choice_sharded_amp(strategy, expected):
    """Test to ensure that plugin native amp plugin is correctly chosen when using sharded."""
    trainer = Trainer(fast_dev_run=True, accelerator="gpu", devices=1, precision=16, strategy=strategy)
    assert isinstance(trainer.strategy, expected)
    assert isinstance(trainer.precision_plugin, NativeMixedPrecisionPlugin)


@RunIf(fairscale=True)
def test_ddp_sharded_strategy_checkpoint_cpu(tmpdir):
    """Test to ensure that checkpoint is saved correctly."""
    model = BoringModel()
    trainer = Trainer(strategy="ddp_sharded_spawn", accelerator="cpu", devices=2, fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for trained_param, loaded_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(trained_param.to("cpu"), loaded_param)


@RunIf(min_cuda_gpus=2, fairscale=True)
def test_ddp_sharded_strategy_checkpoint_multi_gpu(tmpdir):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs."""
    model = BoringModel()
    trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp_sharded_spawn", fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for trained_param, loaded_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(trained_param.to("cpu"), loaded_param)


@RunIf(min_cuda_gpus=2, fairscale=True)
def test_ddp_sharded_strategy_finetune(tmpdir):
    """Test to ensure that we can save and restart training (simulate fine-tuning)"""
    model = BoringModel()
    trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp_sharded_spawn", fast_dev_run=True)
    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(saved_model)


@RunIf(fairscale=True)
def test_ddp_sharded_strategy_fit_ckpt_path(tmpdir):
    """Test to ensure that resuming from checkpoint works."""
    model = BoringModel()
    trainer = Trainer(strategy="ddp_sharded_spawn", accelerator="cpu", devices=2, fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(strategy="ddp_sharded_spawn", accelerator="cpu", devices=2, fast_dev_run=True)

    trainer.fit(model, ckpt_path=checkpoint_path)


@RunIf(min_cuda_gpus=1, fairscale=True)
def test_ddp_sharded_strategy_fit_ckpt_path_gpu_to_cpu(tmpdir):
    """Test to ensure that resuming from checkpoint works when going from GPUs- > CPU."""
    model = BoringModel()
    trainer = Trainer(strategy="ddp_sharded_spawn", accelerator="gpu", devices=1, fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(strategy="ddp_sharded_spawn", accelerator="cpu", devices=2, fast_dev_run=True)

    trainer.fit(model, ckpt_path=checkpoint_path)


@RunIf(standalone=True, fairscale=True)
@pytest.mark.parametrize(
    "trainer_kwargs",
    (
        dict(accelerator="cpu", devices=2),
        pytest.param(dict(accelerator="gpu", devices=2), marks=RunIf(min_cuda_gpus=2)),
    ),
)
def test_ddp_sharded_strategy_test_multigpu(trainer_kwargs):
    """Test to ensure we can use validate and test without fit."""
    model = BoringModel()
    trainer = Trainer(
        strategy="ddp_sharded_spawn",
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        **trainer_kwargs,
    )

    trainer.validate(model)
    trainer.test(model)


class ManualBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        output = self(batch)
        loss = self.loss(batch, output)
        self.manual_backward(loss)
        opt.step()
        return {"loss": loss}


@RunIf(min_cuda_gpus=2, standalone=True, fairscale=True)
@pytest.mark.parametrize("strategy", ("ddp_sharded", "ddp_sharded_spawn"))
def test_ddp_sharded_strategy_manual_optimization(tmpdir, strategy):
    model = ManualBoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        strategy=strategy,
        fast_dev_run=2,
        accelerator="gpu",
        devices=2,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


class BoringModelSharded(BoringModel):
    def on_train_start(self) -> None:
        """Check if trainer module is wrapped as ShardedDataParallel during training stage."""
        assert isinstance(self.trainer.model, ShardedDataParallel)

    def on_test_start(self) -> None:
        """Check if trainer module remains as LightningModule during test stage."""
        assert isinstance(self.trainer.model, LightningModule)

    def on_validation_start(self) -> None:
        """Check if trainer module remains as LightningModule during test stage."""
        if self.trainer.state.fn == TrainerFn.FITTING:
            assert isinstance(self.trainer.model, ShardedDataParallel)
        else:
            assert isinstance(self.trainer.model, LightningModule)

    def on_predict_start(self) -> None:
        """Check if trainer module remains as LightningModule during prediction stage."""
        assert isinstance(self.trainer.model, LightningModule)


@RunIf(fairscale=True)
def test_configure_ddp(tmpdir):
    """Tests with ddp sharded strategy."""
    trainer = Trainer(default_root_dir=tmpdir, strategy="ddp_sharded", fast_dev_run=True)

    model = BoringModelSharded()

    trainer.fit(model)
    trainer.test(model, dataloaders=model.test_dataloader())
    trainer.validate(model, dataloaders=model.val_dataloader())
    trainer.predict(model, dataloaders=model.predict_dataloader())


@RunIf(fairscale=True)
@mock.patch("pytorch_lightning.strategies.DDPShardedStrategy._wrap_optimizers", autospec=True)
@pytest.mark.parametrize("cls", [DDPShardedStrategy, DDPSpawnShardedStrategy])
def test_custom_kwargs_sharded(_, cls):
    """Tests to ensure that if custom kwargs are passed, they are set correctly."""
    strategy = cls(reduce_fp16=True)
    strategy._lightning_module = Mock(spec=LightningModule)
    strategy._lightning_module.trainer = Mock()
    strategy.parallel_devices = [Mock()]
    class_name = "sharded" if isinstance(strategy, DDPShardedStrategy) else "sharded_spawn"

    with mock.patch(f"pytorch_lightning.strategies.{class_name}.ShardedDataParallel", autospec=True) as mock_sharded:
        strategy.configure_ddp()
    args, kwargs = mock_sharded.call_args
    assert "reduce_fp16" in kwargs
    assert kwargs["reduce_fp16"]


@RunIf(fairscale=True)
@mock.patch("pytorch_lightning.strategies.DDPShardedStrategy._wrap_optimizers", autospec=True)
@pytest.mark.parametrize(["params", "expected_buffer_size"], [(dict(), 0), (dict(reduce_buffer_size=128), 128)])
@pytest.mark.parametrize("num_nodes", [1, 2])
def test_custom_kwargs_sharded_reduce_buffer_size(_, params, expected_buffer_size, num_nodes):
    """Tests to ensure that ``reduce_buffer_size`` is correctly set based on user kwargs."""
    strategy = DDPShardedStrategy(**params)
    strategy.num_nodes = num_nodes
    strategy._lightning_module = Mock(spec=LightningModule)
    strategy._lightning_module.trainer = Mock()
    strategy.parallel_devices = [Mock()]

    with mock.patch("pytorch_lightning.strategies.sharded.ShardedDataParallel", autospec=True) as mock_sharded:
        strategy.configure_ddp()
    args, kwargs = mock_sharded.call_args
    assert "reduce_buffer_size" in kwargs

    if num_nodes > 1 and len(params) == 0:
        # If user has not specified a buffer size and we're using multiple nodes, check to see if default is set
        assert kwargs["reduce_buffer_size"] == DDPShardedStrategy._REDUCE_BUFFER_SIZE_DEFAULT
    else:
        assert kwargs["reduce_buffer_size"] == expected_buffer_size


@RunIf(fairscale=True)
def test_block_backward_sync():
    strategy = DDPShardedStrategy()
    model = mock.MagicMock(spec=ShardedDataParallel)
    with mock.patch.object(strategy, "_model", model):
        with strategy.block_backward_sync():
            pass
    model.no_sync.assert_called_once()


@pytest.mark.parametrize(
    "strategy_name,expected_ddp_kwargs",
    [
        ("ddp_sharded", {}),
        ("ddp_sharded_find_unused_parameters_false", {"find_unused_parameters": False}),
        ("ddp_sharded_spawn", {}),
        ("ddp_sharded_spawn_find_unused_parameters_false", {"find_unused_parameters": False}),
    ],
)
def test_ddp_kwargs_from_registry(strategy_name, expected_ddp_kwargs):
    trainer = Trainer(strategy=strategy_name)
    assert trainer.strategy._ddp_kwargs == expected_ddp_kwargs


class BoringFairScaleOptimizerModel(BoringModel):
    def configure_optimizers(self):
        base_optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return OSS(params=base_optimizer.param_groups, optim=type(base_optimizer), **base_optimizer.defaults)


@RunIf(min_cuda_gpus=2, fairscale=True)
@pytest.mark.parametrize("strategy", (pytest.param("ddp_sharded", marks=RunIf(standalone=True)), "ddp_sharded_spawn"))
def test_ddp_sharded_strategy_checkpoint_multi_gpu_fairscale_optimizer(tmpdir, strategy):
    """Test to ensure that checkpoint is saved correctly when using fairscale optimizers."""
    model = BoringFairScaleOptimizerModel()
    trainer = Trainer(accelerator="gpu", devices=2, strategy=strategy, max_steps=1)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for trained_param, loaded_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(trained_param.to("cpu"), loaded_param)
