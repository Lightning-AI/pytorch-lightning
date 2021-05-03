import json
import os
from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Optimizer

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.plugins import DeepSpeedPlugin, DeepSpeedPrecisionPlugin
from pytorch_lightning.plugins.training_type.deepspeed import LightningDeepSpeedModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf


class ModelParallelBoringModel(BoringModel):

    def __init__(self):
        super().__init__()
        self.linear = None

    def configure_sharded_model(self) -> None:
        self.linear = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.configure_sharded_model()


def test_deepspeed_lightning_module(tmpdir):
    """
    Test to ensure that a model wrapped in `LightningDeepSpeedModule` moves types and device correctly.
    """

    model = BoringModel()
    module = LightningDeepSpeedModule(model, precision=16)

    module.half()
    assert module.dtype == torch.half
    assert model.dtype == torch.half

    module.to(torch.double)
    assert module.dtype == torch.double
    assert model.dtype == torch.double


@RunIf(min_gpus=1)
def test_deepspeed_lightning_module_precision(tmpdir):
    """
    Test to ensure that a model wrapped in `LightningDeepSpeedModule` moves tensors to half when precision 16.
    """

    model = BoringModel()
    module = LightningDeepSpeedModule(model, precision=16)

    module.cuda().half()
    assert module.dtype == torch.half
    assert model.dtype == torch.half

    x = torch.randn((1, 32), dtype=torch.float).cuda()
    out = module(x)

    assert out.dtype == torch.half

    module.to(torch.double)
    assert module.dtype == torch.double
    assert model.dtype == torch.double


@pytest.fixture
def deepspeed_config():
    return {
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 3e-5,
            },
        },
        'scheduler': {
            "type": "WarmupLR",
            "params": {
                "last_batch_iteration": -1,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 100,
            }
        }
    }


@pytest.fixture
def deepspeed_zero_config(deepspeed_config):
    return {**deepspeed_config, 'zero_allow_untested_optimizer': True, 'zero_optimization': {'stage': 2}}


@RunIf(deepspeed=True)
@pytest.mark.parametrize("input", ("deepspeed", DeepSpeedPlugin))
def test_deepspeed_plugin_string(tmpdir, input):
    """
    Test to ensure that the plugin can be passed via string or instance, and parallel devices is correctly set.
    """

    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        plugins=input if isinstance(input, str) else input(),
    )

    assert isinstance(trainer.accelerator.training_type_plugin, DeepSpeedPlugin)
    assert trainer.accelerator.training_type_plugin.parallel_devices == [torch.device('cpu')]


@RunIf(deepspeed=True)
def test_deepspeed_plugin_env(tmpdir, monkeypatch, deepspeed_config):
    """
        Test to ensure that the plugin can be passed via a string with an environment variable.
    """
    config_path = os.path.join(tmpdir, 'temp.json')
    with open(config_path, 'w') as f:
        f.write(json.dumps(deepspeed_config))
    monkeypatch.setenv("PL_DEEPSPEED_CONFIG_PATH", config_path)

    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        plugins='deepspeed',
    )

    plugin = trainer.accelerator.training_type_plugin
    assert isinstance(plugin, DeepSpeedPlugin)
    assert plugin.parallel_devices == [torch.device('cpu')]
    assert plugin.config == deepspeed_config


@RunIf(amp_native=True, deepspeed=True)
@pytest.mark.parametrize(
    "amp_backend", [
        pytest.param("native", marks=RunIf(amp_native=True)),
        pytest.param("apex", marks=RunIf(amp_apex=True)),
    ]
)
def test_deepspeed_precision_choice(amp_backend, tmpdir):
    """
    Test to ensure precision plugin is also correctly chosen.
    DeepSpeed handles precision via Custom DeepSpeedPrecisionPlugin
    """

    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        plugins='deepspeed',
        amp_backend=amp_backend,
        precision=16,
    )

    assert isinstance(trainer.accelerator.training_type_plugin, DeepSpeedPlugin)
    assert isinstance(trainer.accelerator.precision_plugin, DeepSpeedPrecisionPlugin)
    assert trainer.accelerator.precision_plugin.precision == 16


@RunIf(deepspeed=True)
def test_deepspeed_with_invalid_config_path(tmpdir):
    """
        Test to ensure if we pass an invalid config path we throw an exception.
    """

    with pytest.raises(
        MisconfigurationException, match="You passed in a path to a DeepSpeed config but the path does not exist"
    ):
        DeepSpeedPlugin(config='invalid_path.json')


@RunIf(deepspeed=True)
def test_deepspeed_with_env_path(tmpdir, monkeypatch, deepspeed_config):
    """
    Test to ensure if we pass an env variable, we load the config from the path.
    """
    config_path = os.path.join(tmpdir, 'temp.json')
    with open(config_path, 'w') as f:
        f.write(json.dumps(deepspeed_config))
    monkeypatch.setenv("PL_DEEPSPEED_CONFIG_PATH", config_path)
    plugin = DeepSpeedPlugin()
    assert plugin.config == deepspeed_config


@RunIf(deepspeed=True)
def test_deepspeed_defaults(tmpdir):
    """
    Ensure that defaults are correctly set as a config for DeepSpeed if no arguments are passed.
    """
    plugin = DeepSpeedPlugin()
    assert plugin.config is not None
    assert isinstance(plugin.config["zero_optimization"], dict)


@RunIf(min_gpus=1, deepspeed=True)
def test_invalid_deepspeed_defaults_no_precision(tmpdir):
    """Test to ensure that using defaults, if precision is not set to 16, we throw an exception."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        plugins='deepspeed',
    )
    with pytest.raises(
        MisconfigurationException, match='To use DeepSpeed ZeRO Optimization, you must set precision=16.'
    ):
        trainer.fit(model)


@RunIf(min_gpus=1, deepspeed=True, special=True)
def test_warn_deepspeed_override_backward(tmpdir):
    """Test to ensure that if the backward hook in the LightningModule is overridden, we throw a warning."""

    class TestModel(BoringModel):

        def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
            return loss.backward()

    model = TestModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        plugins=DeepSpeedPlugin(),
        gpus=1,
        precision=16,
    )
    with pytest.warns(UserWarning, match='Overridden backward hook in the LightningModule will be ignored'):
        trainer.fit(model)


@RunIf(min_gpus=1, deepspeed=True, special=True)
def test_deepspeed_run_configure_optimizers(tmpdir):
    """
    Test end to end that deepspeed works with defaults (without ZeRO as that requires compilation),
    whilst using configure_optimizers for optimizers and schedulers.
    """

    class TestCB(Callback):

        def on_train_start(self, trainer, pl_module) -> None:
            from deepspeed.runtime.zero.stage2 import FP16_DeepSpeedZeroOptimizer

            assert isinstance(trainer.optimizers[0], FP16_DeepSpeedZeroOptimizer)
            assert isinstance(trainer.optimizers[0].optimizer, torch.optim.SGD)
            assert trainer.lr_schedulers == []  # DeepSpeed manages LR scheduler internally
            # Ensure DeepSpeed engine has initialized with our optimizer/lr_scheduler
            assert isinstance(trainer.model.lr_scheduler, torch.optim.lr_scheduler.StepLR)

    model = BoringModel()
    trainer = Trainer(
        plugins=DeepSpeedPlugin(),  # disable ZeRO so our optimizers are not wrapped
        default_root_dir=tmpdir,
        gpus=1,
        fast_dev_run=True,
        precision=16,
        callbacks=[TestCB()]
    )

    trainer.fit(model)

    _assert_save_model_is_equal(model, tmpdir, trainer)


@RunIf(min_gpus=1, deepspeed=True, special=True)
def test_deepspeed_config(tmpdir, deepspeed_zero_config):
    """
    Test to ensure deepspeed works correctly when passed a DeepSpeed config object including optimizers/schedulers
    and saves the model weights to load correctly.
    """

    class TestCB(Callback):

        def on_train_start(self, trainer, pl_module) -> None:
            from deepspeed.runtime.lr_schedules import WarmupLR
            from deepspeed.runtime.zero.stage2 import FP16_DeepSpeedZeroOptimizer

            assert isinstance(trainer.optimizers[0], FP16_DeepSpeedZeroOptimizer)
            assert isinstance(trainer.optimizers[0].optimizer, torch.optim.SGD)
            assert trainer.lr_schedulers == []  # DeepSpeed manages LR scheduler internally
            # Ensure DeepSpeed engine has initialized with our optimizer/lr_scheduler
            assert isinstance(trainer.model.lr_scheduler, WarmupLR)

    model = BoringModel()
    trainer = Trainer(
        plugins=[DeepSpeedPlugin(config=deepspeed_zero_config)],
        default_root_dir=tmpdir,
        gpus=1,
        fast_dev_run=True,
        precision=16,
        callbacks=[TestCB()]
    )

    trainer.fit(model)
    trainer.test(model)

    _assert_save_model_is_equal(model, tmpdir, trainer)


@RunIf(min_gpus=1, deepspeed=True, special=True)
def test_deepspeed_custom_precision_params(tmpdir):
    """Ensure if we modify the FP16 parameters via the DeepSpeedPlugin, the deepspeed config contains these changes."""

    class TestCB(Callback):

        def on_train_start(self, trainer, pl_module) -> None:
            assert trainer.training_type_plugin.config['fp16']['loss_scale'] == 10
            assert trainer.training_type_plugin.config['fp16']['initial_scale_power'] == 10
            assert trainer.training_type_plugin.config['fp16']['loss_scale_window'] == 10
            assert trainer.training_type_plugin.config['fp16']['hysteresis'] == 10
            assert trainer.training_type_plugin.config['fp16']['min_loss_scale'] == 10
            raise SystemExit()

    model = BoringModel()
    ds = DeepSpeedPlugin(loss_scale=10, initial_scale_power=10, loss_scale_window=10, hysteresis=10, min_loss_scale=10)
    trainer = Trainer(default_root_dir=tmpdir, plugins=[ds], precision=16, gpus=1, callbacks=[TestCB()])
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(deepspeed=True)
def test_deepspeed_custom_activation_checkpointing_params(tmpdir):
    """Ensure if we modify the activation checkpointing parameters, the deepspeed config contains these changes."""
    ds = DeepSpeedPlugin(
        partition_activations=True,
        cpu_checkpointing=True,
        contiguous_memory_optimization=True,
        synchronize_checkpoint_boundary=True
    )
    checkpoint_config = ds.config['activation_checkpointing']
    assert checkpoint_config['partition_activations']
    assert checkpoint_config['cpu_checkpointing']
    assert checkpoint_config['contiguous_memory_optimization']
    assert checkpoint_config['synchronize_checkpoint_boundary']


@RunIf(min_gpus=1, deepspeed=True)
def test_deepspeed_assert_config_zero_offload_disabled(tmpdir, deepspeed_zero_config):
    """Ensure if we use a config and turn off cpu_offload, that this is set to False within the config."""

    deepspeed_zero_config['zero_optimization']['cpu_offload'] = False

    class TestCallback(Callback):

        def on_before_accelerator_backend_setup(self, trainer, pl_module) -> None:
            assert trainer.training_type_plugin.config['zero_optimization']['cpu_offload'] is False
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        max_epochs=1,
        plugins=[DeepSpeedPlugin(config=deepspeed_zero_config)],
        precision=16,
        gpus=1,
        default_root_dir=tmpdir,
        callbacks=[TestCallback()]
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=2, deepspeed=True, special=True)
def test_deepspeed_multigpu(tmpdir, deepspeed_config):
    """
    Test to ensure that DeepSpeed with multiple GPUs works, without ZeRO Optimization as this requires compilation.
    """
    model = BoringModel()
    trainer = Trainer(
        plugins=[DeepSpeedPlugin(zero_optimization=False, stage=2)],
        default_root_dir=tmpdir,
        gpus=2,
        fast_dev_run=True,
        precision=16,
    )
    trainer.fit(model)
    trainer.test(model)

    _assert_save_model_is_equal(model, tmpdir, trainer)


class ModelParallelClassificationModel(LightningModule):

    def __init__(self, lr: float = 0.01, num_blocks: int = 5):
        super().__init__()
        self.lr = lr
        self.num_blocks = num_blocks

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

    def make_block(self):
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())

    def configure_sharded_model(self) -> None:
        self.model = nn.Sequential(*(self.make_block() for x in range(self.num_blocks)), nn.Linear(32, 3))

    def forward(self, x):
        x = self.model(x)
        # Ensure output is in float32 for softmax operation
        x = x.float()
        logits = F.softmax(x, dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc(logits, y), prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log('val_loss', F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log('val_acc', self.valid_acc(logits, y), prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log('test_loss', F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log('test_acc', self.test_acc(logits, y), prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        logits = self.forward(x)
        self.test_acc(logits, y)
        return self.test_acc.compute()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
        }]


@RunIf(min_gpus=2, deepspeed=True, special=True)
def test_deepspeed_multigpu_stage_3(tmpdir, deepspeed_config):
    """
    Test to ensure ZeRO Stage 3 works with a parallel model.
    """
    model = ModelParallelBoringModel()
    trainer = Trainer(
        plugins=[DeepSpeedPlugin(stage=3)],
        default_root_dir=tmpdir,
        gpus=2,
        fast_dev_run=True,
        precision=16,
    )
    trainer.fit(model)
    trainer.test(model)

    _assert_save_model_is_equal(model, tmpdir, trainer, cls=ModelParallelBoringModel)


def run_checkpoint_test(tmpdir, save_full_weights):
    seed_everything(1)
    model = ModelParallelClassificationModel()
    dm = ClassifDataModule()
    ck = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, save_top_k=-1)
    trainer = Trainer(
        max_epochs=10,
        plugins=[DeepSpeedPlugin(stage=3, save_full_weights=save_full_weights)],
        default_root_dir=tmpdir,
        gpus=2,
        precision=16,
        accumulate_grad_batches=2,
        callbacks=[ck]
    )
    trainer.fit(model, datamodule=dm)

    results = trainer.test(model, datamodule=dm)
    assert results[0]['test_acc'] > 0.7

    saved_results = trainer.test(ckpt_path=ck.best_model_path, datamodule=dm)
    assert saved_results[0]['test_acc'] > 0.7
    assert saved_results == results

    trainer = Trainer(
        max_epochs=10,
        plugins=[DeepSpeedPlugin(stage=3, save_full_weights=save_full_weights)],
        default_root_dir=tmpdir,
        gpus=2,
        precision=16,
        accumulate_grad_batches=2,
        callbacks=[ck],
        resume_from_checkpoint=ck.best_model_path
    )
    results = trainer.test(model, datamodule=dm)
    assert results[0]['test_acc'] > 0.7

    dm.predict_dataloader = dm.test_dataloader
    results = trainer.predict(datamodule=dm)
    assert results[-1] > 0.7


@RunIf(min_gpus=2, deepspeed=True, special=True)
def test_deepspeed_multigpu_stage_3_checkpointing(tmpdir):
    """
    Test to ensure with Stage 3 and multiple GPUs that we can save/load a model resuming from a checkpoint,
    and see convergence.
    """
    run_checkpoint_test(tmpdir, save_full_weights=False)


@RunIf(min_gpus=2, deepspeed=True, special=True)
def test_deepspeed_multigpu_stage_3_checkpointing_full_weights(tmpdir):
    """
    Test to ensure with Stage 3 and multiple GPUs that we can save/load a model resuming from a checkpoint,
    where we save the full weights to one file.
    """
    run_checkpoint_test(tmpdir, save_full_weights=True)


@RunIf(min_gpus=2, deepspeed=True, special=True)
@pytest.mark.parametrize('cpu_offload', [True, False])
def test_deepspeed_multigpu_stage_2_accumulated_grad_batches(tmpdir, cpu_offload):
    """
    Test to ensure with Stage 2 and multiple GPUs, accumulated grad batches works.
    """
    seed_everything(42)

    class VerificationCallback(Callback):

        def on_train_batch_start(
            self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
        ) -> None:
            deepspeed_engine = trainer.training_type_plugin.model
            assert trainer.global_step == deepspeed_engine.global_steps

    model = ModelParallelClassificationModel()
    dm = ClassifDataModule()
    trainer = Trainer(
        max_epochs=5,
        plugins=[DeepSpeedPlugin(stage=2, cpu_offload=cpu_offload)],
        gpus=2,
        limit_val_batches=2,
        precision=16,
        accumulate_grad_batches=2,
        callbacks=[VerificationCallback()]
    )
    trainer.fit(model, datamodule=dm)


@RunIf(min_gpus=2, deepspeed=True, special=True)
def test_deepspeed_multigpu_test(tmpdir, deepspeed_config):
    """
    Test to ensure we can use DeepSpeed with just test using ZeRO Stage 3.
    """
    model = ModelParallelBoringModel()
    trainer = Trainer(
        plugins=[DeepSpeedPlugin(stage=3)],
        default_root_dir=tmpdir,
        gpus=2,
        fast_dev_run=True,
        precision=16,
    )
    trainer.test(model)


def _assert_save_model_is_equal(model, tmpdir, trainer, cls=BoringModel):
    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(checkpoint_path)
    # carry out the check only on rank 0
    if trainer.global_rank == 0:
        saved_model = cls.load_from_checkpoint(checkpoint_path)
        if model.dtype == torch.half:
            saved_model = saved_model.half()  # model is loaded in float32 as default, move it to float16
        model = model.cpu()
        # Assert model parameters are identical after loading
        for orig_param, trained_model_param in zip(model.parameters(), saved_model.parameters()):
            assert torch.equal(orig_param, trained_model_param)
