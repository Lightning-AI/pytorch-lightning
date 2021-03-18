import json
import os

import pytest
import torch
from torch import Tensor
from torch.optim import Optimizer

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin, DeepSpeedPrecisionPlugin
from pytorch_lightning.plugins.training_type.deepspeed import LightningDeepSpeedModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


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


@RunIf(min_gpus=1, deepspeed=True, special=True)
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
    """Test end to end that deepspeed works with defaults (without ZeRO as that requires compilation),
        whilst using configure_optimizers for optimizers and schedulers."""

    class TestModel(BoringModel):

        def on_train_start(self) -> None:
            from deepspeed.runtime.zero.stage2 import FP16_DeepSpeedZeroOptimizer

            assert isinstance(self.trainer.optimizers[0], FP16_DeepSpeedZeroOptimizer)
            assert isinstance(self.trainer.optimizers[0].optimizer, torch.optim.SGD)
            assert self.trainer.lr_schedulers == []  # DeepSpeed manages LR scheduler internally
            # Ensure DeepSpeed engine has initialized with our optimizer/lr_scheduler
            assert isinstance(self.trainer.model.lr_scheduler, torch.optim.lr_scheduler.StepLR)

    model = TestModel()
    trainer = Trainer(
        plugins=DeepSpeedPlugin(),  # disable ZeRO so our optimizers are not wrapped
        default_root_dir=tmpdir,
        gpus=1,
        fast_dev_run=True,
        precision=16,
    )

    trainer.fit(model)

    _assert_save_model_is_equal(model, tmpdir, trainer)


@RunIf(min_gpus=1, deepspeed=True, special=True)
def test_deepspeed_config(tmpdir, deepspeed_zero_config):
    """
        Test to ensure deepspeed works correctly when passed a DeepSpeed config object including optimizers/schedulers
        and saves the model weights to load correctly.
    """

    class TestModel(BoringModel):

        def on_train_start(self) -> None:
            from deepspeed.runtime.lr_schedules import WarmupLR
            from deepspeed.runtime.zero.stage2 import FP16_DeepSpeedZeroOptimizer

            assert isinstance(self.trainer.optimizers[0], FP16_DeepSpeedZeroOptimizer)
            assert isinstance(self.trainer.optimizers[0].optimizer, torch.optim.SGD)
            assert self.trainer.lr_schedulers == []  # DeepSpeed manages LR scheduler internally
            # Ensure DeepSpeed engine has initialized with our optimizer/lr_scheduler
            assert isinstance(self.trainer.model.lr_scheduler, WarmupLR)

    model = TestModel()
    trainer = Trainer(
        plugins=[DeepSpeedPlugin(config=deepspeed_zero_config)],
        default_root_dir=tmpdir,
        gpus=1,
        fast_dev_run=True,
        precision=16,
    )

    trainer.fit(model)
    trainer.test(model)

    _assert_save_model_is_equal(model, tmpdir, trainer)


@RunIf(min_gpus=1, deepspeed=True, special=True)
def test_deepspeed_custom_precision_params(tmpdir):
    """Ensure if we modify the FP16 parameters via the DeepSpeedPlugin, the deepspeed config contains these changes."""

    class TestModel(BoringModel):

        def on_train_start(self) -> None:
            assert self.trainer.training_type_plugin.config['fp16']['loss_scale'] == 10
            assert self.trainer.training_type_plugin.config['fp16']['initial_scale_power'] == 10
            assert self.trainer.training_type_plugin.config['fp16']['loss_scale_window'] == 10
            assert self.trainer.training_type_plugin.config['fp16']['hysteresis'] == 10
            assert self.trainer.training_type_plugin.config['fp16']['min_loss_scale'] == 10
            raise SystemExit()

    model = TestModel()
    ds = DeepSpeedPlugin(loss_scale=10, initial_scale_power=10, loss_scale_window=10, hysteresis=10, min_loss_scale=10)
    trainer = Trainer(default_root_dir=tmpdir, plugins=[ds], precision=16, gpus=1)
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1, deepspeed=True, special=True)
def test_deepspeed_assert_config_zero_offload_disabled(tmpdir, deepspeed_zero_config):
    """Ensure if we use a config and turn off cpu_offload, that this is set to False within the config."""

    deepspeed_zero_config['zero_optimization']['cpu_offload'] = False

    class TestModel(BoringModel):

        def on_train_start(self) -> None:
            assert self.trainer.training_type_plugin.config['zero_optimization']['cpu_offload'] is False
            raise SystemExit()

    model = TestModel()
    trainer = Trainer(
        plugins=[DeepSpeedPlugin(config=deepspeed_zero_config)],
        precision=16,
        gpus=1,
        default_root_dir=tmpdir,
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=2, special=True, deepspeed=True)
def test_deepspeed_multigpu(tmpdir, deepspeed_config):
    """
        Test to ensure that DeepSpeed with multiple GPUs works, without ZeRO Optimization as this requires compilation.
    """
    model = BoringModel()
    trainer = Trainer(
        plugins=[DeepSpeedPlugin()],
        default_root_dir=tmpdir,
        gpus=2,
        fast_dev_run=True,
        precision=16,
    )
    trainer.fit(model)
    trainer.test(model)

    _assert_save_model_is_equal(model, tmpdir, trainer)


def _assert_save_model_is_equal(model, tmpdir, trainer):
    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(checkpoint_path)
    # carry out the check only on rank 0
    if trainer.global_rank == 0:
        saved_model = BoringModel.load_from_checkpoint(checkpoint_path)
        if model.dtype == torch.half:
            saved_model = saved_model.half()  # model is loaded in float32 as default, move it to float16
        model = model.cpu()
        # Assert model parameters are identical after loading
        for orig_param, trained_model_param in zip(model.parameters(), saved_model.parameters()):
            assert torch.equal(orig_param, trained_model_param)
