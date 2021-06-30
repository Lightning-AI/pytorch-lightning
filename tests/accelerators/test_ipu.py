# Copyright The PyTorch Lightning team.
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

import pytest
import torch
import torch.nn.functional as F

from pytorch_lightning import Callback, seed_everything, Trainer
from pytorch_lightning.accelerators import IPUAccelerator
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins import IPUPlugin, IPUPrecisionPlugin
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import _IPU_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel

if _IPU_AVAILABLE:
    import poptorch


class IPUModel(BoringModel):

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def training_epoch_end(self, outputs) -> None:
        pass

    def validation_epoch_end(self, outputs) -> None:
        pass

    def test_epoch_end(self, outputs) -> None:
        pass


class IPUClassificationModel(ClassificationModel):

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        # todo (sean): currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log('val_acc', torch.stack(outputs).mean())

    def test_epoch_end(self, outputs) -> None:
        self.log('test_acc', torch.stack(outputs).mean())


@pytest.mark.skipif(_IPU_AVAILABLE, reason="test requires non-IPU machine")
def test_fail_if_no_ipus(tmpdir):
    with pytest.raises(MisconfigurationException, match="IPU Accelerator requires IPU devices to run"):
        Trainer(default_root_dir=tmpdir, ipus=1)

    with pytest.raises(MisconfigurationException, match="IPU Accelerator requires IPU devices to run"):
        Trainer(default_root_dir=tmpdir, ipus=1, accelerator='ipu')


@RunIf(ipu=True)
def test_accelerator_selected(tmpdir):
    trainer = Trainer(default_root_dir=tmpdir, ipus=1)
    assert isinstance(trainer.accelerator, IPUAccelerator)
    trainer = Trainer(default_root_dir=tmpdir, ipus=1, accelerator='ipu')
    assert isinstance(trainer.accelerator, IPUAccelerator)


@RunIf(ipu=True)
@pytest.mark.parametrize('ipus', [1, 4])
def test_all_stages(tmpdir, ipus):
    model = IPUModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, ipus=ipus)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model, model.val_dataloader())


@RunIf(ipu=True)
@pytest.mark.parametrize('ipus', [1, 4])
def test_inference_only(tmpdir, ipus):
    model = IPUModel()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, ipus=ipus)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model, model.val_dataloader())


@RunIf(ipu=True)
def test_optimization(tmpdir):
    seed_everything(42)

    dm = ClassifDataModule(length=1024)
    model = IPUClassificationModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        ipus=2,
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # validate
    result = trainer.validate(datamodule=dm)
    assert dm.trainer is not None
    assert result[0]['val_acc'] > 0.7

    # test
    result = trainer.test(model, datamodule=dm)
    assert dm.trainer is not None
    test_result = result[0]['test_acc']
    assert test_result > 0.6

    # test saved model
    model_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(model_path)

    model = IPUClassificationModel.load_from_checkpoint(model_path)

    trainer = Trainer(default_root_dir=tmpdir, ipus=2)

    result = trainer.test(model, datamodule=dm)
    saved_result = result[0]['test_acc']
    assert saved_result == test_result


@RunIf(ipu=True)
def test_mixed_precision(tmpdir):

    class TestCallback(Callback):

        def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
            assert trainer.accelerator.model.precision == 16
            raise SystemExit

    model = IPUModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, ipus=1, precision=16, callbacks=TestCallback())
    assert isinstance(trainer.accelerator.precision_plugin, IPUPrecisionPlugin)
    assert trainer.accelerator.precision_plugin.precision == 16
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(ipu=True)
def test_pure_half_precision(tmpdir):

    class TestCallback(Callback):

        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            assert trainer.accelerator.model.precision == 16
            for param in trainer.accelerator.model.parameters():
                assert param.dtype == torch.float16
            raise SystemExit

    model = IPUModel()
    model = model.half()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, ipus=1, precision=16, callbacks=TestCallback())

    assert isinstance(trainer.accelerator.training_type_plugin, IPUPlugin)
    assert isinstance(trainer.accelerator.precision_plugin, IPUPrecisionPlugin)
    assert trainer.accelerator.precision_plugin.precision == 16

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(ipu=True)
def test_device_iterations_ipu_plugin(tmpdir):

    class TestCallback(Callback):

        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            assert trainer.accelerator.training_type_plugin.device_iterations == 2
            # assert device iterations has been set correctly within the poptorch options
            poptorch_model = trainer.accelerator.training_type_plugin.poptorch_models[RunningStage.TRAINING]
            assert poptorch_model._options.toDict()['device_iterations'] == 2
            raise SystemExit

    model = IPUModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        ipus=1,
        plugins=IPUPlugin(device_iterations=2),
        callbacks=TestCallback()
    )
    assert isinstance(trainer.accelerator.training_type_plugin, IPUPlugin)
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(ipu=True)
def test_accumulated_batches(tmpdir):

    class TestCallback(Callback):

        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            # ensure the accumulation_scheduler is overridden to accumulate every batch
            # since ipu handle accumulation
            assert trainer.accumulation_scheduler.scheduling == {0: 1}
            # assert poptorch option have been set correctly
            poptorch_model = trainer.accelerator.training_type_plugin.poptorch_models[RunningStage.TRAINING]
            assert poptorch_model._options.Training.toDict()['gradient_accumulation'] == 2
            raise SystemExit

    model = IPUModel()
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, ipus=1, accumulate_grad_batches=2, callbacks=TestCallback()
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(ipu=True)
def test_stages_correct(tmpdir):
    """Ensure all stages correctly are traced correctly by asserting the output for each stage"""

    class StageModel(IPUModel):

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            # tracing requires a loss value that depends on the model.
            # force it to be a value but ensure we use the loss.
            return (loss - loss) + torch.tensor(1)

        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            return (loss - loss) + torch.tensor(2)

        def test_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            return (loss - loss) + torch.tensor(3)

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            output = super().predict_step(batch, batch_idx)
            return (output - output) + torch.tensor(4)

    class TestCallback(Callback):

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
            assert outputs['loss'].item() == 1

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
            assert outputs.item() == 2

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
            assert outputs.item() == 3

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
            assert torch.all(outputs == 4).item()

    model = StageModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, ipus=1, callbacks=TestCallback())
    trainer.fit(model)
    trainer.test(model)
    trainer.validate(model)
    trainer.predict(model, model.test_dataloader())


@RunIf(ipu=True)
def test_accumulate_grad_batches_dict_fails(tmpdir):
    model = IPUModel()
    trainer = Trainer(default_root_dir=tmpdir, ipus=1, accumulate_grad_batches={0: 1})
    with pytest.raises(
        MisconfigurationException, match="IPUs currently only support accumulate_grad_batches being an integer value."
    ):
        trainer.fit(model)


@RunIf(ipu=True)
def test_clip_gradients_fails(tmpdir):
    model = IPUModel()
    trainer = Trainer(default_root_dir=tmpdir, ipus=1, gradient_clip_val=10)
    with pytest.raises(MisconfigurationException, match="IPUs currently do not support clipping gradients."):
        trainer.fit(model)


@RunIf(ipu=True)
def test_autoreport(tmpdir):
    """Ensure autoreport dumps to a file."""
    model = IPUModel()
    autoreport_path = os.path.join(tmpdir, 'report/')
    trainer = Trainer(
        default_root_dir=tmpdir,
        ipus=1,
        fast_dev_run=True,
        plugins=IPUPlugin(autoreport=True, autoreport_dir=autoreport_path)
    )
    trainer.fit(model)
    assert os.path.exists(autoreport_path)
    assert os.path.isfile(autoreport_path + 'profile.pop')


@RunIf(ipu=True)
def test_manual_poptorch_opts(tmpdir):
    """Ensure if the user passes manual poptorch Options, we run with the correct object."""
    model = IPUModel()
    inference_opts = poptorch.Options()
    training_opts = poptorch.Options()

    trainer = Trainer(
        default_root_dir=tmpdir,
        ipus=1,
        fast_dev_run=True,
        plugins=IPUPlugin(inference_opts=inference_opts, training_opts=training_opts)
    )
    trainer.fit(model)

    assert isinstance(trainer.accelerator.training_type_plugin, IPUPlugin)
    assert trainer.accelerator.training_type_plugin.training_opts == training_opts
    assert trainer.accelerator.training_type_plugin.inference_opts == inference_opts


@RunIf(ipu=True)
def test_manual_poptorch_opts_ipu_count(tmpdir):
    """
    Ensure if the user passes manual poptorch Options
    and the number of ipus do not match, we warn and we set it for the user.
    """

    manual_ipus = 1
    expected_ipus = 2
    model = IPUModel()
    inference_opts = poptorch.Options()
    inference_opts.replicationFactor(manual_ipus)

    training_opts = poptorch.Options()
    training_opts.replicationFactor(manual_ipus)

    trainer = Trainer(
        default_root_dir=tmpdir,
        ipus=expected_ipus,
        fast_dev_run=True,
        plugins=IPUPlugin(inference_opts=inference_opts, training_opts=training_opts)
    )
    with pytest.warns(
        UserWarning,
        match=f"Manual poptorch.Options set replicationFactor to {manual_ipus} "
        f"which differs to the ipus={expected_ipus} flag passed to the Trainer. "
        f"Setting to {expected_ipus} in the poptorch.Options."
    ):
        trainer.fit(model)
        assert isinstance(trainer.accelerator.training_type_plugin, IPUPlugin)
        assert trainer.accelerator.training_type_plugin.training_opts.replication_factor == 2
        assert trainer.accelerator.training_type_plugin.inference_opts.replication_factor == 2


@RunIf(ipu=True)
def test_manual_poptorch_opts_inference_grad_accum(tmpdir):
    """
    Ensure if the user passes manual poptorch Options
    and grad accumulation is set greater than 1 for inference, we warn and set to 1.
    """

    model = IPUModel()
    inference_opts = poptorch.Options()
    inference_opts.Training.gradientAccumulation(4)

    training_opts = poptorch.Options()
    training_opts.Training.gradientAccumulation(1)

    trainer = Trainer(
        default_root_dir=tmpdir,
        ipus=1,
        fast_dev_run=True,
        plugins=IPUPlugin(inference_opts=inference_opts, training_opts=training_opts)
    )
    with pytest.warns(
        UserWarning,
        match="Inference poptorch.Options should set gradientAccumulation to 1. "
        "Setting gradientAccumulation to 1 for inference options.",
    ):
        trainer.fit(model)
        assert isinstance(trainer.accelerator.training_type_plugin, IPUPlugin)
        assert trainer.accelerator.training_type_plugin.inference_opts.Training.gradient_accumulation == 1


@RunIf(ipu=True)
def test_manual_poptorch_opts_train_grad_accum(tmpdir):
    """
    Ensure if the user passes manual poptorch Options
    and grad accumulation differs to accumulate_grad_batches, we
    """

    model = IPUModel()
    inference_opts = poptorch.Options()
    inference_opts.Training.gradientAccumulation(1)

    training_opts = poptorch.Options()
    training_opts.Training.gradientAccumulation(2)

    trainer = Trainer(
        default_root_dir=tmpdir,
        ipus=1,
        fast_dev_run=True,
        accumulate_grad_batches=1,
        plugins=IPUPlugin(inference_opts=inference_opts, training_opts=training_opts)
    )
    with pytest.warns(
        UserWarning,
        match=f"Training poptorch.Options set gradientAccumulation to {2}. "
        f"This is different to accumulate_grad_batches which was set to {1}. "
        f"To change gradientAccumulation, please set accumulate_grad_batches in the Trainer. "
        f"Setting poptorch.Options gradientAccumulation to {1}",
    ):
        trainer.fit(model)
        assert isinstance(trainer.accelerator.training_type_plugin, IPUPlugin)
        assert trainer.accelerator.training_type_plugin.inference_opts.Training.gradient_accumulation == 1


@RunIf(ipu=True)
def test_manual_poptorch_opts_custom(tmpdir):
    """
    Ensure if the user passes manual poptorch Options with custom parameters set,
    we respect them in our poptorch options.
    """

    model = IPUModel()
    inference_opts = poptorch.Options()
    inference_opts.deviceIterations(16)
    inference_opts.replicationFactor(2)
    inference_opts.Training.gradientAccumulation(1)

    training_opts = poptorch.Options()
    training_opts.deviceIterations(8)
    training_opts.replicationFactor(2)
    training_opts.Training.gradientAccumulation(2)

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        plugins=IPUPlugin(inference_opts=inference_opts, training_opts=training_opts)
    )
    trainer.fit(model)
    plugin = trainer.accelerator.training_type_plugin
    assert isinstance(plugin, IPUPlugin)
    inference_opts = plugin.inference_opts
    training_opts = plugin.training_opts
    assert inference_opts.device_iterations == 16
    assert inference_opts.replication_factor == 2
    assert inference_opts.Training.gradient_accumulation == 1

    assert training_opts.device_iterations == 8
    assert training_opts.replication_factor == 2
    assert training_opts.Training.gradient_accumulation == 2


@RunIf(ipu=True)
def test_default_opts(tmpdir):
    """
    Ensure default opts are set correctly in the IPUPlugin.
    """

    model = IPUModel()

    trainer = Trainer(default_root_dir=tmpdir, ipus=1, fast_dev_run=True)
    trainer.fit(model)
    assert isinstance(trainer.accelerator.training_type_plugin, IPUPlugin)
    inference_opts = trainer.accelerator.training_type_plugin.inference_opts
    training_opts = trainer.accelerator.training_type_plugin.training_opts
    for opts in (inference_opts, training_opts):
        assert isinstance(opts, poptorch.Options)
        assert opts.Training.gradient_accumulation == 1
        assert opts.device_iterations == 1
        assert opts.replication_factor == 1


@RunIf(ipu=True)
def test_multi_optimizers_fails(tmpdir):
    """
    Ensure if there are multiple optimizers, we throw an exception
    """

    class TestModel(IPUModel):

        def configure_optimizers(self):
            return [torch.optim.Adam(self.parameters()), torch.optim.Adam(self.parameters())]

    model = TestModel()

    trainer = Trainer(default_root_dir=tmpdir, ipus=1)
    with pytest.raises(MisconfigurationException, match="IPUs currently only support one optimizer."):
        trainer.fit(model)


@RunIf(ipu=True)
def test_precision_plugin(tmpdir):
    """
    Ensure precision plugin value is set correctly.
    """

    plugin = IPUPrecisionPlugin(precision=16)
    assert plugin.precision == 16
