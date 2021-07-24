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
import collections
import copy
from typing import Callable, Union

import pytest
import torch
from torchmetrics.functional import mean_absolute_percentage_error as mape

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import get_model_size_mb
from tests.helpers.boring_model import RandomDataset
from tests.helpers.datamodules import RegressDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import RegressionModel


@pytest.mark.parametrize("observe", ["average", "histogram"])
@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("convert", [True, False])
@RunIf(quantization=True)
def test_quantization(tmpdir, observe: str, fuse: bool, convert: bool):
    """Parity test for quant model."""
    seed_everything(42)
    dm = RegressDataModule()
    trainer_args = dict(default_root_dir=tmpdir, max_epochs=7, gpus=int(torch.cuda.is_available()))
    model = RegressionModel()
    qmodel = copy.deepcopy(model)

    trainer = Trainer(**trainer_args)
    trainer.fit(model, datamodule=dm)
    org_size = get_model_size_mb(model)
    org_score = torch.mean(torch.tensor([mape(model(x), y) for x, y in dm.test_dataloader()]))

    fusing_layers = [(f'layer_{i}', f'layer_{i}a') for i in range(3)] if fuse else None
    qcb = QuantizationAwareTraining(
        observer_type=observe,
        modules_to_fuse=fusing_layers,
        quantize_on_fit_end=convert,
        disable_observers=(),
    )
    trainer = Trainer(callbacks=[qcb], **trainer_args)
    trainer.fit(qmodel, datamodule=dm)

    quant_calls = qcb._forward_calls
    assert quant_calls == qcb._forward_calls
    quant_score = torch.mean(torch.tensor([mape(qmodel(x), y) for x, y in dm.test_dataloader()]))
    # test that the test score is almost the same as with pure training
    assert torch.allclose(org_score, quant_score, atol=0.45)
    model_path = trainer.checkpoint_callback.best_model_path

    trainer_args.update(dict(max_epochs=1, enable_checkpointing=False))
    if not convert:
        trainer = Trainer(callbacks=[QuantizationAwareTraining()], **trainer_args)
        trainer.fit(qmodel, datamodule=dm)
        qmodel.eval()
        torch.quantization.convert(qmodel, inplace=True)

    quant_size = get_model_size_mb(qmodel)
    # test that the trained model is smaller then initial
    size_ratio = quant_size / org_size
    assert size_ratio < 0.65

    # todo: make it work also with strict loading
    qmodel2 = RegressionModel.load_from_checkpoint(model_path, strict=False)
    quant2_score = torch.mean(torch.tensor([mape(qmodel2(x), y) for x, y in dm.test_dataloader()]))
    assert torch.allclose(org_score, quant2_score, atol=0.45)


@RunIf(quantization=True)
def test_quantize_torchscript(tmpdir):
    """Test converting to torchscipt."""
    dm = RegressDataModule()
    qmodel = RegressionModel()
    qcb = QuantizationAwareTraining(input_compatible=False)
    trainer = Trainer(callbacks=[qcb], default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(qmodel, datamodule=dm)

    batch = iter(dm.test_dataloader()).next()
    qmodel(qmodel.quant(batch[0]))

    tsmodel = qmodel.to_torchscript()
    tsmodel(tsmodel.quant(batch[0]))


@RunIf(quantization=True)
def test_quantization_exceptions(tmpdir):
    """Test wrong fuse layers."""
    with pytest.raises(MisconfigurationException, match="Unsupported qconfig"):
        QuantizationAwareTraining(qconfig=["abc"])

    with pytest.raises(MisconfigurationException, match="Unsupported observer type"):
        QuantizationAwareTraining(observer_type="abc")

    with pytest.raises(MisconfigurationException, match="Unsupported `collect_quantization`"):
        QuantizationAwareTraining(collect_quantization="abc")

    with pytest.raises(MisconfigurationException, match="Unsupported `collect_quantization`"):
        QuantizationAwareTraining(collect_quantization=1.2)

    with pytest.raises(MisconfigurationException, match='Unsupported stages'):
        QuantizationAwareTraining(disable_observers=('abc', ))

    fusing_layers = [(f'layers.mlp_{i}', f'layers.NONE-mlp_{i}a') for i in range(3)]
    qcb = QuantizationAwareTraining(modules_to_fuse=fusing_layers)
    trainer = Trainer(callbacks=[qcb], default_root_dir=tmpdir, max_epochs=1)
    with pytest.raises(MisconfigurationException, match="one or more of them is not your model attributes"):
        trainer.fit(RegressionModel(), datamodule=RegressDataModule())


def custom_trigger_never(trainer):
    return False


def custom_trigger_even(trainer):
    return trainer.current_epoch % 2 == 0


def custom_trigger_last(trainer):
    return trainer.current_epoch == (trainer.max_epochs - 1)


@pytest.mark.parametrize(
    "trigger_fn,expected_count",
    [(None, 9), (3, 3), (custom_trigger_never, 0), (custom_trigger_even, 5), (custom_trigger_last, 2)],
)
@RunIf(quantization=True)
def test_quantization_triggers(tmpdir, trigger_fn: Union[None, int, Callable], expected_count: int):
    """Test  how many times the quant is called."""
    dm = RegressDataModule()
    qmodel = RegressionModel()
    qcb = QuantizationAwareTraining(collect_quantization=trigger_fn)
    trainer = Trainer(
        callbacks=[qcb], default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, max_epochs=4
    )
    trainer.fit(qmodel, datamodule=dm)

    assert qcb._forward_calls == expected_count


class CheckObserverDisabledModel(RegressionModel):
    """
    The ``CheckObserverDisabledModel`` will check:

    - The fake-quantization modules are restored to initial states after the sanity check if observers should be
      disabled during validating.
    - The observers belonging to fake-quantization modules are disabled during validating/testing/predicting.

    The ``CheckObserverDisabledModel`` may disable all observers after ``disable_observers_after_step`` steps.

    """

    def __init__(self, disable_observers, disable_observers_after_step=-1):
        super().__init__()
        self._disable_observers = set(disable_observers)
        self._disable_observers_after_step = disable_observers_after_step

        self._fake_quants = ()
        self._fake_quant_to_initial_state = {}
        self._last_train_fake_quant_to_observer_enabled = {}

    def _collect_observer_enabled(self):
        return {fake_quant: fake_quant.observer_enabled.clone() for fake_quant in self._fake_quants}

    def _assert_fake_quant_initial_state(self):
        # Check fake-quantization modules are restored to initial states.
        for fake_quant, initial_state in self._fake_quant_to_initial_state.items():
            state = fake_quant.state_dict()
            for key, initial_value in initial_state.items():
                assert torch.equal(state[key], initial_value)

    def _assert_observer_enabled(self, expected_fake_quant_to_observer_enabled):
        # Check observers belonging to fake-quantization modules are set properly.
        fake_quant_to_observer_enabled = self._collect_observer_enabled()
        for fake_quant in self._fake_quants:
            observer_enabled = fake_quant_to_observer_enabled[fake_quant]
            expected_observer_enabled = expected_fake_quant_to_observer_enabled[fake_quant]
            if isinstance(expected_observer_enabled, bool):
                # ``torch.quantization.FakeQuantize`` checks observer_enabled[0] == 1.
                observer_enabled = observer_enabled[0] == 1
                assert observer_enabled == expected_observer_enabled
            else:
                assert torch.equal(observer_enabled, expected_observer_enabled)

    def _check_observer_state(self, should_disable):
        if should_disable:
            self._assert_observer_enabled(collections.defaultdict(lambda: False))
        else:
            self._assert_observer_enabled(self._last_train_fake_quant_to_observer_enabled)

    def on_fit_start(self):
        self._fake_quants = tuple(
            module for module in self.modules() if isinstance(module, torch.quantization.FakeQuantizeBase)
        )
        self._fake_quant_to_initial_state = {fake_quant: fake_quant.state_dict() for fake_quant in self._fake_quants}
        self._last_train_fake_quant_to_observer_enabled = self._collect_observer_enabled()

    def training_step(self, batch, batch_idx):
        self._assert_observer_enabled(self._last_train_fake_quant_to_observer_enabled)

        result = super().training_step(batch, batch_idx)
        if self._disable_observers_after_step == self.global_step:
            self.apply(torch.quantization.disable_observer)

        self._last_train_fake_quant_to_observer_enabled = self._collect_observer_enabled()

        return result

    def on_validation_start(self):
        self._check_observer_state('validate' in self._disable_observers and not self.trainer.sanity_checking)

    def on_validation_end(self):
        if 'validate' in self._disable_observers:
            if self.trainer.sanity_checking:
                self._assert_fake_quant_initial_state()
            else:
                self._assert_observer_enabled(self._last_train_fake_quant_to_observer_enabled)

    def on_test_start(self):
        self._check_observer_state('test' in self._disable_observers)

    def on_test_end(self):
        if 'test' in self._disable_observers:
            self._assert_observer_enabled(self._last_train_fake_quant_to_observer_enabled)

    def on_predict_start(self):
        self._check_observer_state('predict' in self._disable_observers)

    def on_predict_end(self):
        if 'predict' in self._disable_observers:
            self._assert_observer_enabled(self._last_train_fake_quant_to_observer_enabled)


@pytest.mark.parametrize("observe", ['average', 'histogram'])
@pytest.mark.parametrize("disable_observers", [('validate', 'test', 'predict'), ('validate', ), ()])
@RunIf(quantization=True)
def test_disable_observers(tmpdir, observe, disable_observers):
    """Test disabling observers"""
    num_features = 16
    dm = RegressDataModule(num_features=num_features, length=800, batch_size=10)
    qmodel = CheckObserverDisabledModel(disable_observers, disable_observers_after_step=2)
    qconfig = 'fbgemm'
    if observe == 'average':
        qconfig = torch.quantization.get_default_qat_qconfig(backend=qconfig)
    elif observe == 'histogram':
        # Currently passing ``observer_type='histogram'`` to ``QuantizationAwareTraining`` will only do quantization
        # range calibration without any fake-quantization modules. We create the ``qconfig`` for histogram observers
        # manually.
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.HistogramObserver, reduce_range=True
            ),
            weight=torch.quantization.get_default_qat_qconfig(backend=qconfig).weight,
        )
    qcb = QuantizationAwareTraining(
        qconfig=qconfig,
        quantize_on_fit_end=False,
        disable_observers=disable_observers,
    )
    trainer = Trainer(
        callbacks=[qcb],
        default_root_dir=tmpdir,
        limit_train_batches=5,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        val_check_interval=1,
        num_sanity_val_steps=1,
        max_epochs=1,
    )
    trainer.fit(qmodel, datamodule=dm)
    trainer.validate(verbose=False)
    trainer.test(verbose=False)
    trainer.predict(dataloaders=[torch.utils.data.DataLoader(RandomDataset(num_features, 16))])
