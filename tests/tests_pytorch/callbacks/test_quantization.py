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
import copy
from typing import Callable, Union

import pytest
import torch
from torch.quantization import FakeQuantizeBase

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import QuantizationAwareTraining
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.datamodules import RegressDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import RegressionModel


@RunIf(quantization=True, sklearn=True)
def test_quantize_torchscript(tmpdir):
    """Test converting to torchscipt."""
    dm = RegressDataModule()
    qmodel = RegressionModel()
    qcb = QuantizationAwareTraining(input_compatible=False)
    trainer = Trainer(callbacks=[qcb], default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(qmodel, datamodule=dm)

    batch = next(iter(dm.test_dataloader()))
    qmodel(qmodel.quant(batch[0]))

    tsmodel = qmodel.to_torchscript()
    tsmodel(tsmodel.quant(batch[0]))


@RunIf(quantization=True, sklearn=True)
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

    with pytest.raises(MisconfigurationException, match="Unsupported stages"):
        QuantizationAwareTraining(observer_enabled_stages=("abc",))

    fusing_layers = [(f"layers.mlp_{i}", f"layers.NONE-mlp_{i}a") for i in range(3)]
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
@RunIf(quantization=True, sklearn=True)
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


def _get_observer_enabled(fake_quant: FakeQuantizeBase):
    # ``torch.quantization.FakeQuantize`` checks ``observer_enabled[0] == 1``.
    return fake_quant.observer_enabled[0] == 1


@pytest.mark.parametrize(
    "observer_enabled_stages",
    [("train", "validate", "test", "predict"), ("train",), ("validate",), ("test",), ("predict",), ()],
)
@RunIf(quantization=True)
def test_quantization_disable_observers(tmpdir, observer_enabled_stages):
    """Test disabling observers."""
    qmodel = RegressionModel()
    qcb = QuantizationAwareTraining(observer_enabled_stages=observer_enabled_stages)
    trainer = Trainer(callbacks=[qcb], default_root_dir=tmpdir)

    # Quantize qmodel.
    qcb.on_fit_start(trainer, qmodel)
    fake_quants = list(module for module in qmodel.modules() if isinstance(module, FakeQuantizeBase))
    # Disable some of observers before fitting.
    for fake_quant in fake_quants[::2]:
        fake_quant.disable_observer()

    for stage, on_stage_start, on_stage_end in [
        ("train", qcb.on_train_start, qcb.on_train_end),
        ("validate", qcb.on_validation_start, qcb.on_validation_end),
        ("test", qcb.on_test_start, qcb.on_test_end),
        ("predict", qcb.on_predict_start, qcb.on_predict_end),
    ]:
        before_stage_observer_enabled = torch.as_tensor(list(map(_get_observer_enabled, fake_quants)))

        on_stage_start(trainer, qmodel)
        expected_stage_observer_enabled = torch.as_tensor(
            before_stage_observer_enabled if stage in observer_enabled_stages else [False] * len(fake_quants)
        )
        assert torch.equal(
            torch.as_tensor(list(map(_get_observer_enabled, fake_quants))), expected_stage_observer_enabled
        )

        on_stage_end(trainer, qmodel)
        assert torch.equal(
            torch.as_tensor(list(map(_get_observer_enabled, fake_quants))), before_stage_observer_enabled
        )


@RunIf(quantization=True, sklearn=True)
def test_quantization_val_test_predict(tmpdir):
    """Test the default quantization aware training not affected by validating, testing and predicting."""
    seed_everything(42)
    num_features = 16
    dm = RegressDataModule(num_features=num_features)
    qmodel = RegressionModel()

    val_test_predict_qmodel = copy.deepcopy(qmodel)
    trainer = Trainer(
        callbacks=[QuantizationAwareTraining(quantize_on_fit_end=False)],
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        val_check_interval=1,
        num_sanity_val_steps=1,
        max_epochs=4,
    )
    trainer.fit(val_test_predict_qmodel, datamodule=dm)
    trainer.validate(model=val_test_predict_qmodel, datamodule=dm, verbose=False)
    trainer.test(model=val_test_predict_qmodel, datamodule=dm, verbose=False)
    trainer.predict(
        model=val_test_predict_qmodel, dataloaders=[torch.utils.data.DataLoader(RandomDataset(num_features, 16))]
    )

    expected_qmodel = copy.deepcopy(qmodel)
    # No validation in ``expected_qmodel`` fitting.
    Trainer(
        callbacks=[QuantizationAwareTraining(quantize_on_fit_end=False)],
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=4,
    ).fit(expected_qmodel, datamodule=dm)

    expected_state_dict = expected_qmodel.state_dict()
    for key, value in val_test_predict_qmodel.state_dict().items():
        expected_value = expected_state_dict[key]
        assert torch.allclose(value, expected_value)
