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
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.metrics.functional import mean_absolute_error
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class RandDataset(Dataset):

    def __init__(self, size=32, length=64, target_val=100.):
        self.len = length
        self.data = torch.randn(length, size)
        self.target_val = torch.tensor([target_val], dtype=self.data.dtype)

    def __getitem__(self, index):
        return self.data[index], self.target_val

    def __len__(self):
        return self.len


class RandDataModule(LightningDataModule):

    def __init__(self):
        super().__init__()
        self.random_train = RandDataset(32, 128)
        self.random_val = RandDataset(32, 64)
        self.random_test = RandDataset(32, 64)

    def train_dataloader(self):
        return DataLoader(self.random_train)

    def val_dataloader(self):
        return DataLoader(self.random_val)

    def test_dataloader(self):
        return DataLoader(self.random_test)


class LinearModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.layers[f"mlp_0"] = nn.Linear(32, 64)
        self.layers[f"mlp_0a"] = torch.nn.ReLU()
        for i in range(1, 3):
            self.layers[f"mlp_{i}"] = nn.Linear(64, 64)
            self.layers[f"mlp_{i}a"] = torch.nn.ReLU()
        self.layers["mlp_end"] = nn.Linear(64, 1)

    def forward(self, x):
        for n in sorted(self.layers):
            x = self.layers[n](x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=0.01)
        return [optimizer], []

    def measure(self, output, target):
        return mean_absolute_error(output, target).item()

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        loss = torch.nn.functional.mse_loss(output, y)
        self.log('train_MAE', self.measure(output, y), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        self.log('val_MAE', self.measure(output, y), prog_bar=True)


@pytest.mark.parametrize("observe", [None, 'histogram', 'average'])
def test_quantization(tmpdir, observe):
    """Parity for  quant model"""
    dm = RandDataModule()
    trainer_args = dict(
        default_root_dir=tmpdir,
        max_epochs=5,
    )

    model = LinearModel()
    Trainer(**trainer_args).fit(model, datamodule=dm)
    org_size = model.model_size
    org_mae = torch.mean(torch.tensor([model.measure(model(x), y) for x, y in dm.test_dataloader()]))

    qmodel = LinearModel()
    fusing_layers = [(f'layers.mlp_{i}', f'layers.mlp_{i}a') for i in range(3)]
    qcb = QuantizationAwareTraining(modules_to_fuse=fusing_layers, observer_type=observe)
    Trainer(callbacks=[qcb], **trainer_args).fit(qmodel, datamodule=dm)
    quant_calls = qcb._forward_calls
    quant_size = qmodel.model_size
    quant_mae = torch.mean(torch.tensor([model.measure(qmodel(x), y) for x, y in dm.test_dataloader()]))
    assert quant_calls == qcb._forward_calls

    # test that the trained model is smaller then initial
    size_ratio = quant_size / org_size
    assert size_ratio < 0.55
    # test that the test score is almost the same as with pure training
    diff_mae = abs(org_mae - quant_mae)
    assert diff_mae < 15


def test_quantization_exceptions(tmpdir):
    """Test  wrong fuse layers"""
    with pytest.raises(MisconfigurationException, match='Unsupported qconfig'):
        QuantizationAwareTraining(qconfig=['abc'])

    with pytest.raises(MisconfigurationException, match='Unsupported observer type'):
        QuantizationAwareTraining(observer_type='abc')

    with pytest.raises(MisconfigurationException, match='Unsupported `collect_quantization`'):
        QuantizationAwareTraining(collect_quantization='abc')

    with pytest.raises(MisconfigurationException, match='Unsupported `collect_quantization`'):
        QuantizationAwareTraining(collect_quantization=1.2)

    fusing_layers = [(f'layers.mlp_{i}', f'layers.NONE-mlp_{i}a') for i in range(3)]
    qcb = QuantizationAwareTraining(modules_to_fuse=fusing_layers)
    trainer = Trainer(callbacks=[qcb], default_root_dir=tmpdir, max_epochs=1)
    with pytest.raises(MisconfigurationException, match='one or more of them is not your model attributes'):
        trainer.fit(LinearModel(), datamodule=RandDataModule())


def custom_trigger_never(trainer):
    return False


def custom_trigger_even(trainer):
    return trainer.current_epoch % 2 == 0


def custom_trigger_last(trainer):
    return trainer.current_epoch == (trainer.max_epochs - 1)


@pytest.mark.parametrize(
    "trigger_fn,expected_count", [
        (None, 9),
        (3, 3),
        (custom_trigger_never, 0),
        (custom_trigger_even, 5),
        (custom_trigger_last, 2),
    ]
)
def test_quantization_triggers(tmpdir, trigger_fn, expected_count):
    """Test  how many times the quant is called"""
    dm = RandDataModule()
    qmodel = LinearModel()
    qcb = QuantizationAwareTraining(collect_quantization=trigger_fn)
    Trainer(
        callbacks=[qcb],
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=4,
    ).fit(qmodel, datamodule=dm)

    assert qcb._forward_calls == expected_count
