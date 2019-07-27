import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.examples.new_project_templates.lightning_module_template import LightningTemplateModel
from argparse import Namespace
from test_tube import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import warnings
import torch
import os
import shutil
import pdb

import pytorch_lightning as ptl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class CoolModel(ptl.LightningModule):

    def __init(self):
        super(CoolModel, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x))

    def my_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'tng_loss': self.my_loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.my_loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs['val_loss']]).mean()
        return avg_loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @ptl.data_loader
    def tng_dataloader(self):
        return DataLoader(MNIST('path/to/save', train=True), batch_size=32)

    @ptl.data_loader
    def val_dataloader(self):
        return DataLoader(MNIST('path/to/save', train=False), batch_size=32)

    @ptl.data_loader
    def test_dataloader(self):
        return DataLoader(MNIST('path/to/save', train=False), batch_size=32)


def get_model():
    # set up model with these hyperparams
    root_dir = os.path.dirname(os.path.realpath(__file__))
    hparams = Namespace(**{'drop_prob': 0.2,
                           'batch_size': 32,
                           'in_features': 28*28,
                           'learning_rate': 0.001*8,
                           'optimizer_name': 'adam',
                           'data_root': os.path.join(root_dir, 'mnist'),
                           'out_features': 10,
                           'hidden_dim': 1000})
    model = LightningTemplateModel(hparams)

    return model, hparams


def get_exp(debug=True):
    # set up exp object without actually saving logs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    exp = Experiment(debug=debug, save_dir=root_dir, name='tests_tt_dir')
    return exp


def init_save_dir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(root_dir, 'save_dir')

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def clear_save_dir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(root_dir, 'save_dir')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


def load_model(exp, save_dir):

    # load trained model
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')

    checkpoints = [x for x in os.listdir(save_dir) if '.ckpt' in x]
    weights_dir = os.path.join(save_dir, checkpoints[0])

    trained_model = LightningTemplateModel.load_from_metrics(weights_path=weights_dir, tags_csv=tags_path, on_gpu=True)

    assert trained_model is not None, 'loading model failed'

    return trained_model


def run_prediction(dataloader, trained_model):
    # run prediction on 1 batch
    for batch in dataloader:
        break

    x, y = batch
    x = x.view(x.size(0), -1)

    y_hat = trained_model(x)

    # acc
    labels_hat = torch.argmax(y_hat, dim=1)
    val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
    val_acc = torch.tensor(val_acc)
    val_acc = val_acc.item()

    print(val_acc)

    assert val_acc > 0.70, f'this model is expected to get > 0.7 in test set (it got {val_acc})'


def main():

    save_dir = init_save_dir()

    # exp file to get meta
    exp = get_exp(False)
    exp.save()

    # exp file to get weights
    checkpoint = ModelCheckpoint(save_dir)

    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        progress_bar=True,
        max_nb_epochs=1,
        gpus=[0, 1],
        distributed_backend='dp',
    )

    model = CoolModel()

    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test model loading
    pretrained_model = load_model(exp, save_dir)

    # test model preds
    run_prediction(model.test_dataloader, pretrained_model)

    clear_save_dir()


if __name__ == '__main__':
    main()
