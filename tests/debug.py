from pytorch_lightning import Trainer
from examples import LightningTemplateModel
from pytorch_lightning.testing import LightningTestModel
from argparse import Namespace
from test_tube import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import shutil

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import pdb


class CoolModel(pl.LightningModule):

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

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(MNIST('path/to/save', train=True), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(MNIST('path/to/save', train=False), batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(MNIST('path/to/save', train=False), batch_size=32)


def get_model():
    # set up model with these hyperparams
    root_dir = os.path.dirname(os.path.realpath(__file__))
    hparams = Namespace(**{'drop_prob': 0.2,
                           'batch_size': 32,
                           'in_features': 28 * 28,
                           'learning_rate': 0.001 * 8,
                           'optimizer_name': 'adam',
                           'data_root': os.path.join(root_dir, 'mnist'),
                           'out_features': 10,
                           'hidden_dim': 1000})
    model = LightningTemplateModel(hparams)

    return model, hparams


def get_exp(debug=True, version=None):
    # set up exp object without actually saving logs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(root_dir, 'save_dir')
    exp = Experiment(debug=debug, save_dir=save_dir, name='tests_tt_dir', version=version)
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


def load_model(exp, save_dir, on_gpu, map_location=None, module_class=LightningTemplateModel):

    # load trained model
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')

    checkpoints = [x for x in os.listdir(save_dir) if '.ckpt' in x]
    weights_dir = os.path.join(save_dir, checkpoints[0])

    trained_model = module_class.load_from_metrics(weights_path=weights_dir,
                                                   tags_csv=tags_path,
                                                   on_gpu=on_gpu,
                                                   map_location=map_location)

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
    assert val_acc > 0.70, 'this model is expected to get > 0.7 in test set (it got %f)' % val_acc


# ------------------------------------------------------------------------
def run_gpu_model_test(trainer_options, model, hparams, on_gpu=True):
    save_dir = init_save_dir()

    # exp file to get meta
    exp = get_exp(False)
    exp.argparse(hparams)
    exp.save()

    # exp file to get weights
    checkpoint = ModelCheckpoint(save_dir)

    # add these to the trainer options
    trainer_options['checkpoint_callback'] = checkpoint
    trainer_options['experiment'] = exp

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test model loading
    pretrained_model = load_model(exp, save_dir, on_gpu)

    # test new model accuracy
    run_prediction(model.test_dataloader, pretrained_model)

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()

    # test HPC loading / saving
    trainer.hpc_save(save_dir, exp)
    trainer.hpc_load(save_dir, on_gpu=on_gpu)

    clear_save_dir()


def assert_ok_val_acc(trainer):
    # this model should get 0.80+ acc
    acc = trainer.tng_tqdm_dic['val_acc']
    assert acc > 0.50, f'model failed to get expected 0.50 validation accuracy. Got: {acc}'


def assert_ok_test_acc(trainer):
    # this model should get 0.80+ acc
    acc = trainer.tng_tqdm_dic['test_acc']
    assert acc > 0.50, f'model failed to get expected 0.50 validation accuracy. Got: {acc}'


def get_hparams(continue_training=False, hpc_exp_number=0):
    root_dir = os.path.dirname(os.path.realpath(__file__))

    args = {
        'drop_prob': 0.2,
        'batch_size': 32,
        'in_features': 28 * 28,
        'learning_rate': 0.001 * 8,
        'optimizer_name': 'adam',
        'data_root': os.path.join(root_dir, 'mnist'),
        'out_features': 10,
        'hidden_dim': 1000}

    if continue_training:
        args['test_tube_do_checkpoint_load'] = True
        args['hpc_exp_number'] = hpc_exp_number

    hparams = Namespace(**args)
    return hparams

def assert_same_weights(model_a, model_b):
    for (_, param), (_, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.all(torch.eq(param, param_b))

def main():
    """Verify test() on fitted model"""
    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
    exp = get_exp(False)
    exp.argparse(hparams)
    exp.save()

    # exp file to get weights
    checkpoint = ModelCheckpoint(save_dir)

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=1.0,
        val_percent_check=0.2,
        checkpoint_callback=checkpoint,
        experiment=exp
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    trainer.test()
    assert_ok_test_acc(trainer)

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = load_model(
        exp, save_dir, on_gpu=False, module_class=LightningTestModel
    )

    def check():
        pdb.set_trace()

    pretrained_model.on_pre_performance_check = check

    assert_same_weights(model, pretrained_model)

    pdb.set_trace()
    run_prediction(pretrained_model.test_dataloader, pretrained_model)

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    pdb.set_trace()
    run_prediction(pretrained_model.test_dataloader, pretrained_model)

    # test we have good test accuracy
    assert_ok_test_acc(new_trainer)
    clear_save_dir()


if __name__ == '__main__':
    main()
