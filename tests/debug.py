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

    return model


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


def main():

    save_dir = init_save_dir()
    model = get_model()

    # exp file to get meta
    exp = get_exp(False)
    exp.save()

    # exp file to get weights
    checkpoint = ModelCheckpoint(save_dir)

    trainer = Trainer(
        checkpoint_callback=checkpoint,
        progress_bar=True,
        experiment=exp,
        max_nb_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus=[0, 1],
        distributed_backend='ddp',
        use_amp=True
    )

    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # load trained model
    pdb.set_trace()
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    trained_model = LightningTemplateModel.load_from_metrics(weights_path=save_dir, tags_csv=tags_path, on_gpu=True)

    # run prediction
    for batch in model.test_dataloader:
        break

    x, y = batch
    x = x.view(x.size(0), -1)

    y_hat = model(x)

    # acc
    labels_hat = torch.argmax(y_hat, dim=1)
    val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
    val_acc = torch.tensor(val_acc)

    print(val_acc)

    clear_save_dir()


if __name__ == '__main__':
    main()