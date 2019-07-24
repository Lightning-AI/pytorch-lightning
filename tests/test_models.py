import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.examples.new_project_templates.lightning_module_template import LightningTemplateModel
from argparse import Namespace
from test_tube import Experiment
import os


def get_model():
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

def get_exp():
    exp = Experiment(debug=True)
    return exp

def test_cpu_model():
    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4
    )

    result = trainer.fit(model)

    assert result == 1


def test_single_gpu_model():
    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0]
    )

    result = trainer.fit(model)

    assert result == 1


def test_multi_gpu_model_dp():
    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0, 1]
    )

    result = trainer.fit(model)

    assert result == 1


def test_multi_gpu_model_ddp():
    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0, 1],
        distributed_backend='ddp'
    )

    result = trainer.fit(model)

    assert result == 1


if __name__ == '__main__':
    pytest.main([__file__])
