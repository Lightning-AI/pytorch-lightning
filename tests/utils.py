import os
import shutil
import warnings
from argparse import Namespace

import numpy as np
import torch

from pl_examples import LightningTemplateModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.testing import (
    LightningTestModel,
)

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))
ROOT_SEED = 1234
torch.manual_seed(ROOT_SEED)
np.random.seed(ROOT_SEED)
RANDOM_SEEDS = list(np.random.randint(0, 10000, 1000))


def run_model_test_no_loggers(trainer_options, model, min_acc=0.50):
    save_dir = trainer_options['default_save_path']

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test model loading
    pretrained_model = load_model(trainer.logger.experiment,
                                  trainer.checkpoint_callback.filepath)

    # test new model accuracy
    for dataloader in model.test_dataloader():
        run_prediction(dataloader, pretrained_model, min_acc=min_acc)

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()


def run_model_test(trainer_options, model, on_gpu=True):
    save_dir = trainer_options['default_save_path']

    # logger file to get meta
    logger = get_test_tube_logger(save_dir, False)

    # logger file to get weights
    checkpoint = init_checkpoint_callback(logger)

    # add these to the trainer options
    trainer_options['checkpoint_callback'] = checkpoint
    trainer_options['logger'] = logger

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test model loading
    pretrained_model = load_model(logger.experiment, trainer.checkpoint_callback.filepath)

    # test new model accuracy
    [run_prediction(dataloader, pretrained_model) for dataloader in model.test_dataloader()]

    if trainer.use_ddp or trainer.use_ddp2:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()

    # test HPC loading / saving
    trainer.hpc_save(save_dir, logger)
    trainer.hpc_load(save_dir, on_gpu=on_gpu)


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
        'hidden_dim': 1000,
    }

    if continue_training:
        args['test_tube_do_checkpoint_load'] = True
        args['hpc_exp_number'] = hpc_exp_number

    hparams = Namespace(**args)
    return hparams


def get_model(use_test_model=False, lbfgs=False):
    # set up model with these hyperparams
    hparams = get_hparams()
    if lbfgs:
        setattr(hparams, 'optimizer_name', 'lbfgs')
        setattr(hparams, 'learning_rate', 0.002)

    if use_test_model:
        model = LightningTestModel(hparams)
    else:
        model = LightningTemplateModel(hparams)

    return model, hparams


def get_test_tube_logger(save_dir, debug=True, version=None):
    # set up logger object without actually saving logs
    logger = TestTubeLogger(save_dir, name='lightning_logs', debug=debug, version=version)
    return logger


def load_model(exp, root_weights_dir, module_class=LightningTemplateModel):
    # load trained model
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')

    checkpoints = [x for x in os.listdir(root_weights_dir) if '.ckpt' in x]
    weights_dir = os.path.join(root_weights_dir, checkpoints[0])

    trained_model = module_class.load_from_metrics(weights_path=weights_dir,
                                                   tags_csv=tags_path)

    assert trained_model is not None, 'loading model failed'

    return trained_model


def run_prediction(dataloader, trained_model, dp=False, min_acc=0.50):
    # run prediction on 1 batch
    for batch in dataloader:
        break

    x, y = batch
    x = x.view(x.size(0), -1)

    if dp:
        output = trained_model(batch, 0)
        acc = output['val_acc']
        acc = torch.mean(acc).item()

    else:
        y_hat = trained_model(x)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        acc = torch.tensor(acc)
        acc = acc.item()

    assert acc > min_acc, f'this model is expected to get > {min_acc} in test set (it got {acc})'


def assert_ok_val_acc(trainer):
    # this model should get 0.80+ acc
    acc = trainer.training_tqdm_dict['val_acc']
    assert acc > 0.50, f'model failed to get expected 0.50 validation accuracy. Got: {acc}'


def assert_ok_test_acc(trainer):
    # this model should get 0.80+ acc
    acc = trainer.training_tqdm_dict['test_acc']
    assert acc > 0.50, f'model failed to get expected 0.50 validation accuracy. Got: {acc}'


def can_run_gpu_test():
    if not torch.cuda.is_available():
        warnings.warn('test_multi_gpu_model_ddp cannot run.'
                      ' Rerun on a GPU node to run this test')
        return False
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_multi_gpu_model_ddp cannot run.'
                      ' Rerun on a node with 2+ GPUs to run this test')
        return False
    return True


def reset_seed():
    SEED = RANDOM_SEEDS.pop()
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def set_random_master_port():
    port = RANDOM_PORTS.pop()
    os.environ['MASTER_PORT'] = str(port)


def init_checkpoint_callback(logger):
    exp = logger.experiment
    exp_path = exp.get_data_path(exp.name, exp.version)
    ckpt_dir = os.path.join(exp_path, 'checkpoints')
    checkpoint = ModelCheckpoint(ckpt_dir)
    return checkpoint
