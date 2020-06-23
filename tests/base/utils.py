import os

import numpy as np

# from pl_examples import LightningTemplateModel
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tests import TEMP_PATH, RANDOM_PORTS, RANDOM_SEEDS


def assert_speed_parity_relative(pl_times, pt_times, max_diff: float = 0.1):
    # assert speeds
    diffs = np.asarray(pl_times) - np.asarray(pt_times)
    # norm by vanila time
    diffs = diffs / np.asarray(pt_times)
    assert np.alltrue(diffs < max_diff), \
        f"lightning {diffs} was slower than PT (threshold {max_diff})"


def assert_speed_parity_absolute(pl_times, pt_times, nb_epochs, max_diff: float = 0.6):
    # assert speeds
    diffs = np.asarray(pl_times) - np.asarray(pt_times)
    # norm by vanila time
    diffs = diffs / nb_epochs
    assert np.alltrue(diffs < max_diff), \
        f"lightning {diffs} was slower than PT (threshold {max_diff})"


def get_default_logger(save_dir, version=None):
    # set up logger object without actually saving logs
    logger = TensorBoardLogger(save_dir, name='lightning_logs', version=version)
    return logger


def get_data_path(expt_logger, path_dir=None):
    # some calls contain only experiment not complete logger
    expt = expt_logger.experiment if hasattr(expt_logger, 'experiment') else expt_logger
    # each logger has to have these attributes
    name, version = expt_logger.name, expt_logger.version
    # only the test-tube experiment has such attribute
    if hasattr(expt, 'get_data_path'):
        return expt.get_data_path(name, version)
    # the other experiments...
    if not path_dir:
        if hasattr(expt_logger, 'save_dir') and expt_logger.save_dir:
            path_dir = expt_logger.save_dir
        else:
            path_dir = TEMP_PATH
    path_expt = os.path.join(path_dir, name, 'version_%s' % version)
    # try if the new sub-folder exists, typical case for test-tube
    if not os.path.isdir(path_expt):
        path_expt = path_dir
    return path_expt


def assert_ok_model_acc(trainer, key='test_acc', thr=0.5):
    # this model should get 0.80+ acc
    acc = trainer.progress_bar_dict[key]
    assert acc > thr, f"Model failed to get expected {thr} accuracy. {key} = {acc}"


def reset_seed():
    seed = RANDOM_SEEDS.pop()
    seed_everything(seed)


def set_random_master_port():
    reset_seed()
    port = RANDOM_PORTS.pop()
    os.environ['MASTER_PORT'] = str(port)


def init_checkpoint_callback(logger, path_dir=None):
    exp_path = get_data_path(logger, path_dir=path_dir)
    ckpt_dir = os.path.join(exp_path, 'checkpoints')
    os.mkdir(ckpt_dir)
    checkpoint = ModelCheckpoint(ckpt_dir)
    return checkpoint
