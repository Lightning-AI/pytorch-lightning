import os

import numpy as np

# from pl_examples import LightningTemplateModel
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tests import TEMP_PATH, RANDOM_PORTS, RANDOM_SEEDS
from tests.base.model_template import EvalModelTemplate


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


def init_checkpoint_callback(logger):
    exp_path = logger.save_dir
    ckpt_dir = os.path.join(exp_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(ckpt_dir)
    return checkpoint
