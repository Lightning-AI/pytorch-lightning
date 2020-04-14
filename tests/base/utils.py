import os
from argparse import Namespace

import numpy as np
import torch

# from pl_examples import LightningTemplateModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger, TensorBoardLogger
from tests.base import LightningTestModel
from tests.base.datasets import PATH_DATASETS

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))
ROOT_SEED = 1234
torch.manual_seed(ROOT_SEED)
np.random.seed(ROOT_SEED)
RANDOM_SEEDS = list(np.random.randint(0, 10000, 1000))
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


def assert_speed_parity(pl_times, pt_times, num_epochs):

    # assert speeds
    max_diff_per_epoch = 0.9
    pl_times = np.asarray(pl_times)
    pt_times = np.asarray(pt_times)
    diffs = pl_times - pt_times
    diffs = diffs / num_epochs

    assert np.alltrue(diffs < max_diff_per_epoch), \
        f"lightning was slower than PT (threshold {max_diff_per_epoch})"


def run_model_test_no_loggers(trainer_options, model, min_acc=0.50):
    # save_dir = trainer_options['default_root_dir']

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test model loading
    pretrained_model = load_model(trainer.logger,
                                  trainer.checkpoint_callback.dirpath,
                                  path_expt=trainer_options.get('default_root_dir'))

    # test new model accuracy
    test_loaders = model.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    for dataloader in test_loaders:
        run_prediction(dataloader, pretrained_model, min_acc=min_acc)

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()


def run_model_test(trainer_options, model, on_gpu=True):
    save_dir = trainer_options['default_root_dir']

    # logger file to get meta
    logger = get_default_testtube_logger(save_dir, False)

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
    pretrained_model = load_model(logger, trainer.checkpoint_callback.dirpath)

    # test new model accuracy
    test_loaders = model.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    [run_prediction(dataloader, pretrained_model) for dataloader in test_loaders]

    if trainer.use_ddp or trainer.use_ddp2:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers, trainer.optimizer_frequencies = \
            trainer.init_optimizers(pretrained_model)

    # test HPC loading / saving
    trainer.hpc_save(save_dir, logger)
    trainer.hpc_load(save_dir, on_gpu=on_gpu)


def get_default_hparams(continue_training=False, hpc_exp_number=0):
    _ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    args = {
        'drop_prob': 0.2,
        'batch_size': 32,
        'in_features': 28 * 28,
        'learning_rate': 0.001 * 8,
        'optimizer_name': 'adam',
        'data_root': PATH_DATASETS,
        'out_features': 10,
        'hidden_dim': 1000,
    }

    if continue_training:
        args['test_tube_do_checkpoint_load'] = True
        args['hpc_exp_number'] = hpc_exp_number

    hparams = Namespace(**args)
    return hparams


def get_default_model(lbfgs=False):
    # set up model with these hyperparams
    hparams = get_default_hparams()
    if lbfgs:
        setattr(hparams, 'optimizer_name', 'lbfgs')
        setattr(hparams, 'learning_rate', 0.002)

    model = LightningTestModel(hparams)

    return model, hparams


def get_default_testtube_logger(save_dir, debug=True, version=None):
    # set up logger object without actually saving logs
    logger = TestTubeLogger(save_dir, name='lightning_logs', debug=debug, version=version)
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
        path_dir = ROOT_PATH
    path_expt = os.path.join(path_dir, name, 'version_%s' % version)
    # try if the new sub-folder exists, typical case for test-tube
    if not os.path.isdir(path_expt):
        path_expt = path_dir
    return path_expt


def load_model(exp, root_weights_dir, module_class=LightningTestModel, path_expt=None):
    # load trained model
    path_expt_dir = get_data_path(exp, path_dir=path_expt)
    tags_path = os.path.join(path_expt_dir, TensorBoardLogger.NAME_CSV_TAGS)

    checkpoints = [x for x in os.listdir(root_weights_dir) if '.ckpt' in x]
    weights_dir = os.path.join(root_weights_dir, checkpoints[0])

    trained_model = module_class.load_from_checkpoint(
        checkpoint_path=weights_dir,
        tags_csv=tags_path
    )

    assert trained_model is not None, 'loading model failed'

    return trained_model


def load_model_from_checkpoint(root_weights_dir, module_class=LightningTestModel):
    # load trained model
    checkpoints = [x for x in os.listdir(root_weights_dir) if '.ckpt' in x]
    weights_dir = os.path.join(root_weights_dir, checkpoints[0])

    trained_model = module_class.load_from_checkpoint(
        checkpoint_path=weights_dir,
    )

    assert trained_model is not None, 'loading model failed'

    return trained_model


def run_prediction(dataloader, trained_model, dp=False, min_acc=0.5):
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

    assert acc >= min_acc, f"This model is expected to get > {min_acc} in test set (it got {acc})"


def assert_ok_model_acc(trainer, key='test_acc', thr=0.5):
    # this model should get 0.80+ acc
    acc = trainer.training_tqdm_dict[key]
    assert acc > thr, f"Model failed to get expected {thr} accuracy. {key} = {acc}"


def reset_seed():
    seed = RANDOM_SEEDS.pop()
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_random_master_port():
    port = RANDOM_PORTS.pop()
    os.environ['MASTER_PORT'] = str(port)


def init_checkpoint_callback(logger, path_dir=None):
    exp_path = get_data_path(logger, path_dir=path_dir)
    ckpt_dir = os.path.join(exp_path, 'checkpoints')
    os.mkdir(ckpt_dir)
    checkpoint = ModelCheckpoint(ckpt_dir)
    return checkpoint
