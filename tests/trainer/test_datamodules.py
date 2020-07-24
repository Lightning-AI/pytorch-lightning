import pickle
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from tests.base.datamodules import MNISTDataModule


def test_base_datamodule(tmpdir):
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()


def test_dm_add_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', './my_data'])
    assert args.data_dir == './my_data'


def test_dm_init_from_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', './my_data'])
    dm = MNISTDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()


def test_dm_pickle_after_init(tmpdir):
    dm = MNISTDataModule()
    pickle.dumps(dm)


def test_dm_pickle_after_setup(tmpdir):
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()
    pickle.dumps(dm)
