import os
import sys
from argparse import Namespace

import pytest
import torch
from omegaconf import OmegaConf, Container

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities import AttributeDict
from tests.base import EvalModelTemplate
import pickle
import cloudpickle


class SaveHparamsModel(EvalModelTemplate):
    """ Tests that a model can take an object """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)


class AssignHparamsModel(EvalModelTemplate):
    """ Tests that a model can take an object with explicit setter """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams


# -------------------------
# STANDARD TESTS
# -------------------------
def _run_standard_hparams_test(tmpdir, model, cls):
    """
    Tests for the existence of an arg 'test_arg=14'
    """
    # test proper property assignments
    assert model.hparams.test_arg == 14

    # verify we can train
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_pct=0.5)
    trainer.fit(model)

    # make sure the raw checkpoint saved the properties
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]['test_arg'] == 14

    # verify that model loads correctly
    model = cls.load_from_checkpoint(raw_checkpoint_path)
    assert model.hparams.test_arg == 14

    # todo
    # verify that we can overwrite the property
    # model = cls.load_from_checkpoint(raw_checkpoint_path, test_arg=78)
    # assert model.hparams.test_arg == 78

    return raw_checkpoint_path


@pytest.mark.parametrize("cls", [SaveHparamsModel, AssignHparamsModel])
def test_namespace_hparams(tmpdir, cls):
    # init model
    model = cls(hparams=Namespace(test_arg=14))

    # run standard test suite
    _run_standard_hparams_test(tmpdir, model, cls)


@pytest.mark.parametrize("cls", [SaveHparamsModel, AssignHparamsModel])
def test_dict_hparams(tmpdir, cls):
    # init model
    model = cls(hparams={'test_arg': 14})

    # run standard test suite
    _run_standard_hparams_test(tmpdir, model, cls)


@pytest.mark.parametrize("cls", [SaveHparamsModel, AssignHparamsModel])
def test_omega_conf_hparams(tmpdir, cls):
    # init model
    conf = OmegaConf.create(dict(test_arg=14, mylist=[15.4, dict(a=1, b=2)]))
    model = cls(hparams=conf)

    # run standard test suite
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, cls)
    model = cls.load_from_checkpoint(raw_checkpoint_path)

    # config specific tests
    assert model.hparams.test_arg == 14
    assert model.hparams.mylist[0] == 15.4


def test_explicit_args_hparams(tmpdir):
    """
    Tests that a model can take implicit args and assign
    """

    # define model
    class TestModel(EvalModelTemplate):
        def __init__(self, test_arg, test_arg2):
            super().__init__()
            self.save_hyperparameters('test_arg', 'test_arg2')

    model = TestModel(test_arg=14, test_arg2=90)

    # run standard test suite
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, TestModel)
    model = TestModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=120)

    # config specific tests
    assert model.hparams.test_arg2 == 120


def test_implicit_args_hparams(tmpdir):
    """
    Tests that a model can take regular args and assign
    """

    # define model
    class TestModel(EvalModelTemplate):
        def __init__(self, test_arg, test_arg2):
            super().__init__()
            self.save_hyperparameters()

    model = TestModel(test_arg=14, test_arg2=90)

    # run standard test suite
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, TestModel)
    model = TestModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=120)

    # config specific tests
    assert model.hparams.test_arg2 == 120


def test_explicit_missing_args_hparams(tmpdir):
    """
    Tests that a model can take regular args and assign
    """

    # define model
    class TestModel(EvalModelTemplate):
        def __init__(self, test_arg, test_arg2):
            super().__init__()
            self.save_hyperparameters('test_arg')

    model = TestModel(test_arg=14, test_arg2=90)

    # test proper property assignments
    assert model.hparams.test_arg == 14

    # verify we can train
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_pct=0.5)
    trainer.fit(model)

    # make sure the raw checkpoint saved the properties
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]['test_arg'] == 14

    # verify that model loads correctly
    model = TestModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=123)
    assert model.hparams.test_arg == 14
    assert 'test_arg2' not in model.hparams  # test_arg2 is not registered in class init

    return raw_checkpoint_path

# -------------------------
# SPECIFIC TESTS
# -------------------------


def test_class_nesting():

    class MyModule(LightningModule):
        def forward(self):
            ...

    # make sure PL modules are always nn.Module
    a = MyModule()
    assert isinstance(a, torch.nn.Module)

    def test_outside():
        a = MyModule()
        _ = a.hparams

    class A:
        def test(self):
            a = MyModule()
            _ = a.hparams

        def test2(self):
            test_outside()

    test_outside()
    A().test2()
    A().test()


@pytest.mark.xfail(sys.version_info >= (3, 6), reason='OmegaConf only for Python >= 3.8')
def test_omegaconf(tmpdir):
    class OmegaConfModel(EvalModelTemplate):
        def __init__(self, ogc):
            super().__init__()
            self.ogc = ogc
            self.size = ogc.list[0]

    conf = OmegaConf.create({"k": "v", "list": [15.4, {"a": "1", "b": "2"}]})
    model = OmegaConfModel(conf)

    # ensure ogc passed values correctly
    assert model.size == 15.4

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_pct=0.5)
    result = trainer.fit(model)

    assert result == 1


# class SubClassEvalModel(EvalModelTemplate):
#     any_other_loss = torch.nn.CrossEntropyLoss()
#
#     def __init__(self, *args, subclass_arg=1200, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.save_hyperparameters()
#
#
# class SubSubClassEvalModel(SubClassEvalModel):
#     pass
#
#
# class AggSubClassEvalModel(SubClassEvalModel):
#
#     def __init__(self, *args, my_loss=torch.nn.CrossEntropyLoss(), **kwargs):
#         super().__init__(*args, **kwargs)
#         self.save_hyperparameters()
#
#
# class UnconventionalArgsEvalModel(EvalModelTemplate):
#     """ A model that has unconventional names for "self", "*args" and "**kwargs". """
#
#     def __init__(obj, *more_args, other_arg=300, **more_kwargs):
#         # intentionally named obj
#         super().__init__(*more_args, **more_kwargs)
#         obj.save_hyperparameters()
#
#
# class DictConfSubClassEvalModel(SubClassEvalModel):
#     def __init__(self, *args, dict_conf=OmegaConf.create(dict(my_param='something')), **kwargs):
#         super().__init__(*args, **kwargs)
#         self.save_hyperparameters()
#
#
# @pytest.mark.parametrize("cls", [
#     EvalModelTemplate,
#     SubClassEvalModel,
#     SubSubClassEvalModel,
#     AggSubClassEvalModel,
#     UnconventionalArgsEvalModel,
#     DictConfSubClassEvalModel,
# ])
# def test_collect_init_arguments(tmpdir, cls):
#     """ Test that the model automatically saves the arguments passed into the constructor """
#     extra_args = {}
#     if cls is AggSubClassEvalModel:
#         extra_args.update(my_loss=torch.nn.CosineEmbeddingLoss())
#     elif cls is DictConfSubClassEvalModel:
#         extra_args.update(dict_conf=OmegaConf.create(dict(my_param='anything')))
#
#     model = cls(**extra_args)
#     assert model.batch_size == 32
#     model = cls(batch_size=179, **extra_args)
#     assert model.batch_size == 179
#
#     if isinstance(model, SubClassEvalModel):
#         assert model.subclass_arg == 1200
#
#     if isinstance(model, AggSubClassEvalModel):
#         assert isinstance(model.my_loss, torch.nn.CosineEmbeddingLoss)
#
#     # verify that the checkpoint saved the correct values
#     trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_pct=0.5)
#     trainer.fit(model)
#     raw_checkpoint_path = _raw_checkpoint_path(trainer)
#
#     raw_checkpoint = torch.load(raw_checkpoint_path)
#     assert LightningModule.CHECKPOINT_KEY_HYPER_PARAMS in raw_checkpoint
#     assert raw_checkpoint[LightningModule.CHECKPOINT_KEY_HYPER_PARAMS]['batch_size'] == 179
#
#     # verify that model loads correctly
#     model = cls.load_from_checkpoint(raw_checkpoint_path)
#     assert model.batch_size == 179
#
#     if isinstance(model, AggSubClassEvalModel):
#         assert isinstance(model.my_loss, torch.nn.CrossEntropyLoss)
#
#     if isinstance(model, DictConfSubClassEvalModel):
#         assert isinstance(model.dict_conf, DictConfig)
#         assert model.dict_conf == 'anything'
#
#     # verify that we can overwrite whatever we want
#     model = cls.load_from_checkpoint(raw_checkpoint_path, batch_size=99)
#     assert model.batch_size == 99


def _raw_checkpoint_path(trainer) -> str:
    raw_checkpoint_paths = os.listdir(trainer.checkpoint_callback.dirpath)
    raw_checkpoint_paths = [x for x in raw_checkpoint_paths if '.ckpt' in x]
    assert raw_checkpoint_paths
    raw_checkpoint_path = raw_checkpoint_paths[0]
    raw_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, raw_checkpoint_path)
    return raw_checkpoint_path


class LocalVariableModelSuperLast(EvalModelTemplate):
    """ This model has the super().__init__() call at the end. """

    def __init__(self, arg1, arg2, *args, **kwargs):
        self.argument1 = arg1  # arg2 intentionally not set
        arg1 = 'overwritten'
        local_var = 1234
        super().__init__(*args, **kwargs)  # this is intentionally here at the end


class LocalVariableModelSuperFirst(EvalModelTemplate):
    """ This model has the _auto_collect_arguments() call at the end. """

    def __init__(self, arg1, arg2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.argument1 = arg1  # arg2 intentionally not set
        arg1 = 'overwritten'
        local_var = 1234
        self.save_hyperparameters()  # this is intentionally here at the end


@pytest.mark.parametrize("cls", [
    LocalVariableModelSuperFirst,
    # LocalVariableModelSuperLast,
])
def test_collect_init_arguments_with_local_vars(cls):
    """ Tests that only the arguments are collected and not local variables. """
    model = cls(arg1=1, arg2=2)
    assert 'local_var' not in model.hparams
    assert model.hparams['arg1'] == 'overwritten'
    assert model.hparams['arg2'] == 2


# @pytest.mark.parametrize("cls,config", [
#     (SaveHparamsModel, Namespace(my_arg=42)),
#     (SaveHparamsModel, dict(my_arg=42)),
#     (SaveHparamsModel, OmegaConf.create(dict(my_arg=42))),
#     (AssignHparamsModel, Namespace(my_arg=42)),
#     (AssignHparamsModel, dict(my_arg=42)),
#     (AssignHparamsModel, OmegaConf.create(dict(my_arg=42))),
# ])
# def test_single_config_models(tmpdir, cls, config):
#     """ Test that the model automatically saves the arguments passed into the constructor """
#     model = cls(config)
#
#     # no matter how you do it, it should be assigned
#     assert model.hparams.my_arg == 42
#
#     # verify that the checkpoint saved the correct values
#     trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_pct=0.5)
#     trainer.fit(model)
#
#     # verify that model loads correctly
#     raw_checkpoint_path = _raw_checkpoint_path(trainer)
#     model = cls.load_from_checkpoint(raw_checkpoint_path)
#     assert model.hparams.my_arg == 42


class AnotherArgModel(EvalModelTemplate):
    def __init__(self, arg1):
        super().__init__()
        self.save_hyperparameters(arg1)


class OtherArgsModel(EvalModelTemplate):
    def __init__(self, arg1, arg2):
        super().__init__()
        self.save_hyperparameters(arg1, arg2)


@pytest.mark.parametrize("cls,config", [
    (AnotherArgModel, dict(arg1=42)),
    (OtherArgsModel, dict(arg1=3.14, arg2='abc')),
])
def test_single_config_models_fail(tmpdir, cls, config):
    """ Test fail on passing unsupported config type. """
    with pytest.raises(ValueError):
        _ = cls(**config)


def test_hparams_pickle(tmpdir):
    ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    pkl = pickle.dumps(ad)
    assert ad == pickle.loads(pkl)
    pkl = cloudpickle.dumps(ad)
    assert ad == pickle.loads(pkl)
