# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pickle
from argparse import Namespace

import cloudpickle
import pytest
import torch
from fsspec.implementations.local import LocalFileSystem
from omegaconf import OmegaConf, Container
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.core.saving import save_hparams_to_yaml, load_hparams_from_yaml
from pytorch_lightning.utilities import AttributeDict, is_picklable
from tests.base import EvalModelTemplate, TrialMNIST, BoringModel


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
def _run_standard_hparams_test(tmpdir, model, cls, try_overwrite=False):
    """
    Tests for the existence of an arg 'test_arg=14'
    """
    hparam_type = type(model.hparams)
    # test proper property assignments
    assert model.hparams.test_arg == 14

    # verify we can train
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, overfit_batches=2)
    trainer.fit(model)

    # make sure the raw checkpoint saved the properties
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]['test_arg'] == 14

    # verify that model loads correctly
    model2 = cls.load_from_checkpoint(raw_checkpoint_path)
    assert model2.hparams.test_arg == 14

    assert isinstance(model2.hparams, hparam_type)

    if try_overwrite:
        # verify that we can overwrite the property
        model3 = cls.load_from_checkpoint(raw_checkpoint_path, test_arg=78)
        assert model3.hparams.test_arg == 78

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
    assert isinstance(model.hparams, Container)

    # run standard test suite
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, cls)
    model2 = cls.load_from_checkpoint(raw_checkpoint_path)
    assert isinstance(model2.hparams, Container)

    # config specific tests
    assert model2.hparams.test_arg == 14
    assert model2.hparams.mylist[0] == 15.4


def test_explicit_args_hparams(tmpdir):
    """
    Tests that a model can take implicit args and assign
    """

    # define model
    class LocalModel(EvalModelTemplate):
        def __init__(self, test_arg, test_arg2):
            super().__init__()
            self.save_hyperparameters('test_arg', 'test_arg2')

    model = LocalModel(test_arg=14, test_arg2=90)

    # run standard test suite
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, LocalModel)
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=120)

    # config specific tests
    assert model.hparams.test_arg2 == 120


def test_implicit_args_hparams(tmpdir):
    """
    Tests that a model can take regular args and assign
    """

    # define model
    class LocalModel(EvalModelTemplate):
        def __init__(self, test_arg, test_arg2):
            super().__init__()
            self.save_hyperparameters()

    model = LocalModel(test_arg=14, test_arg2=90)

    # run standard test suite
    raw_checkpoint_path = _run_standard_hparams_test(tmpdir, model, LocalModel)
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=120)

    # config specific tests
    assert model.hparams.test_arg2 == 120


def test_explicit_missing_args_hparams(tmpdir):
    """
    Tests that a model can take regular args and assign
    """

    # define model
    class LocalModel(EvalModelTemplate):
        def __init__(self, test_arg, test_arg2):
            super().__init__()
            self.save_hyperparameters('test_arg')

    model = LocalModel(test_arg=14, test_arg2=90)

    # test proper property assignments
    assert model.hparams.test_arg == 14

    # verify we can train
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_batches=0.5)
    trainer.fit(model)

    # make sure the raw checkpoint saved the properties
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]['test_arg'] == 14

    # verify that model loads correctly
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, test_arg2=123)
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


class SubClassEvalModel(EvalModelTemplate):
    any_other_loss = torch.nn.CrossEntropyLoss()

    def __init__(self, *args, subclass_arg=1200, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()


class SubSubClassEvalModel(SubClassEvalModel):
    pass


class AggSubClassEvalModel(SubClassEvalModel):

    def __init__(self, *args, my_loss=torch.nn.CrossEntropyLoss(), **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()


class UnconventionalArgsEvalModel(EvalModelTemplate):
    """ A model that has unconventional names for "self", "*args" and "**kwargs". """

    def __init__(obj, *more_args, other_arg=300, **more_kwargs):
        # intentionally named obj
        super().__init__(*more_args, **more_kwargs)
        obj.save_hyperparameters()


class DictConfSubClassEvalModel(SubClassEvalModel):
    def __init__(self, *args, dict_conf=OmegaConf.create(dict(my_param='something')), **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()


@pytest.mark.parametrize("cls", [
    EvalModelTemplate,
    SubClassEvalModel,
    SubSubClassEvalModel,
    AggSubClassEvalModel,
    UnconventionalArgsEvalModel,
    DictConfSubClassEvalModel,
])
def test_collect_init_arguments(tmpdir, cls):
    """ Test that the model automatically saves the arguments passed into the constructor """
    extra_args = {}
    if cls is AggSubClassEvalModel:
        extra_args.update(my_loss=torch.nn.CosineEmbeddingLoss())
    elif cls is DictConfSubClassEvalModel:
        extra_args.update(dict_conf=OmegaConf.create(dict(my_param='anything')))

    model = cls(**extra_args)
    assert model.hparams.batch_size == 32
    model = cls(batch_size=179, **extra_args)
    assert model.hparams.batch_size == 179

    if isinstance(model, SubClassEvalModel):
        assert model.hparams.subclass_arg == 1200

    if isinstance(model, AggSubClassEvalModel):
        assert isinstance(model.hparams.my_loss, torch.nn.CosineEmbeddingLoss)

    # verify that the checkpoint saved the correct values
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_batches=0.5)
    trainer.fit(model)

    raw_checkpoint_path = _raw_checkpoint_path(trainer)

    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in raw_checkpoint
    assert raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]['batch_size'] == 179

    # verify that model loads correctly
    model = cls.load_from_checkpoint(raw_checkpoint_path)
    assert model.hparams.batch_size == 179

    if isinstance(model, AggSubClassEvalModel):
        assert isinstance(model.hparams.my_loss, torch.nn.CosineEmbeddingLoss)

    if isinstance(model, DictConfSubClassEvalModel):
        assert isinstance(model.hparams.dict_conf, Container)
        assert model.hparams.dict_conf['my_param'] == 'anything'

    # verify that we can overwrite whatever we want
    model = cls.load_from_checkpoint(raw_checkpoint_path, batch_size=99)
    assert model.hparams.batch_size == 99


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
#     trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_batches=0.5)
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


@pytest.mark.parametrize("past_key", ['module_arguments'])
def test_load_past_checkpoint(tmpdir, past_key):
    model = EvalModelTemplate()

    # verify we can train
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)

    # make sure the raw checkpoint saved the properties
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    raw_checkpoint[past_key] = raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
    raw_checkpoint['hparams_type'] = 'Namespace'
    raw_checkpoint[past_key]['batch_size'] = -17
    del raw_checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
    # save back the checkpoint
    torch.save(raw_checkpoint, raw_checkpoint_path)

    # verify that model loads correctly
    model2 = EvalModelTemplate.load_from_checkpoint(raw_checkpoint_path)
    assert model2.hparams.batch_size == -17


def test_hparams_pickle(tmpdir):
    ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    pkl = pickle.dumps(ad)
    assert ad == pickle.loads(pkl)
    pkl = cloudpickle.dumps(ad)
    assert ad == pickle.loads(pkl)


class UnpickleableArgsEvalModel(EvalModelTemplate):
    """ A model that has an attribute that cannot be pickled. """

    def __init__(self, foo='bar', pickle_me=(lambda x: x + 1), **kwargs):
        super().__init__(**kwargs)
        assert not is_picklable(pickle_me)
        self.save_hyperparameters()


def test_hparams_pickle_warning(tmpdir):
    model = UnpickleableArgsEvalModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    with pytest.warns(UserWarning, match="attribute 'pickle_me' removed from hparams because it cannot be pickled"):
        trainer.fit(model)
    assert 'pickle_me' not in model.hparams


def test_hparams_save_yaml(tmpdir):
    hparams = dict(batch_size=32, learning_rate=0.001, data_root='./any/path/here',
                   nasted=dict(any_num=123, anystr='abcd'))
    path_yaml = os.path.join(tmpdir, 'testing-hparams.yaml')

    save_hparams_to_yaml(path_yaml, hparams)
    assert load_hparams_from_yaml(path_yaml) == hparams

    save_hparams_to_yaml(path_yaml, Namespace(**hparams))
    assert load_hparams_from_yaml(path_yaml) == hparams

    save_hparams_to_yaml(path_yaml, AttributeDict(hparams))
    assert load_hparams_from_yaml(path_yaml) == hparams

    save_hparams_to_yaml(path_yaml, OmegaConf.create(hparams))
    assert load_hparams_from_yaml(path_yaml) == hparams


class NoArgsSubClassEvalModel(EvalModelTemplate):
    def __init__(self):
        super().__init__()


class SimpleNoArgsModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def test_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


@pytest.mark.parametrize("cls", [
    SimpleNoArgsModel,
    NoArgsSubClassEvalModel,
])
def test_model_nohparams_train_test(tmpdir, cls):
    """Test models that do not tae any argument in init."""

    model = cls()
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
    )

    train_loader = DataLoader(TrialMNIST(os.getcwd(), train=True, download=True), batch_size=32)
    trainer.fit(model, train_loader)

    test_loader = DataLoader(TrialMNIST(os.getcwd(), train=False, download=True), batch_size=32)
    trainer.test(test_dataloaders=test_loader)


def test_model_ignores_non_exist_kwargument(tmpdir):
    """Test that the model takes only valid class arguments."""

    class LocalModel(EvalModelTemplate):
        def __init__(self, batch_size=15):
            super().__init__(batch_size=batch_size)
            self.save_hyperparameters()

    model = LocalModel()
    assert model.hparams.batch_size == 15

    # verify that the checkpoint saved the correct values
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)

    # verify that we can overwrite whatever we want
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    model = LocalModel.load_from_checkpoint(raw_checkpoint_path, non_exist_kwarg=99)
    assert 'non_exist_kwarg' not in model.hparams


class SuperClassPositionalArgs(EvalModelTemplate):

    def __init__(self, hparams):
        super().__init__()
        self._hparams = None  # pretend EvalModelTemplate did not call self.save_hyperparameters()
        self.hparams = hparams


class SubClassVarArgs(SuperClassPositionalArgs):
    """ Loading this model should accept hparams and init in the super class """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_args(tmpdir):
    """ Test for inheritance: super class takes positional arg, subclass takes varargs. """
    hparams = dict(test=1)
    model = SubClassVarArgs(hparams)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    trainer.fit(model)

    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    with pytest.raises(TypeError, match=r"__init__\(\) got an unexpected keyword argument 'test'"):
        SubClassVarArgs.load_from_checkpoint(raw_checkpoint_path)


class RuntimeParamChangeModelSaving(BoringModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()


class RuntimeParamChangeModelAssign(BoringModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = kwargs


@pytest.mark.parametrize("cls", [RuntimeParamChangeModelSaving, RuntimeParamChangeModelAssign])
def test_init_arg_with_runtime_change(tmpdir, cls):
    """Test that we save/export only the initial hparams, no other runtime change allowed"""
    model = cls(running_arg=123)
    assert model.hparams.running_arg == 123
    model.hparams.running_arg = -1
    assert model.hparams.running_arg == -1
    model.hparams = Namespace(abc=42)
    assert model.hparams.abc == 42

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
    )
    trainer.fit(model)

    path_yaml = os.path.join(trainer.logger.log_dir, trainer.logger.NAME_HPARAMS_FILE)
    hparams = load_hparams_from_yaml(path_yaml)
    assert hparams.get('running_arg') == 123


class UnsafeParamModel(BoringModel):
    def __init__(self, my_path, any_param=123):
        super().__init__()
        self.save_hyperparameters()


def test_model_with_fsspec_as_parameter(tmpdir):
    model = UnsafeParamModel(LocalFileSystem(tmpdir))
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
    )
    trainer.fit(model)
    trainer.test()
