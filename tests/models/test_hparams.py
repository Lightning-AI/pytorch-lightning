import os
import sys
from argparse import Namespace

import pytest
import torch
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.core.lightning import CHECKPOINT_KEY_MODULE_ARGS
from tests.base import EvalModelTemplate


def test_class_nesting():

    class MyModule(LightningModule):
        def forward(self):
            return 0

    # make sure PL modules are always nn.Module
    a = MyModule()
    assert isinstance(a, torch.nn.Module)

    def test_outside():
        a = MyModule()
        print(a.module_arguments)

    class A:
        def test(self):
            a = MyModule()
            print(a.module_arguments)

        def test2(self):
            test_outside()

    test_outside()
    A().test2()
    A().test()


@pytest.mark.xfail(sys.version_info >= (3, 8), reason='OmegaConf only for Python >= 3.8')
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


class SubClassEvalModel(EvalModelTemplate):
    any_other_loss = torch.nn.CrossEntropyLoss()

    def __init__(self, *args, subclass_arg=1200, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()


class SubSubClassEvalModel(SubClassEvalModel):
    pass


class UnconventionalArgsEvalModel(EvalModelTemplate):
    """ A model that has unconventional names for "self", "*args" and "**kwargs". """

    def __init__(obj, *more_args, other_arg=300, **more_kwargs):
        # intentionally named obj
        super().__init__(*more_args, **more_kwargs)
        obj.save_hyperparameters()
        other_arg = 321


class AggSubClassEvalModel(SubClassEvalModel):

    def __init__(self, *args, my_loss=torch.nn.CrossEntropyLoss(), **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()


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
    assert model.batch_size == 32
    model = cls(batch_size=179, **extra_args)
    assert model.batch_size == 179

    if isinstance(model, SubClassEvalModel):
        assert model.subclass_arg == 1200

    if isinstance(model, AggSubClassEvalModel):
        assert isinstance(model.my_loss, torch.nn.CosineEmbeddingLoss)

    # verify that the checkpoint saved the correct values
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_pct=0.5)
    trainer.fit(model)
    raw_checkpoint_path = _raw_checkpoint_path(trainer)

    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert CHECKPOINT_KEY_MODULE_ARGS in raw_checkpoint
    assert raw_checkpoint[CHECKPOINT_KEY_MODULE_ARGS]['batch_size'] == 179

    # verify that model loads correctly
    model = cls.load_from_checkpoint(raw_checkpoint_path)
    assert model.batch_size == 179

    if isinstance(model, AggSubClassEvalModel):
        assert isinstance(model.my_loss, torch.nn.CrossEntropyLoss)

    if isinstance(model, DictConfSubClassEvalModel):
        assert isinstance(model.dict_conf, DictConfig)
        assert model.dict_conf == 'anything'

    # verify that we can overwrite whatever we want
    model = cls.load_from_checkpoint(raw_checkpoint_path, batch_size=99)
    assert model.batch_size == 99


def _raw_checkpoint_path(trainer) -> str:
    raw_checkpoint_paths = os.listdir(trainer.checkpoint_callback.dirpath)
    raw_checkpoint_path = [x for x in raw_checkpoint_paths if '.ckpt' in x][0]
    raw_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, raw_checkpoint_path)
    return raw_checkpoint_path


class LocalVariableModel1(EvalModelTemplate):
    """ This model has the super().__init__() call at the end. """

    def __init__(self, arg1, arg2, *args, **kwargs):
        self.argument1 = arg1  # arg2 intentionally not set
        arg1 = 'overwritten'
        local_var = 1234
        super().__init__(*args, **kwargs)  # this is intentionally here at the end


class LocalVariableModel2(EvalModelTemplate):
    """ This model has the _auto_collect_arguments() call at the end. """

    def __init__(self, arg1, arg2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.argument1 = arg1  # arg2 intentionally not set
        arg1 = 'overwritten'
        local_var = 1234
        self.save_hyperparameters()  # this is intentionally here at the end


@pytest.mark.parametrize("cls", [
    LocalVariableModel1,
    LocalVariableModel2,
])
def test_collect_init_arguments_with_local_vars(cls):
    """ Tests that only the arguments are collected and not local variables. """
    model = cls(arg1=1, arg2=2)
    assert 'local_var' not in model.module_arguments
    assert model.module_arguments['arg1'] == 'overwritten'
    assert model.module_arguments['arg2'] == 2


class NamespaceArgModel(EvalModelTemplate):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)


class DictArgModel(EvalModelTemplate):
    def __init__(self, some_dict: dict):
        super().__init__()
        self.save_hyperparameters(some_dict)


class OmegaConfArgModel(EvalModelTemplate):
    def __init__(self, conf: OmegaConf):
        super().__init__()
        self.save_hyperparameters(conf)


@pytest.mark.parametrize("cls,config", [
    (NamespaceArgModel, Namespace(my_arg=42)),
    (DictArgModel, dict(my_arg=42)),
    (OmegaConfArgModel, OmegaConf.create(dict(my_arg=42))),
])
def test_single_config_models(tmpdir, cls, config):
    """ Test that the model automatically saves the arguments passed into the constructor """
    model = cls(config)

    # verify that the checkpoint saved the correct values
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, overfit_pct=0.5)
    trainer.fit(model)

    # verify that model loads correctly
    raw_checkpoint_path = _raw_checkpoint_path(trainer)
    model = cls.load_from_checkpoint(raw_checkpoint_path)
    assert model.module_arguments == config


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
    (OtherArgsModel, dict(arg1=42, arg2='abc')),
])
def test_single_config_models_fail(tmpdir, cls, config):
    """ Test fail on passing unsupported config type. """
    with pytest.raises(ValueError):
        _ = cls(**config)
