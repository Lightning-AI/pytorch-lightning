import os
from argparse import Namespace

import pytest
import torch

from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


class SubClassEvalModel(EvalModelTemplate):
    object_that_should_not_be_saved = torch.nn.CrossEntropyLoss()

    def __init__(self, *args, subclass_arg=1200, **kwargs):
        super().__init__()


class HparamsNamespaceEvalModel(EvalModelTemplate):
    def __init__(self, *args, hparams=Namespace(hparam_arg=123), **kwargs):
        super().__init__()


class HparamsDictEvalModel(EvalModelTemplate):
    def __init__(self, *args, hparams=dict(hparam_arg=123), **kwargs):
        super().__init__()


class SubSubClassEvalModel(SubClassEvalModel):
    pass


class PersistClassEvalModel(SubClassEvalModel):
    def __init__(self, *args, skip_arg=450, **kwargs):
        self.skip_arg = 15
        super().__init__()


@pytest.mark.parametrize("cls", [EvalModelTemplate,
                                 SubClassEvalModel,
                                 SubSubClassEvalModel,
                                 HparamsNamespaceEvalModel,
                                 HparamsDictEvalModel,
                                 PersistClassEvalModel])
def test_auto_hparams(tmpdir, cls):
    # test that the model automatically sets the args passed into init as attrs
    model = cls()
    assert model.batch_size == 32
    model = cls(batch_size=179)
    assert model.batch_size == 179

    if isinstance(model, SubClassEvalModel):
        assert model.subclass_arg == 1200

    if isinstance(model, (HparamsNamespaceEvalModel, HparamsDictEvalModel)):
        assert model.hparam_arg == 123

    if isinstance(model, PersistClassEvalModel):
        assert model.skip_arg == 15

    # verify that the checkpoint saved the correct values
    trainer = Trainer(max_steps=5)
    trainer.fit(model)
    raw_checkpoint_path = os.listdir(trainer.checkpoint_callback.dirpath)
    raw_checkpoint_path = [x for x in raw_checkpoint_path if '.ckpt' in x][0]
    raw_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, raw_checkpoint_path)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert 'module_arguments' in raw_checkpoint
    assert raw_checkpoint['module_arguments']['batch_size'] == 179

    # verify that model loads correctly
    model = cls.load_from_checkpoint(raw_checkpoint_path)
    assert model.batch_size == 179

    # verify that we can overwrite whatever we want
    model = cls.load_from_checkpoint(raw_checkpoint_path, batch_size=99)
    assert model.batch_size == 99
