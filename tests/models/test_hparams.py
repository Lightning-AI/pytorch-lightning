import os

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import CHECKPOINT_KEY_MODULE_ARGS
from tests.base import EvalModelTemplate


class SubClassEvalModel(EvalModelTemplate):
    object_that_should_not_be_saved = torch.nn.CrossEntropyLoss()

    def __init__(self, *args, subclass_arg=1200, **kwargs):
        super().__init__(*args, **kwargs)
        self.subclass_arg = subclass_arg


class SubSubClassEvalModel(SubClassEvalModel):
    pass


@pytest.mark.parametrize("cls", [EvalModelTemplate,
                                 SubClassEvalModel,
                                 SubSubClassEvalModel])
def test_collect_init_arguments(tmpdir, cls):
    """ Test that the model automatically saves the arguments passed into the constructor """
    model = cls()
    assert model.batch_size == 32
    model = cls(batch_size=179)
    assert model.batch_size == 179

    if isinstance(model, SubClassEvalModel):
        assert model.subclass_arg == 1200

    # verify that the checkpoint saved the correct values
    trainer = Trainer(max_steps=5, default_root_dir=tmpdir)
    trainer.fit(model)
    raw_checkpoint_path = os.listdir(trainer.checkpoint_callback.dirpath)
    raw_checkpoint_path = [x for x in raw_checkpoint_path if '.ckpt' in x][0]
    raw_checkpoint_path = os.path.join(trainer.checkpoint_callback.dirpath, raw_checkpoint_path)
    raw_checkpoint = torch.load(raw_checkpoint_path)
    assert CHECKPOINT_KEY_MODULE_ARGS in raw_checkpoint
    assert raw_checkpoint[CHECKPOINT_KEY_MODULE_ARGS]['batch_size'] == 179

    # verify that model loads correctly
    model = cls.load_from_checkpoint(raw_checkpoint_path)
    assert model.batch_size == 179

    # verify that we can overwrite whatever we want
    model = cls.load_from_checkpoint(raw_checkpoint_path, batch_size=99)
    assert model.batch_size == 99
