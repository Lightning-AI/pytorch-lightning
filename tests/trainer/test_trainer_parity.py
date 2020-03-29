import glob
import math
import os
from argparse import Namespace

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.core.lightning import load_hparams_from_tags_csv
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.utilities.debugging import MisconfigurationException
from tests.base import (
    TestModelBase,
    DictHparamsModel,
    LightningTestModel,
    LightEmptyTestStep,
    LightValidationStepMixin,
    LightValidationMultipleDataloadersMixin,
    LightTrainDataloader,
    LightTestDataloader,
)


def test_pytorch_parity(tmpdir):
    """
    Verify that the same pytorch and lightning models achieve the same results
    :param tmpdir:
    :return:
    """

    model = DictHparamsModel({'in_features': 28 * 28, 'out_features': 10})

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=2,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1

    # try to load the model now
    pretrained_model = tutils.load_model_from_checkpoint(
        trainer.checkpoint_callback.dirpath,
        module_class=DictHparamsModel
    )
