from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule
from tests.base.datasets import TrialMNIST, PATH_DATASETS
from tests.base.model_optimizers import ConfigureOptimizersPool
from tests.base.model_test_dataloaders import TestDataloaderVariations
from tests.base.model_test_epoch_ends import TestEpochEndVariations
from tests.base.model_test_steps import TestStepVariations
from tests.base.model_train_dataloaders import TrainDataloaderVariations
from tests.base.model_train_steps import TrainingStepVariations
from tests.base.model_utilities import ModelTemplateUtils, ModelTemplateData
from tests.base.model_valid_dataloaders import ValDataloaderVariations
from tests.base.model_valid_epoch_ends import ValidationEpochEndVariations
from tests.base.model_valid_steps import ValidationStepVariations


class EvalModelTemplate(
    ModelTemplateData,
    ModelTemplateUtils,
    TrainingStepVariations,
    ValidationStepVariations,
    ValidationEpochEndVariations,
    TestStepVariations,
    TestEpochEndVariations,
    TrainDataloaderVariations,
    ValDataloaderVariations,
    TestDataloaderVariations,
    ConfigureOptimizersPool,
    LightningModule
):
    """
    This template houses all  combinations of model  configurations  we want to test

    >>> model = EvalModelTemplate()
    """
    def __init__(self, hparams: object = None) -> object:
        """Pass in parsed HyperOptArgumentParser to the model."""
        if hparams is None:
            hparams = EvalModelTemplate.get_default_hparams()
        # init superclass
        super().__init__()
        self.hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand(5, 28 * 28)

        # build model
        self.__build_model()

    def __build_model(self):
        """
        Simple model for testing
        :return:
        """
        self.c_d1 = nn.Linear(
            in_features=self.hparams.in_features,
            out_features=self.hparams.hidden_dim
        )
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)

        self.c_d2 = nn.Linear(
            in_features=self.hparams.hidden_dim,
            out_features=self.hparams.out_features
        )

    def forward(self, x):
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)

        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)

        return logits

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll

    def prepare_data(self):
        _ = TrialMNIST(root=self.hparams.data_root, train=True, download=True)

    @staticmethod
    def get_default_hparams(continue_training: bool = False, hpc_exp_number: int = 0) -> Namespace:
        args = dict(
            drop_prob=0.2,
            batch_size=32,
            in_features=28 * 28,
            learning_rate=0.001 * 8,
            optimizer_name='adam',
            data_root=PATH_DATASETS,
            out_features=10,
            hidden_dim=1000,
            b1=0.5,
            b2=0.999,
        )

        if continue_training:
            args.update(
                test_tube_do_checkpoint_load=True,
                hpc_exp_number=hpc_exp_number,
            )

        hparams = Namespace(**args)
        return hparams
