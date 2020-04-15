import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.base.datasets import TestingMNIST
from pytorch_lightning.core.lightning import LightningModule
from tests.base.training_step_variations import TrainingStepVariationsMixin
from tests.base.test_step_variations import TestStepVariationsMixin
from tests.base.validation_step_variations import ValidationStepVariationsMixin
from tests.base.test_epoch_end_variations import TestEpochEndVariationsMixin
from tests.base.config_optimizers_variations import ConfigureOptimizersVariationsMixin
from tests.base.val_dataloader_variations import ValDataloaderVariationsMixin
from tests.base.train_dataloader_variations import TrainDataloaderVariationsMixin
from tests.base.test_dataloader_variations import TestDataloaderVariationsMixin
from tests.base.validation_epoch_end_variations import ValidationEpochEndVariationsMixin


class TemplateTestModel(
    TrainingStepVariationsMixin,
    ValidationStepVariationsMixin,
    ValidationEpochEndVariationsMixin,
    TestStepVariationsMixin,
    TestEpochEndVariationsMixin,
    TrainDataloaderVariationsMixin,
    ValDataloaderVariationsMixin,
    TestDataloaderVariationsMixin,
    ConfigureOptimizersVariationsMixin,
    LightningModule
):
    """
    This template houses all  combinations of model  configurations  we want to test
    """
    def __init__(self, hparams):
        """Pass in parsed HyperOptArgumentParser to the model."""
        # init superclass
        super().__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

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
        _ = TestingMNIST(root=self.hparams.data_root, train=True, download=True)
