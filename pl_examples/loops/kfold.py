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
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
import torchvision.transforms as T
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

from pl_examples import _DATASETS_PATH
from pl_examples.basic_examples.mnist_datamodule import MNIST
from pl_examples.basic_examples.simple_image_classifier import LitClassifier
from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn

#############################################################################################
#                           KFold Loop / Cross Validation Example                           #
# This example demonstrates how to leverage Lightning Loop Customization introduced in v1.5 #
# Learn more about the loop structure from the documentation                                #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html                  #
#############################################################################################


seed_everything(42)


#############################################################################################
#                           Step 1 / 5: Define KFold DataModule API                         #
# Our KFold DataModule should require to implement `setup_folds` and `setup_fold_index`    #
# function.                                                                                 #
#############################################################################################


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be splitted accordingly to        #
# the current fold split.                                                                   #
#############################################################################################


@dataclass
class MNISTKFoldDataModule(BaseKFoldDataModule):

    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # download the data.
        MNIST(_DATASETS_PATH, transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))]))

    def setup(self, stage: Optional[str] = None) -> None:
        # load the data
        dataset = MNIST(_DATASETS_PATH, transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))]))
        self.train_dataset, self.test_dataset = random_split(dataset, [50000, 10000])

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)


#############################################################################################
#                           Step 3 / 5: Implement the EnsembleVotingModel module            #
# The `EnsembleVotingModel` will take our custom LightningModule and                        #
# several checkpoint_paths.                                                                 #
# On `__init__`, it would create multiple models by reloading the fold weights              #
# On `test_step`, the model will perform a forward through all the models and take          #
# the average logits produced by the `num_folds` models, and loss the enssembling loss      #
#                                                                                           #
#############################################################################################


class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]):
        super().__init__()
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.cross_entropy(logits, batch[1])
        self.log("test_loss", loss)


#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several to do    #
# so and refer to the documentation to learn more.                                          #
# Here, we will implement an outter fit_loop. It means we will implement subclass the       #
# base Loop and wrap the current trainer `fit_loop`.                                        #
# On `on_run_start`, the `KFoldLoop` will call the `KFoldDataModule` `setup_folds` function #
# and store the original weights of the model.                                              #
# On `on_advance_start`, the `KFoldLoop` will call the `KFoldDataModule` `setup_fold_index` #
# function.                                                                                 #
# On `advance`, the `KFoldLoop` will run the original trainer `fit_loop` and                #
# the trainer `test_loop`.                                                                  #
# On `advance_end`, the `KFoldLoop` will reset the model weight and optimizers / schedulers #
#############################################################################################


class KFoldLoop(Loop):
    def __init__(self, num_folds: int, fit_loop: FitLoop, export_path: str):
        super().__init__()
        self.num_folds = num_folds
        self.fit_loop = fit_loop
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self._reset_fitting()  # requires to reset the tracking stage
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage
        self.trainer.test_loop.run()
        self.current_fold += 1

    def on_advance_end(self) -> None:
        self.trainer.save_checkpoint(os.path.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.accelerator.setup_optimizers(self.trainer)

    def on_run_end(self) -> None:
        checkpoint_paths = [
            os.path.join(self.export_path, f"model.{fold_index + 1}.pt") for fold_index in range(self.num_folds)
        ]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        self.trainer.accelerator.connect(voting_model)
        self.trainer.training_type_plugin.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop as being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]


#############################################################################################
#                           Step 5 / 5: Connect the KFoldLoop to the Trainer                #
# After creating the `KFoldDataModule` and our model, the `KFoldLoop` is being connected to #
# the Trainer.                                                                              #
# Finally, use `trainer.fit` to start the cross validation training.                        #
#############################################################################################

model = LitClassifier()
datamodule = MNISTKFoldDataModule()
trainer = Trainer(
    max_epochs=10,
    limit_train_batches=2,
    limit_val_batches=2,
    limit_test_batches=2,
    num_sanity_val_steps=0,
    devices=1,
    accelerator="auto",
    strategy="ddp",
)
trainer.fit_loop = KFoldLoop(5, trainer.fit_loop, export_path="./")
trainer.fit(model, datamodule)
