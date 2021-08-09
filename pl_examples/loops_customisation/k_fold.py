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

"""
WARNING: Loop customization is in `pre-alpha release` and the API is likely to change quite a lot !
Please, open issues with your own particular requests, so the Lightning Team can progressively converge to a great API.
"""

from typing import Any, Dict, List, Optional, Type

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

from pytorch_lightning import _logger as log
from pytorch_lightning import LightningDataModule, seed_everything
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loops.external_loop import ExternalLoop
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.boring_model import BoringModel, RandomDataset

seed_everything(42)


class BaseDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.non_picklable = None
        self.checkpoint_state: Optional[str] = None

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._predict_dataset: Optional[Dataset] = None

        self._processed_train_dataset: Optional[Dataset] = None
        self._processed_val_dataset: Optional[Dataset] = None
        self._processed_test_dataset: Optional[Dataset] = None
        self._processed_predict_dataset: Optional[Dataset] = None

    @property
    def train_dataset(self) -> Optional[Dataset]:
        return self._train_dataset

    @property
    def val_dataset(self) -> Optional[Dataset]:
        return self._val_dataset

    @property
    def test_dataset(self) -> Optional[Dataset]:
        return self._test_dataset

    @property
    def predict_dataset(self) -> Optional[Dataset]:
        return self._predict_dataset

    @property
    def processed_train_dataset(self) -> Optional[Dataset]:
        return self._processed_train_dataset or self.train_dataset

    @property
    def processed_val_dataset(self) -> Optional[Dataset]:
        return self._processed_val_dataset or self.val_dataset

    @property
    def processed_test_dataset(self) -> Optional[Dataset]:
        return self._processed_test_dataset or self.test_dataset

    @property
    def processed_predict_dataset(self) -> Optional[Dataset]:
        return self._processed_predict_dataset or self.predict_dataset

    @processed_train_dataset.setter
    def processed_train_dataset(self, processed_train_dataset) -> None:
        self._processed_train_dataset = processed_train_dataset

    @processed_val_dataset.setter
    def processed_val_dataset(self, processed_val_dataset) -> None:
        self._processed_val_dataset = processed_val_dataset

    @processed_val_dataset.setter
    def processed_val_dataset(self, processed_val_dataset) -> None:
        self._processed_val_dataset = processed_val_dataset

    @processed_test_dataset.setter
    def processed_test_dataset(self, processed_test_dataset) -> None:
        self._processed_test_dataset = processed_test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.processed_train_dataset)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.processed_val_dataset)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.processed_test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.processed_predict_dataset)


class BoringDataModule(BaseDataModule):
    def prepare_data(self) -> None:
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self._train_dataset = Subset(self.random_full, indices=range(64))
            self.dims = self._train_dataset[0].shape

        if stage in ("fit", "validate") or stage is None:
            self._val_dataset = Subset(self.random_full, indices=range(64, 64 * 2))

        if stage == "test" or stage is None:
            self._test_dataset = Subset(self.random_full, indices=range(64 * 2, 64 * 3))
            self.dims = getattr(self, "dims", self._test_dataset[0].shape)

        if stage == "predict" or stage is None:
            self._predict_dataset = Subset(self.random_full, indices=range(64 * 3, 64 * 4))
            self.dims = getattr(self, "dims", self._predict_dataset[0].shape)


class KFoldLoop(ExternalLoop):
    def __init__(
        self,
        num_folds: int,
        best_model_paths: List[str] = [],
        restarting: bool = False,
    ):
        super().__init__()
        self.num_folds = num_folds
        self.best_model_paths = best_model_paths
        self.restarting = restarting

    @staticmethod
    def loop_base_callback() -> Type[Callback]:
        class BaseKFoldCallback(Callback):
            @rank_zero_only
            def on_fold_start(self, trainer, pl_module, counter):
                """Override with your own logic"""

        return BaseKFoldCallback

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def reset(self) -> None:
        if not self.restarting:
            self.current_fold = 0

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        # temporary hack
        self.trainer.datamodule.setup("fit")

    def on_advance_start(self) -> None:
        # more reproducible as re-creating a different trainer.
        self.create_trainer(max_epochs=np.random.randint(10))
        # reload dataset for the current fold
        dm = self.trainer.datamodule
        dm.processed_train_dataset = self.process_dataset("train", dm.train_dataset)
        dm.processed_val_dataset = self.process_dataset("val", dm.val_dataset)
        # call user hook
        self.trainer.call_hook("on_fold_start", self.current_fold)
        # reset model parameters
        self.trainer.lightning_module.reset_parameters()

    def advance(self) -> Any:
        # dataloaders will be automatically reloaded
        return self.trainer.fit(self.trainer.lightning_module, datamodule=self.trainer.datamodule)

    def on_advance_end(self) -> None:
        self.current_fold += 1
        # stored best weight path for this fold
        self.best_model_paths.append(self.trainer.checkpoint_callback.best_model_path)

    # utilities for creating a hold
    def process_dataset(self, stage: str, dataset: Dataset) -> Subset:
        kfold = KFold(self.num_folds, random_state=42, shuffle=True)
        train_indices, validation_indices = list(kfold.split(range(len(dataset))))[self.current_fold]
        indices = train_indices if stage == "train" else validation_indices
        return Subset(dataset, indices.tolist())

    def on_save_checkpoint(self) -> Dict:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict) -> None:
        self.current_fold = state_dict["current_fold"]


class KFoldCallback(KFoldLoop.loop_base_callback()):

    """This callback demonstrates how to implement your own callback API."""

    @rank_zero_only
    def on_fold_start(self, trainer, pl_module, counter) -> None:
        log.info(f"Starting to train on fold {counter}")


loop = KFoldLoop(5)
model = BoringModel()
datamodule = BoringDataModule()
loop.connect_trainer(max_epochs=10, callbacks=KFoldCallback())
loop.run(model, datamodule=datamodule)
