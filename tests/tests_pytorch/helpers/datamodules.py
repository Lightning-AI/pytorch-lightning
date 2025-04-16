# Copyright The Lightning AI team.
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

import torch
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader

from lightning.pytorch.core.datamodule import LightningDataModule
from tests_pytorch.helpers.datasets import MNIST, SklearnDataset, TrialMNIST

_SKLEARN_AVAILABLE = RequirementCache("scikit-learn")


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32, use_trials: bool = False) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        # TrialMNIST is a constrained MNIST dataset
        self.dataset_cls = TrialMNIST if use_trials else MNIST

    def prepare_data(self):
        # download only
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            self.mnist_train = self.dataset_cls(self.data_dir, train=True)
        if stage == "test":
            self.mnist_test = self.dataset_cls(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)


class SklearnDataModule(LightningDataModule):
    def __init__(self, sklearn_dataset, x_type, y_type, batch_size: int = 10):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(str(_SKLEARN_AVAILABLE))

        super().__init__()
        self.batch_size = batch_size
        self._x, self._y = sklearn_dataset
        self._split_data()
        self._x_type = x_type
        self._y_type = y_type

    def _split_data(self):
        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self._x, self._y, test_size=0.20, random_state=42
        )
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.x_train, self.y_train, test_size=0.40, random_state=42
        )

    def train_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_train, self.y_train, self._x_type, self._y_type),
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_valid, self.y_valid, self._x_type, self._y_type), batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_test, self.y_test, self._x_type, self._y_type), batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_test, self.y_test, self._x_type, self._y_type), batch_size=self.batch_size
        )

    @property
    def sample(self):
        return torch.tensor([self._x[0]], dtype=self._x_type)


class ClassifDataModule(SklearnDataModule):
    def __init__(
        self, num_features=32, length=800, num_classes=3, batch_size=10, n_clusters_per_class=1, n_informative=2
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(str(_SKLEARN_AVAILABLE))

        from sklearn.datasets import make_classification

        data = make_classification(
            n_samples=length,
            n_features=num_features,
            n_classes=num_classes,
            n_clusters_per_class=n_clusters_per_class,
            n_informative=n_informative,
            random_state=42,
        )
        super().__init__(data, x_type=torch.float32, y_type=torch.long, batch_size=batch_size)


class RegressDataModule(SklearnDataModule):
    def __init__(self, num_features=16, length=800, batch_size=10):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(str(_SKLEARN_AVAILABLE))

        from sklearn.datasets import make_regression

        x, y = make_regression(n_samples=length, n_features=num_features, random_state=42)
        y = [[v] for v in y]
        super().__init__((x, y), x_type=torch.float32, y_type=torch.float32, batch_size=batch_size)
