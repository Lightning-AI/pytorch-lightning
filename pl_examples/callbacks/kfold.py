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
import inspect
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

from sklearn.model_selection import KFold
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset

import pytorch_lightning as pl
from pl_examples.boring_model import BoringModel
from pytorch_lightning import Callback
from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY, LightningCLI
from pytorch_lightning.utilities.exceptions import MisconfigurationException

# python pl_examples/callbacks/kfold.py --trainer.max_epochs=3 --trainer.callbacks=KFoldCallback --trainer.callbacks.num_splits=6   # noqa E501


class BaseKFold(Callback):
    @abstractmethod
    def compute_fold(self, dataset: Dataset):
        pass

    def __init__(self, num_splits: int, shuffle: bool = True, random_state: int = 42):
        self.num_splits = num_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.max_epochs: Optional[int] = None
        self.current_fold: int = 0
        self.best_model_paths: List[str] = []

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.max_epochs = trainer.fit_loop.max_epochs
        trainer.fit_loop.max_epochs *= self.num_splits
        self.train_dataloader = pl_module.train_dataloader

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._shared_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._shared_epoch_start(trainer, pl_module)

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> dict:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ) -> None:
        self.current_fold = callback_state["current_fold"]

    def _shared_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # reached new fold
        if trainer.current_epoch % self.max_epochs == 0 and trainer.training:
            trainer.train_dataloader = None
            pl_module.layer.reset_parameters()
            pl_module.train_dataloader = self.train_dataloader
            pl_module.train_dataloader = self._patch_dataloader(pl_module.train_dataloader(), self.compute_fold)
            trainer.reset_train_dataloader(pl_module)
            self.best_model_paths.append(trainer.checkpoint_callback.best_model_path)
            trainer.checkpoint_callback.best_model_path = None
            trainer.checkpoint_callback.best_model_score = None
            trainer.checkpoint_callback.best_model_score = None
            # reset all callbacks ...

    def _patch_dataloader(self, dataloader: DataLoader, fn: Callable) -> _PatchDataLoader:
        if not isinstance(dataloader, DataLoader):
            raise ValueError(f"The dataloader {dataloader} needs to subclass `torch.utils.data.DataLoader`")

        # get the dataloader instance attributes
        attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}
        # not part of `vars`
        attrs["multiprocessing_context"] = dataloader.multiprocessing_context

        # get the dataloader instance `__init__` parameters
        params = dict(inspect.signature(dataloader.__init__).parameters)

        # keep only the params whose default is different to the current attr value
        non_defaults = {name for name, p in params.items() if name in attrs and p.default != attrs[name]}
        # add `dataset` as it might have been replaced with `*args`
        non_defaults.add("dataset")

        # kwargs to re-construct the dataloader
        dl_kwargs = {k: v for k, v in attrs.items() if k in non_defaults}
        dl_kwargs.update(self._resolve_batch_sampler(dataloader, dataloader.sampler))

        required_args = {
            p.name
            for p in params.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
            and p.name not in dl_kwargs
        }
        # the dataloader has required args which we could not extract from the existing attributes
        if required_args:
            required_args = sorted(required_args)
            dataloader_cls_name = dataloader.__class__.__name__
            raise MisconfigurationException(
                f"Trying to inject `DistributedSampler` into the `{dataloader_cls_name}` instance. "
                "This would fail as some of the `__init__` arguments are not available as instance attributes. "
                f"The missing attributes are {required_args}. "
                f"HINT: If you wrote the `{dataloader_cls_name}` class, define `self.missing_arg_name` or "
                "manually add the `DistributedSampler` as: "
                f"`{dataloader_cls_name}(dataset, sampler=DistributedSampler(dataset))`."
            )

        has_variadic_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
        if not has_variadic_kwargs:
            # the dataloader signature does not allow keyword arguments that need to be passed
            missing_kwargs = dl_kwargs.keys() - params.keys()
            if missing_kwargs:
                missing_kwargs = sorted(missing_kwargs)
                dataloader_cls_name = dataloader.__class__.__name__
                raise MisconfigurationException(
                    f"Trying to inject `DistributedSampler` into the `{dataloader_cls_name}` instance. "
                    "This would fail as it doesn't expose all its attributes in the `__init__` signature. "
                    f"The missing arguments are {missing_kwargs}. "
                    f"HINT: If you wrote the `{dataloader_cls_name}` class, add the `__init__` arguments or "
                    "manually add the `DistributedSampler` as: "
                    f"`{dataloader_cls_name}(dataset, sampler=DistributedSampler(dataset))`."
                )

        dl_cls = type(dataloader)
        # apply dataset transformation
        dl_kwargs["dataset"] = fn(dl_kwargs["dataset"])
        # re-attach the dataset to the sampler
        dl_kwargs["sampler"].data_source = dl_kwargs["dataset"]
        dataloader = dl_cls(**dl_kwargs)
        return _PatchDataLoader(dataloader)

    @staticmethod
    def _resolve_batch_sampler(dataloader, sampler) -> Dict[str, Any]:
        batch_sampler = getattr(dataloader, "batch_sampler")
        if batch_sampler is not None and type(batch_sampler) is not BatchSampler:
            return {
                "sampler": None,
                "shuffle": False,
                "batch_sampler": batch_sampler,
                "batch_size": 1,
                "drop_last": False,
            }
        return {"sampler": sampler, "shuffle": False, "batch_sampler": None}


@CALLBACK_REGISTRY
class KFoldCallback(BaseKFold):
    def compute_fold(self, dataset: Dataset) -> Dataset:
        fold = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.random_state)
        indices = list(range(len(dataset)))
        _, indices = list(fold.split(indices))[self.current_fold]
        return Subset(dataset, indices)


LightningCLI(BoringModel)
