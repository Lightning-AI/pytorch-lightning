from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from pytorch_lightning import _logger as log
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loops.base import ExternalLoop
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.boring_model import BoringDataModule, BoringModel
from pytorch_lightning.utilities.exceptions import MisconfigurationException

seed_everything(42)


class SplitDataset(Dataset):
    """SplitDataset is used to create Dataset Subset using indices.
    Args:
        dataset: A dataset to be splitted
        indices: List of indices to expose from the dataset
        use_duplicated_indices: Whether to allow duplicated indices.
    Example::
        split_ds = SplitDataset(dataset, indices=[10, 14, 25])
        split_ds = SplitDataset(dataset, indices=[10, 10, 10, 14, 25], use_duplicated_indices=True)
    """

    _INTERNAL_KEYS = ("dataset", "indices", "data")

    def __init__(self, dataset: Any, indices: List[int] = None, use_duplicated_indices: bool = False) -> None:
        if indices is None:
            indices = []
        if not isinstance(indices, list):
            raise MisconfigurationException("indices should be a list")

        if use_duplicated_indices:
            indices = list(indices)
        else:
            indices = list(np.unique(indices))

        if np.max(indices) >= len(dataset) or np.min(indices) < 0:
            raise MisconfigurationException(f"`indices` should be within [0, {len(dataset) -1}].")

        self.dataset = dataset
        self.indices = indices

    def __getattr__(self, key: str):
        if key not in self._INTERNAL_KEYS:
            return self.dataset.__getattribute__(key)
        raise AttributeError

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._INTERNAL_KEYS:
            self.__dict__[name] = value
        else:
            setattr(self.dataset, name, value)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices) - 1


@dataclass
class KFoldLoop(ExternalLoop):

    num_folds: int
    num_epochs: int = 10
    best_model_paths: List[str] = field(default_factory=lambda: [])
    restarting: bool = False

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
            self.set_max_epochs(self.num_epochs)

    def generate_fold(self, dataloader_kwargs: Dict[str, Any], stage: str):
        dataset = dataloader_kwargs["dataset"]
        kfold = KFold(self.num_folds, random_state=42, shuffle=True)
        train_indices, validation_indices = list(kfold.split(range(len(dataset))))[self.current_fold]
        if stage == "train":
            dataloader_kwargs["dataset"] = SplitDataset(dataset, train_indices.tolist())
        else:
            dataloader_kwargs["dataset"] = SplitDataset(dataset, validation_indices.tolist())
        dataloader_kwargs["sampler"].data_source = dataloader_kwargs["dataset"]
        return dataloader_kwargs

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        # temporary hack
        self.trainer.datamodule.setup("fit")

    def on_advance_start(self):
        self.reload_train_dataloader(self.generate_fold)
        self.reload_val_dataloaders(self.generate_fold)
        self.trainer.call_hook("on_fold_start", self.current_fold)
        self.lightning_module.reset_parameters()

    def advance(self):
        return self.trainer.fit(self.lightning_module, train_dataloader=self.train_dataloader)

    def on_advance_end(self) -> None:
        self.current_fold += 1
        self.increment_max_epochs(self.num_epochs)
        # stored best weight path for this fold
        self.best_model_paths.append(self.trainer.checkpoint_callback.best_model_path)

    def on_save_checkpoint(self) -> Dict:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict) -> None:
        self.current_fold = state_dict["current_fold"]


class KFoldCallback(KFoldLoop.loop_base_callback()):

    """This callback demonstrates how to implement your create callbacks."""

    @rank_zero_only
    def on_fold_start(self, trainer, pl_module, counter):
        log.info(f"Starting to train on fold {counter}")


loop = KFoldLoop(5)
model = BoringModel()
datamodule = BoringDataModule()
trainer = Trainer(callbacks=KFoldCallback())
trainer.run_loop(model, datamodule=datamodule, loop=loop)
