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
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Optional, Union
from weakref import proxy

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import (
    AbstractDataFetcher,
    DataFetcher,
    DataLoaderIterDataFetcher,
    InterBatchParallelDataFetcher,
)
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities.warnings import rank_zero_warn


class DataConnector:
    def __init__(
        self,
        trainer: "pl.Trainer",
        multiple_trainloader_mode: str = "max_size_cycle",
        train_data_fetcher: Optional[AbstractDataFetcher] = None,
        validate_data_fetcher: Optional[AbstractDataFetcher] = None,
        test_data_fetcher: Optional[AbstractDataFetcher] = None,
    ):
        self.trainer = trainer
        self.multiple_trainloader_mode = multiple_trainloader_mode

        self.train_data_fetcher = train_data_fetcher
        self.validate_data_fetcher = validate_data_fetcher
        self.test_data_fetcher = test_data_fetcher
        self.sanity_check_data_fetcher: Optional[AbstractDataFetcher] = None

        self._train_dataloader_source = _DataLoaderSource(None, "")
        self._val_dataloader_source = _DataLoaderSource(None, "")
        self._test_dataloader_source = _DataLoaderSource(None, "")
        self._predict_dataloader_source = _DataLoaderSource(None, "")

    @property
    def evaluation_data_fetcher(self) -> Optional[AbstractDataFetcher]:
        if self.trainer.sanity_checking:
            return self.sanity_check_data_fetcher
        return self.test_data_fetcher if self.trainer.testing else self.validate_data_fetcher

    def on_trainer_init(
        self,
        check_val_every_n_epoch: int,
        reload_dataloaders_every_n_epochs: int,
        reload_dataloaders_every_epoch: bool,
        prepare_data_per_node: Optional[bool] = None,
    ) -> None:
        self.trainer.datamodule = None

        if prepare_data_per_node is not None:
            rank_zero_deprecation(
                "Setting `prepare_data_per_node` with the trainer flag is deprecated and will be removed in v1.7.0! "
                "Please set `prepare_data_per_node` in LightningDataModule or LightningModule directly instead. "
            )
        self.trainer.prepare_data_per_node = prepare_data_per_node

        if not isinstance(check_val_every_n_epoch, int):
            raise MisconfigurationException(
                f"check_val_every_n_epoch should be an integer. Found {check_val_every_n_epoch}"
            )

        self.trainer.check_val_every_n_epoch = check_val_every_n_epoch

        if reload_dataloaders_every_epoch:
            reload_dataloaders_every_n_epochs = int(reload_dataloaders_every_epoch)
            rank_zero_deprecation(
                "`reload_dataloaders_every_epoch` is deprecated in v1.4 and will be removed in v1.6."
                " Please use `reload_dataloaders_every_n_epochs` in Trainer."
            )

        if not isinstance(reload_dataloaders_every_n_epochs, int) or (reload_dataloaders_every_n_epochs < 0):
            raise MisconfigurationException(
                f"`reload_dataloaders_every_n_epochs` should be an int >= 0, got {reload_dataloaders_every_n_epochs}."
            )

        self.trainer.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.trainer._is_data_prepared = False

    def _select_data_fetcher(self) -> AbstractDataFetcher:
        if self.trainer.sanity_checking:
            return DataFetcher()

        training_step_fx = getattr(self.trainer.lightning_module, "training_step")
        if self.trainer.training and is_param_in_hook_signature(training_step_fx, "dataloader_iter", explicit=True):
            rank_zero_warn(
                "Found `dataloader_iter` argument in the `training_step`. Note that the support for "
                "this signature is experimental and the behavior is subject to change."
            )
            return DataLoaderIterDataFetcher()

        elif self.trainer.training and os.getenv("PL_INTER_BATCH_PARALLELISM", "0") == "1":
            # note: this is an experimental feature
            if not self.trainer.training_type_plugin.on_gpu:
                raise MisconfigurationException("Inter batch parallelism is available only when using Nvidia GPUs.")
            return InterBatchParallelDataFetcher()

        return DataFetcher()

    def get_profiled_dataloader(self, dataloader: Iterable, dataloader_idx: int = 0) -> Iterable:
        stage: str = self.trainer.state.stage.value
        data_fetcher = setattr(self, f"{stage}_data_fetcher", None) or self._select_data_fetcher()
        data_fetcher.setup(
            dataloader,
            stage=stage,
            batch_to_device=partial(self.trainer.accelerator.batch_to_device, dataloader_idx=dataloader_idx),
            profiler=self.trainer.profiler,
        )
        setattr(self, f"{stage}_data_fetcher", data_fetcher)
        return data_fetcher

    def prepare_data(self) -> None:
        # on multi-gpu jobs we only want to manipulate (download, etc) on node_rank=0, local_rank=0
        # or in the case where each node needs to do its own manipulation in which case just local_rank=0
        local_rank_zero = self.trainer.local_rank == 0
        global_rank_zero = self.trainer.local_rank == 0 and self.trainer.node_rank == 0

        datamodule = self.trainer.datamodule
        lightning_module = self.trainer.lightning_module
        # handle datamodule prepare data:
        # check for prepare_data_per_node & datamodule lifecycle properties before calling datamodule.prepare_data
        if datamodule is not None and not datamodule._has_prepared_data:
            dm_prepare_data_per_node = datamodule.prepare_data_per_node
            dm_eq_prepare_data = datamodule.prepare_data_per_node == self.trainer.prepare_data_per_node
            if self.trainer.prepare_data_per_node is not None and not dm_eq_prepare_data:
                raise MisconfigurationException(
                    "Inconsistent settings found for `prepare_data_per_node`."
                    f" Value was set with both `Trainer(prepare_data_per_node={self.trainer.prepare_data_per_node}.)`"
                    f" and `DataModule.prepare_data_per_node={datamodule.prepare_data_per_node}`."
                    " Move `prepare_data_per_node` setting to DataModule property."
                )
            if (dm_prepare_data_per_node and local_rank_zero) or (not dm_prepare_data_per_node and global_rank_zero):
                self.trainer.datamodule.prepare_data()
        # handle lightning module prepare data:
        # check for prepare_data_per_node before calling lightning_module.prepare_data
        if lightning_module is not None:
            lm_prepare_data_per_node = lightning_module.prepare_data_per_node
            lm_eq_prepare_data = lightning_module.prepare_data_per_node == self.trainer.prepare_data_per_node
            if (self.trainer.prepare_data_per_node is not None) and not lm_eq_prepare_data:
                raise MisconfigurationException(
                    "Inconsistent settings found for `prepare_data_per_node`."
                    f" Value was set with both `Trainer(prepare_data_per_node={self.trainer.prepare_data_per_node}.)`"
                    f" and `LightningModule.prepare_data_per_node={lightning_module.prepare_data_per_node}`."
                    " Move `prepare_data_per_node` setting to LightningModule property."
                )
            if (lm_prepare_data_per_node and local_rank_zero) or (not lm_prepare_data_per_node and global_rank_zero):
                self.trainer.call_hook("prepare_data")
                self.trainer._is_data_prepared = True

    def attach_data(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[TRAIN_DATALOADERS] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        test_dataloaders: Optional[EVAL_DATALOADERS] = None,
        predict_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional["pl.LightningDataModule"] = None,
    ) -> None:
        # set up the passed in dataloaders (if needed)
        self.attach_dataloaders(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            test_dataloaders=test_dataloaders,
            predict_dataloaders=predict_dataloaders,
        )
        self.attach_datamodule(model, datamodule=datamodule)
        # set local properties on the model
        self._copy_trainer_model_properties(model)

    def _copy_trainer_model_properties(self, model):
        ref_model = self.trainer.lightning_module or model

        for m in [model, ref_model]:
            m.trainer = proxy(self.trainer)
            m.use_amp = self.trainer.amp_backend is not None
            m.precision = self.trainer.precision

    def attach_dataloaders(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[TRAIN_DATALOADERS] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        test_dataloaders: Optional[EVAL_DATALOADERS] = None,
        predict_dataloaders: Optional[EVAL_DATALOADERS] = None,
    ) -> None:
        self.trainer.train_dataloader = None
        self.trainer.val_dataloaders = None
        self.trainer.test_dataloaders = None
        self.trainer.predict_dataloaders = None

        self._train_dataloader_source = _DataLoaderSource(
            train_dataloaders if train_dataloaders is not None else model, "train_dataloader"
        )
        self._val_dataloader_source = _DataLoaderSource(
            val_dataloaders if val_dataloaders is not None else model, "val_dataloader"
        )
        self._test_dataloader_source = _DataLoaderSource(
            test_dataloaders if test_dataloaders is not None else model, "test_dataloader"
        )
        self._predict_dataloader_source = _DataLoaderSource(
            predict_dataloaders if predict_dataloaders is not None else model, "predict_dataloader"
        )

    def attach_datamodule(
        self, model: "pl.LightningModule", datamodule: Optional["pl.LightningDataModule"] = None
    ) -> None:
        # If we have a datamodule, attach necessary hooks + dataloaders
        if datamodule is None:
            return

        self._train_dataloader_source = _DataLoaderSource(datamodule, "train_dataloader")
        self._val_dataloader_source = _DataLoaderSource(datamodule, "val_dataloader")
        self._test_dataloader_source = _DataLoaderSource(datamodule, "test_dataloader")
        self._predict_dataloader_source = _DataLoaderSource(datamodule, "predict_dataloader")

        # Override data transfer hooks if dataset-specific to_device logic has been defined in datamodule
        batch_transfer_hooks = ("on_before_batch_transfer", "transfer_batch_to_device", "on_after_batch_transfer")
        for hook in batch_transfer_hooks:
            if is_overridden(hook, datamodule):
                setattr(model, hook, getattr(datamodule, hook))

        self.trainer.datamodule = datamodule
        datamodule.trainer = self.trainer

        # experimental feature for Flash
        if hasattr(datamodule, "data_pipeline"):
            model.data_pipeline = datamodule.data_pipeline

    def teardown(self) -> None:
        if self.train_data_fetcher:
            self.train_data_fetcher.teardown()
            self.train_data_fetcher = None
        if self.validate_data_fetcher:
            self.validate_data_fetcher.teardown()
            self.validate_data_fetcher = None
        if self.test_data_fetcher:
            self.test_data_fetcher.teardown()
            self.test_data_fetcher = None
        if self.sanity_check_data_fetcher:
            self.sanity_check_data_fetcher.teardown()
            self.sanity_check_data_fetcher = None


@dataclass
class _DataLoaderSource:
    """Stores the information where the dataloaders come from.

    The source can be

    1. from a ``*_datalaoder()`` method on the :class:`~pytorch_lightning.core.lightning.LightningModule`,
    2. from a ``*_datalaoder()`` method on the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`,
    3. a direct instance of a :class:`~torch.utils.data.DataLoader` or supported collections thereof.

    Arguments:
        instance: A LightningModule, LightningDataModule, or (a collection of) dataloader(s).
        name: A name for this dataloader source. If the instance is a module, the name corresponds to the hook
            that returns the desired dataloader(s).
    """

    instance: Optional[Union[TRAIN_DATALOADERS, EVAL_DATALOADERS, "pl.LightningModule", "pl.LightningDataModule"]]
    name: str

    def dataloader(self) -> Union[TRAIN_DATALOADERS, EVAL_DATALOADERS]:
        """Returns the dataloader from the source.

        If the source is a module, the method with the corresponding :attr:`name` gets called.
        """
        from pytorch_lightning import LightningDataModule, LightningModule  # prevent cyclic import

        if not self.name:
            return self.instance

        if isinstance(self.instance, LightningModule):
            return self.instance.trainer.call_hook(self.name, pl_module=self.instance)

        if isinstance(self.instance, LightningDataModule):
            method = getattr(self.instance, self.name)
            return method()

        return self.instance

    def is_defined(self) -> bool:
        """Returns whether the source dataloader can be retrieved or not.

        If the source is a module it checks that the method with given :attr:`name` is overridden.
        """
        return not self.is_module() or is_overridden(self.name, self.instance)

    def is_module(self) -> bool:
        """Returns whether the the DataLoader source is a LightningModule or a LightningDataModule.

        It does not check whether ``*_dataloader`` methods are actually overridden.
        """
        from pytorch_lightning import LightningDataModule, LightningModule  # prevent cyclic import

        return isinstance(self.instance, (LightningModule, LightningDataModule))
