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
import multiprocessing
import os
from abc import ABC
from typing import Any, Callable, Collection, List, Optional, Tuple, Union

from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.overrides.distributed import UnrepeatedDistributedSampler
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.auto_restart import _add_capture_metadata_collate, _validate_fault_tolerant_automatic
from pytorch_lightning.utilities.data import (
    _auto_add_worker_init_fn,
    _replace_dataloader_init_method,
    _update_dataloader,
    has_iterable_dataset,
    has_len_all_ranks,
)
from pytorch_lightning.utilities.enums import _StrategyType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.warnings import PossibleUserWarning


class TrainerDataLoadingMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    val_check_interval: float
    tpu_local_core_rank: int
    train_dataloader: DataLoader
    limit_train_batches: Union[int, float]
    num_training_batches: int
    val_check_batch: float
    val_dataloaders: List[DataLoader]
    limit_val_batches: Union[int, float]
    num_val_batches: List[int]
    test_dataloaders: List[DataLoader]
    limit_test_batches: Union[int, float]
    num_test_batches: List[int]
    predict_dataloaders: List[DataLoader]
    limit_predict_batches: Union[int, float]
    num_predict_batches: List[int]
    log_every_n_steps: int
    overfit_batches: Union[int, float]
    distributed_sampler_kwargs: dict
    accelerator: Accelerator
    call_hook: Callable
    _accelerator_connector: AcceleratorConnector

    def _worker_check(self, dataloader: DataLoader, name: str) -> None:
        if not isinstance(dataloader, DataLoader):
            return

        using_spawn = self._accelerator_connector._distrib_type == _StrategyType.DDP_SPAWN
        num_cpus = multiprocessing.cpu_count()

        # ddp_spawn + num_workers > 0 don't mix! tell the user
        if dataloader.num_workers > 0 and using_spawn:
            # checks for the attr persistent_workers available in pytorch >= 1.7
            if hasattr(dataloader, "persistent_workers"):
                if not dataloader.persistent_workers:
                    rank_zero_warn(
                        "num_workers>0, persistent_workers=False, and strategy=ddp_spawn"
                        " may result in data loading bottlenecks."
                        " Consider setting persistent_workers=True"
                        " (this is a limitation of Python .spawn() and PyTorch)"
                    )
            else:
                rank_zero_warn(
                    "num_workers>0 and strategy=ddp_spawn do not mix well"
                    " and may result in data loading bottlenecks."
                    " Consider setting strategy=ddp to use num_workers>0"
                    " (this is a limitation of Python .spawn() and PyTorch)"
                )

        elif dataloader.num_workers == 0 and using_spawn:
            # checks for the attr persistent_workers available in pytorch >= 1.7
            if hasattr(dataloader, "persistent_workers"):
                if not dataloader.persistent_workers:
                    rank_zero_warn(
                        "strategy=ddp_spawn and num_workers=0 may result in data loading bottlenecks."
                        " Consider setting num_workers>0 and persistent_workers=True"
                    )
            else:
                rank_zero_warn(
                    "strategy=ddp_spawn and num_workers=0 may result in data loading bottlenecks."
                    " Consider setting strategy=ddp and set num_workers>0"
                )

        elif dataloader.num_workers <= 2 < num_cpus and not using_spawn:
            # if changed, update the `filterwarnings` snippet in 'speed.html#num-workers'
            rank_zero_warn(
                f"The dataloader, {name}, does not have many workers which may be a bottleneck."
                " Consider increasing the value of the `num_workers` argument`"
                f" (try {num_cpus} which is the number of cpus on this machine)"
                " in the `DataLoader` init to improve performance.",
                category=PossibleUserWarning,
            )

    def _requires_distributed_sampler(self, dataloader) -> bool:
        return (
            self._accelerator_connector.replace_sampler_ddp
            and self._accelerator_connector.is_distributed
            and not isinstance(dataloader.sampler, DistributedSampler)
            and not has_iterable_dataset(dataloader)
        )

    def prepare_dataloader(self, dataloader: Any, shuffle: bool, mode: Optional[RunningStage] = None) -> Any:
        """This function handles to following functionalities:

        - Injecting a `DistributedDataSampler` into the `DataLoader` if on a distributed environment
        - Wrapping the datasets and samplers into fault-tolerant components
        """
        if isinstance(dataloader, CombinedLoader):
            # apply `prepare_dataloader` on all the collection of loaders
            dataloader.loaders = apply_to_collection(
                dataloader.loaders, (DataLoader, CycleIterator), self.prepare_dataloader, shuffle, mode=mode
            )
            # the length need to recomputed across all dataloaders in case of special behavior.
            dataloader._apply_cycle_iterator_length()
            return dataloader

        # don't do anything if it's not a dataloader
        if not isinstance(dataloader, (DataLoader, CycleIterator)):
            return dataloader

        cycle_iterator: Optional[CycleIterator] = None

        if isinstance(dataloader, CycleIterator):
            cycle_iterator = dataloader
            dataloader = dataloader.loader

        if (
            _fault_tolerant_training()  # injects components to track the state
            or self._requires_distributed_sampler(dataloader)  # sets the distributed sampler
            or mode == RunningStage.PREDICTING  # to track indices for the predictions
            or self._accelerator_connector.use_ipu  # IPUs use a custom `DataLoader`
        ):
            sampler = self._resolve_sampler(dataloader, shuffle=shuffle, mode=mode)
            dataloader = _update_dataloader(dataloader, sampler, mode=mode)

        if cycle_iterator is not None:
            cycle_iterator.loader = dataloader
            return cycle_iterator

        return dataloader

    def _resolve_sampler(self, dataloader: DataLoader, shuffle: bool, mode: Optional[RunningStage] = None) -> Sampler:
        if self._requires_distributed_sampler(dataloader):
            if not isinstance(dataloader.sampler, (SequentialSampler, RandomSampler)):
                raise MisconfigurationException(
                    "You seem to have configured a sampler in your DataLoader. This will be replaced"
                    " by `DistributedSampler` since `replace_sampler_ddp` is True and you are using"
                    " distributed training. Either remove the sampler from your DataLoader or set"
                    " `replace_sampler_ddp=False` if you want to use your custom sampler."
                )
            return self._get_distributed_sampler(
                dataloader, shuffle, mode=mode, overfit_batches=self.overfit_batches, **self.distributed_sampler_kwargs
            )

        return dataloader.sampler

    @staticmethod
    def _get_distributed_sampler(
        dataloader: DataLoader,
        shuffle: bool,
        overfit_batches: Union[int, float],
        mode: Optional[RunningStage] = None,
        **kwargs: Any,
    ) -> DistributedSampler:
        """This function is used to created the distributed sampler injected within the user DataLoader."""
        kwargs["shuffle"] = shuffle and not overfit_batches
        kwargs.setdefault("seed", int(os.getenv("PL_GLOBAL_SEED", 0)))
        cls = UnrepeatedDistributedSampler if mode == RunningStage.PREDICTING else DistributedSampler
        sampler = cls(dataloader.dataset, **kwargs)
        return sampler

    def reset_train_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the train dataloader and initialises required variables (number of batches, when to validate,
        etc.).

        Args:
            model: The `LightningModule` if calling this outside of the trainer scope.
        """
        self.train_dataloader = self.request_dataloader(RunningStage.TRAINING, model=model)

        if self.overfit_batches > 0:
            self.train_dataloader = self._resolve_overfit_batches(self.train_dataloader)

        # automatically add samplers
        self.train_dataloader = apply_to_collection(
            self.train_dataloader, DataLoader, self.prepare_dataloader, shuffle=True, mode=RunningStage.TRAINING
        )

        # check the workers recursively
        apply_to_collection(self.train_dataloader, DataLoader, self._worker_check, "train_dataloader")

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(self.train_dataloader, DataLoader, _auto_add_worker_init_fn, rank=self.global_rank)

        # add collate_fn to collect metadata for fault tolerant training
        if _fault_tolerant_training():
            apply_to_collection(self.train_dataloader, DataLoader, _add_capture_metadata_collate)

        # wrap the sequence of train loaders to a CombinedLoader object for computing the num_training_batches
        self.train_dataloader = CombinedLoader(self.train_dataloader, self._data_connector.multiple_trainloader_mode)

        module = model or self.lightning_module or self.datamodule
        self.num_training_batches = (
            len(self.train_dataloader)
            if has_len_all_ranks(self.train_dataloader, self.training_type_plugin, module)
            else float("inf")
        )

        if isinstance(self.limit_train_batches, int) or self.limit_train_batches == 0.0:
            self.num_training_batches = min(self.num_training_batches, int(self.limit_train_batches))
        elif self.num_training_batches != float("inf"):
            self.num_training_batches = int(self.num_training_batches * self.limit_train_batches)
        elif self.limit_train_batches != 1.0:
            raise MisconfigurationException(
                "When using an IterableDataset for `limit_train_batches`,"
                " `Trainer(limit_train_batches)` must be `0.0`, `1.0` or an int. An int k specifies"
                " `num_training_batches` to use."
            )

        # determine when to check validation
        # if int passed in, val checks that often
        # otherwise, it checks in [0, 1.0] % range of a training epoch
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches:
                raise ValueError(
                    f"`val_check_interval` ({self.val_check_interval}) must be less than or equal "
                    f"to the number of the training batches ({self.num_training_batches}). "
                    "If you want to disable validation set `limit_val_batches` to 0.0 instead."
                )
        else:
            if not has_len_all_ranks(self.train_dataloader, self.training_type_plugin, module):
                if self.val_check_interval == 1.0:
                    self.val_check_batch = float("inf")
                else:
                    raise MisconfigurationException(
                        "When using an IterableDataset for `train_dataloader`,"
                        " `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies"
                        " checking validation every k training batches."
                    )
            else:
                self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
                self.val_check_batch = max(1, self.val_check_batch)

        if self.logger and self.num_training_batches < self.log_every_n_steps:
            rank_zero_warn(
                f"The number of training samples ({self.num_training_batches}) is smaller than the logging interval"
                f" Trainer(log_every_n_steps={self.log_every_n_steps}). Set a lower value for log_every_n_steps if"
                " you want to see logs for the training epoch.",
                category=PossibleUserWarning,
            )

    def _reset_eval_dataloader(
        self, mode: RunningStage, model: Optional["pl.LightningModule"] = None
    ) -> Tuple[List[Union[int, float]], List[DataLoader]]:
        """Generic method to reset a dataloader for evaluation.

        Args:
            mode: The running stage of the ``Trainer``
            model: The ``LightningModule`` if calling this outside of the trainer scope.

        Returns:
            Tuple (num_batches, dataloaders)
        """
        assert mode.evaluating or mode == RunningStage.PREDICTING

        # always get the loaders first so we can count how many there are
        dataloaders = self.request_dataloader(mode, model=model)

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        if any(dl is None for dl in dataloaders):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")

        for loader in dataloaders:
            apply_to_collection(
                loader.loaders if isinstance(loader, CombinedLoader) else loader,
                DataLoader,
                self._check_eval_shuffling,
                mode=mode,
            )

        # add samplers
        dataloaders = [self.prepare_dataloader(dl, False, mode=mode) for dl in dataloaders if dl is not None]

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(dataloaders, dtype=DataLoader, function=_auto_add_worker_init_fn, rank=self.global_rank)

        loader_num_batches = []

        # determine number of batches
        # datasets could be none, 1 or 2+
        module = model or self.lightning_module or self.datamodule
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                orig_num_batches = num_batches = (
                    len(dataloader)
                    if has_len_all_ranks(dataloader, self.training_type_plugin, module)
                    else float("inf")
                )
                self._worker_check(dataloader, f"{mode.dataloader_prefix}_dataloader {i}")

                # percent or num_steps
                limit_eval_batches = getattr(self, f"limit_{mode.dataloader_prefix}_batches")

                # limit num batches either as a percent or num steps
                if isinstance(limit_eval_batches, int) or limit_eval_batches == 0.0:
                    num_batches = min(num_batches, int(limit_eval_batches))
                elif num_batches != float("inf"):
                    num_batches = int(num_batches * limit_eval_batches)
                elif limit_eval_batches != 1.0:
                    raise MisconfigurationException(
                        f"When using an IterableDataset for `limit_{mode}_batches`,"
                        f" `Trainer(limit_{mode.dataloader_prefix}_batches)` must be `0.0`, `1.0` or an int. An int k"
                        f" specifies `num_{mode.dataloader_prefix}_batches` to use."
                    )

                if num_batches == 0 and limit_eval_batches > 0.0 and isinstance(limit_eval_batches, float):
                    min_pct = 1.0 / len(dataloader)
                    raise MisconfigurationException(
                        f"you requested to check {limit_eval_batches} of the `{mode.dataloader_prefix}_dataloader` but"
                        f" {limit_eval_batches} * {orig_num_batches} < 1. Please increase the"
                        f" `limit_{mode.dataloader_prefix}_batches` flag. Try at least"
                        f" `limit_{mode.dataloader_prefix}_batches={min_pct}`"
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders

    def reset_val_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The `LightningModule` if called outside of the trainer scope.
        """
        source = self._data_connector._val_dataloader_source
        pl_module = self.lightning_module or model
        has_step = is_overridden("validation_step", pl_module)
        if source.is_defined() and has_step:
            self.num_val_batches, self.val_dataloaders = self._reset_eval_dataloader(
                RunningStage.VALIDATING, model=pl_module
            )

    def reset_test_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the test dataloader and determines the number of batches.

        Args:
            model: The `LightningModule` if called outside of the trainer scope.
        """
        source = self._data_connector._test_dataloader_source
        pl_module = self.lightning_module or model
        has_step = is_overridden("test_step", pl_module)
        if source.is_defined() and has_step:
            self.num_test_batches, self.test_dataloaders = self._reset_eval_dataloader(
                RunningStage.TESTING, model=pl_module
            )

    def reset_predict_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the predict dataloader and determines the number of batches.

        Args:
            model: The `LightningModule` if called outside of the trainer scope.
        """
        source = self._data_connector._predict_dataloader_source
        pl_module = self.lightning_module or model
        if source.is_defined():
            self.num_predict_batches, self.predict_dataloaders = self._reset_eval_dataloader(
                RunningStage.PREDICTING, model=pl_module
            )

    def reset_train_val_dataloaders(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets train and val dataloaders if none are attached to the trainer.

        The val dataloader must be initialized before training loop starts, as the training loop
        inspects the val dataloader to determine whether to run the evaluation loop.

        Args:
            model: The `LightningModule` if called outside of the trainer scope.
        """
        if self.train_dataloader is None:
            self.reset_train_dataloader(model=model)
        if self.val_dataloaders is None:
            self.reset_val_dataloader(model=model)

    def request_dataloader(
        self, stage: RunningStage, model: Optional["pl.LightningModule"] = None
    ) -> Union[DataLoader, List[DataLoader]]:
        """Requests a dataloader from the given model by calling dataloader hooks corresponding to the given stage.

        Returns:
            The requested dataloader
        """
        source = getattr(self._data_connector, f"_{stage.dataloader_prefix}_dataloader_source")

        hook = f"{stage.dataloader_prefix}_dataloader"
        self._call_lightning_module_hook("on_" + hook, pl_module=model)
        with _replace_dataloader_init_method():
            # under this context manager, the arguments passed to `DataLoader.__init__` will be captured and saved as
            # attributes on the instance in case the dataloader needs to be re-instantiated later by Ligtning
            dataloader = source.dataloader()
        if isinstance(dataloader, tuple):
            dataloader = list(dataloader)
        self.training_type_plugin.barrier("get_dataloaders")
        _validate_fault_tolerant_automatic(dataloader, stage)
        return dataloader

    @staticmethod
    def _resolve_overfit_batches(dataloader: Collection[DataLoader]) -> Collection[DataLoader]:
        all_have_sequential_sampler = True

        def resolve_has_no_sequential_sampler(dataloader: DataLoader):
            nonlocal all_have_sequential_sampler
            all_have_sequential_sampler = all_have_sequential_sampler & isinstance(
                dataloader.sampler, SequentialSampler
            )

        apply_to_collection(dataloader, DataLoader, resolve_has_no_sequential_sampler)

        if not all_have_sequential_sampler:
            rank_zero_warn(
                "You requested to overfit but enabled training dataloader shuffling."
                " We are turning off the training dataloader shuffling for you."
            )

            def replace_sampler(dataloader: DataLoader) -> DataLoader:
                return _update_dataloader(dataloader, SequentialSampler(dataloader.dataset), mode=RunningStage.TRAINING)

            dataloader = apply_to_collection(dataloader, DataLoader, replace_sampler)

        return dataloader

    @staticmethod
    def _check_eval_shuffling(dataloader, mode):
        if (
            hasattr(dataloader, "sampler")
            and not isinstance(dataloader.sampler, SequentialSampler)
            and not isinstance(dataloader.dataset, IterableDataset)
        ):
            rank_zero_warn(
                f"Your `{mode.dataloader_prefix}_dataloader` has `shuffle=True`,"
                " it is strongly recommended that you turn this off for val/test/predict dataloaders.",
                category=PossibleUserWarning,
            )
