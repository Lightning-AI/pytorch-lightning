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
from dataclasses import dataclass, field
from typing import Any, Collection, List, Optional, Tuple, Union
from weakref import proxy

from torch.utils.data import BatchSampler, DataLoader, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.accelerators.ipu import IPUAccelerator
from pytorch_lightning.overrides.distributed import DistributedSamplerWrapper, UnrepeatedDistributedSamplerWrapper
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.auto_restart import _validate_fault_tolerant_automatic
from pytorch_lightning.utilities.data import (
    _auto_add_worker_init_fn,
    _is_dataloader_shuffled,
    _replace_dunder_methods,
    _update_dataloader,
    has_iterable_dataset,
    has_len_all_ranks,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities.warnings import PossibleUserWarning, WarningCache

warning_cache = WarningCache()


class DataConnector:
    def __init__(self, trainer: "pl.Trainer", multiple_trainloader_mode: str = "max_size_cycle"):
        self.trainer = trainer
        self.multiple_trainloader_mode = multiple_trainloader_mode
        self._train_dataloader_source = _DataLoaderSource(None, "")
        self._val_dataloader_source = _DataLoaderSource(None, "")
        self._test_dataloader_source = _DataLoaderSource(None, "")
        self._predict_dataloader_source = _DataLoaderSource(None, "")

        self._datahook_selector = _DataHookSelector(None, None)

    @property
    def _should_reload_train_dl(self) -> bool:
        """Check if train dataloader should be reloaded."""
        n_epochs = self.trainer.reload_dataloaders_every_n_epochs
        return n_epochs and (
            self.trainer._last_train_dl_reload_epoch is None
            or self.trainer.current_epoch - self.trainer._last_train_dl_reload_epoch >= n_epochs
        )

    @property
    def _should_reload_val_dl(self) -> bool:
        """Check if validation dataloader should be reloaded."""
        n_epochs = self.trainer.reload_dataloaders_every_n_epochs
        return n_epochs and (
            self.trainer._last_val_dl_reload_epoch is None
            or self.trainer.current_epoch - self.trainer._last_val_dl_reload_epoch >= n_epochs
        )

    def on_trainer_init(
        self,
        val_check_interval: Union[int, float],
        reload_dataloaders_every_n_epochs: int,
        check_val_every_n_epoch: Optional[int],
    ) -> None:
        self.trainer.datamodule = None

        if check_val_every_n_epoch is not None and not isinstance(check_val_every_n_epoch, int):
            raise MisconfigurationException(
                f"`check_val_every_n_epoch` should be an integer, found {check_val_every_n_epoch!r}."
            )

        if check_val_every_n_epoch is None and isinstance(val_check_interval, float):
            raise MisconfigurationException(
                "`val_check_interval` should be an integer when `check_val_every_n_epoch=None`,"
                f" found {val_check_interval!r}."
            )

        self.trainer.check_val_every_n_epoch = check_val_every_n_epoch

        if not isinstance(reload_dataloaders_every_n_epochs, int) or (reload_dataloaders_every_n_epochs < 0):
            raise MisconfigurationException(
                f"`reload_dataloaders_every_n_epochs` should be an int >= 0, got {reload_dataloaders_every_n_epochs}."
            )

        self.trainer.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.trainer._is_data_prepared = False

    def prepare_data(self) -> None:
        # on multi-gpu jobs we only want to manipulate (download, etc) on node_rank=0, local_rank=0
        # or in the case where each node needs to do its own manipulation in which case just local_rank=0
        local_rank_zero = self.trainer.local_rank == 0
        global_rank_zero = self.trainer.local_rank == 0 and self.trainer.node_rank == 0

        datamodule = self.trainer.datamodule
        lightning_module = self.trainer.lightning_module
        # handle datamodule prepare data:
        # check for prepare_data_per_node & datamodule lifecycle properties before calling datamodule.prepare_data
        if datamodule is not None:
            dm_prepare_data_per_node = datamodule.prepare_data_per_node
            if (dm_prepare_data_per_node and local_rank_zero) or (not dm_prepare_data_per_node and global_rank_zero):
                self.trainer._call_lightning_datamodule_hook("prepare_data")
        # handle lightning module prepare data:
        # check for prepare_data_per_node before calling lightning_module.prepare_data
        if lightning_module is not None:
            lm_prepare_data_per_node = lightning_module.prepare_data_per_node
            if (lm_prepare_data_per_node and local_rank_zero) or (not lm_prepare_data_per_node and global_rank_zero):
                self.trainer._call_lightning_module_hook("prepare_data")
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

        # Validate that the required data sources are available
        if self.trainer.state.fn == TrainerFn.FITTING:
            _check_dataloader_none(train_dataloaders, self._train_dataloader_source, self.trainer.state.fn)
        elif self.trainer.state.fn == TrainerFn.VALIDATING:
            _check_dataloader_none(val_dataloaders, self._val_dataloader_source, self.trainer.state.fn)
        elif self.trainer.state.fn == TrainerFn.TESTING:
            _check_dataloader_none(test_dataloaders, self._test_dataloader_source, self.trainer.state.fn)
        elif self.trainer.state.fn == TrainerFn.PREDICTING:
            _check_dataloader_none(predict_dataloaders, self._predict_dataloader_source, self.trainer.state.fn)

        # set local properties on the model
        self._copy_trainer_model_properties(model)

    def _copy_trainer_model_properties(self, model: "pl.LightningModule") -> None:
        model.trainer = proxy(self.trainer)
        # Remove setting use_amp in v1.8
        model._use_amp = self.trainer.amp_backend is not None
        model.precision = self.trainer.precision

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
        self._datahook_selector = _DataHookSelector(model, datamodule)

        if datamodule is None:
            return

        self._train_dataloader_source = _DataLoaderSource(datamodule, "train_dataloader")
        self._val_dataloader_source = _DataLoaderSource(datamodule, "val_dataloader")
        self._test_dataloader_source = _DataLoaderSource(datamodule, "test_dataloader")
        self._predict_dataloader_source = _DataLoaderSource(datamodule, "predict_dataloader")

        self.trainer.datamodule = datamodule
        datamodule.trainer = self.trainer

    def _worker_check(self, dataloader: DataLoader, name: str) -> None:
        if not isinstance(dataloader, DataLoader):
            return

        using_spawn = isinstance(self.trainer.strategy, DDPSpawnStrategy)
        num_cpus = multiprocessing.cpu_count()

        # ddp_spawn + num_workers > 0 don't mix! tell the user
        if dataloader.num_workers > 0 and using_spawn:
            if not dataloader.persistent_workers:
                rank_zero_warn(
                    "num_workers>0, persistent_workers=False, and strategy=ddp_spawn"
                    " may result in data loading bottlenecks."
                    " Consider setting persistent_workers=True"
                    " (this is a limitation of Python .spawn() and PyTorch)"
                )

        elif dataloader.num_workers == 0 and using_spawn:
            if not dataloader.persistent_workers:
                rank_zero_warn(
                    "strategy=ddp_spawn and num_workers=0 may result in data loading bottlenecks."
                    " Consider setting num_workers>0 and persistent_workers=True"
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
            self.trainer._accelerator_connector.replace_sampler_ddp
            and self.trainer._accelerator_connector.is_distributed
            and not isinstance(dataloader.sampler, DistributedSampler)
            and not has_iterable_dataset(dataloader)
            # `DistributedSampler` is never used with `poptorch.DataLoader`
            and not isinstance(self.trainer.accelerator, IPUAccelerator)
        )

    # TODO: shuffle here is kept for BC. Remove it once data_loading.py is removed (#11248)
    def _prepare_dataloader(
        self, dataloader: Any, shuffle: Optional[bool] = None, mode: Optional[RunningStage] = None
    ) -> Any:
        """This function handles the following functionalities:

        - Injecting a `DistributedDataSamplerWrapper` into the `DataLoader` if on a distributed environment
        - Wrapping the datasets and samplers into fault-tolerant components
        - Wrapping the dataloader based on strategy-specific logic
        """
        if isinstance(dataloader, CombinedLoader):
            # apply `_prepare_dataloader` on all the collection of loaders
            dataloader.loaders = apply_to_collection(
                dataloader.loaders, (DataLoader, CycleIterator), self._prepare_dataloader, shuffle, mode=mode
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
            # IPUs use a custom `poptorch.DataLoader` which we might need to convert to
            or isinstance(self.trainer.accelerator, IPUAccelerator)
        ):
            if shuffle is None:
                # for training, set to True always
                # for evaluation, decide based on existing sampler
                shuffle = True if mode == RunningStage.TRAINING else _is_dataloader_shuffled(dataloader)

            sampler = self._resolve_sampler(dataloader, shuffle=shuffle, mode=mode)
            dataloader = _update_dataloader(dataloader, sampler, mode=mode)

        dataloader = self.trainer.strategy.process_dataloader(dataloader)

        if cycle_iterator is not None:
            cycle_iterator.loader = dataloader
            return cycle_iterator

        return dataloader

    def _resolve_sampler(self, dataloader: DataLoader, shuffle: bool, mode: Optional[RunningStage] = None) -> Sampler:
        if self._requires_distributed_sampler(dataloader):
            sampler = self._get_distributed_sampler(
                dataloader,
                shuffle,
                mode=mode,
                overfit_batches=self.trainer.overfit_batches,
                **self.trainer.distributed_sampler_kwargs,
            )

            # update docs too once this is resolved
            trainer_fn = self.trainer.state.fn
            if (
                isinstance(sampler, DistributedSampler)
                and sampler.num_replicas > 1
                and trainer_fn in (TrainerFn.VALIDATING, TrainerFn.TESTING)
            ):
                rank_zero_warn(
                    f"Using `DistributedSampler` with the dataloaders. During `trainer.{trainer_fn.value}()`, it is"
                    " recommended to use `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated"
                    " exactly once. Otherwise, multi-device settings use `DistributedSampler` that replicates"
                    " some samples to make sure all devices have same batch size in case of uneven inputs.",
                    category=PossibleUserWarning,
                )

            return sampler

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
        cls = UnrepeatedDistributedSamplerWrapper if mode == RunningStage.PREDICTING else DistributedSamplerWrapper
        sampler = cls(dataloader.sampler, **kwargs)
        return sampler

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
        dataloaders = self._request_dataloader(mode)

        if self.trainer.overfit_batches > 0:
            dataloaders = self._resolve_overfit_batches(dataloaders, mode)

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
        dataloaders = [self._prepare_dataloader(dl, mode=mode) for dl in dataloaders if dl is not None]

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(
            dataloaders, dtype=DataLoader, function=_auto_add_worker_init_fn, rank=self.trainer.global_rank
        )

        loader_num_batches = []

        # determine number of batches
        module = model or self.trainer.lightning_module or self.datamodule
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                orig_num_batches = num_batches = (
                    len(dataloader) if has_len_all_ranks(dataloader, self.trainer.strategy, module) else float("inf")
                )

                if orig_num_batches == 0:
                    loader_num_batches.append(orig_num_batches)
                    continue

                self._worker_check(dataloader, f"{mode.dataloader_prefix}_dataloader {i}")

                # percent or num_steps
                limit_eval_batches = getattr(self.trainer, f"limit_{mode.dataloader_prefix}_batches")

                # limit num batches either as a percent or num steps
                if isinstance(limit_eval_batches, int):
                    num_batches = min(orig_num_batches, limit_eval_batches)
                elif isinstance(limit_eval_batches, float) and orig_num_batches != float("inf"):
                    num_batches = int(orig_num_batches * limit_eval_batches)
                elif limit_eval_batches != 1.0:
                    raise MisconfigurationException(
                        f"When using an `IterableDataset`, `Trainer(limit_{mode.dataloader_prefix}_batches)` must be"
                        f" `1.0` or an int. An int specifies `num_{mode.dataloader_prefix}_batches` to use."
                    )

                if (
                    num_batches == 0
                    and limit_eval_batches > 0.0
                    and isinstance(limit_eval_batches, float)
                    and orig_num_batches != float("inf")
                ):
                    min_percentage = 1.0 / orig_num_batches
                    raise MisconfigurationException(
                        f"You requested to check {limit_eval_batches} of the `{mode.dataloader_prefix}_dataloader` but"
                        f" {limit_eval_batches} * {orig_num_batches} < 1. Please increase the"
                        f" `limit_{mode.dataloader_prefix}_batches` argument. Try at least"
                        f" `limit_{mode.dataloader_prefix}_batches={min_percentage}`"
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders

    def _request_dataloader(self, stage: RunningStage) -> Union[DataLoader, List[DataLoader]]:
        """Requests a dataloader from the given model by calling dataloader hooks corresponding to the given stage.

        Returns:
            The requested dataloader
        """
        source = getattr(self, f"_{stage.dataloader_prefix}_dataloader_source")

        with _replace_dunder_methods(DataLoader, "dataset"), _replace_dunder_methods(BatchSampler):
            # under this context manager, the arguments passed to `DataLoader.__init__` will be captured and saved as
            # attributes on the instance in case the dataloader needs to be re-instantiated later by Lightning.
            # Also, it records all attribute setting and deletion using patched `__setattr__` and `__delattr__`
            # methods so that the re-instantiated object is as close to the original as possible.
            dataloader = source.dataloader()
        if isinstance(dataloader, tuple):
            dataloader = list(dataloader)
        self.trainer.strategy.barrier("get_dataloaders")
        _validate_fault_tolerant_automatic(dataloader, stage)
        return dataloader

    @staticmethod
    def _resolve_overfit_batches(dataloaders: Collection[DataLoader], mode: RunningStage) -> Collection[DataLoader]:
        all_have_sequential_sampler = True

        def resolve_has_no_sequential_sampler(dataloader: DataLoader):
            nonlocal all_have_sequential_sampler
            all_have_sequential_sampler = all_have_sequential_sampler & isinstance(
                dataloader.sampler, SequentialSampler
            )

        apply_to_collection(dataloaders, DataLoader, resolve_has_no_sequential_sampler)

        if not all_have_sequential_sampler:
            rank_zero_warn(
                "You requested to overfit but enabled training dataloader shuffling."
                f" We are turning off the {mode.dataloader_prefix} dataloader shuffling for you."
            )

            def replace_sampler(dataloader: DataLoader) -> DataLoader:
                return _update_dataloader(dataloader, sampler=SequentialSampler(dataloader.dataset), mode=mode)

            dataloaders = apply_to_collection(dataloaders, DataLoader, replace_sampler)

        return dataloaders

    @staticmethod
    def _check_eval_shuffling(dataloader, mode):
        # limit this warning only for samplers assigned automatically when shuffle is set
        if _is_dataloader_shuffled(dataloader):
            rank_zero_warn(
                f"Your `{mode.dataloader_prefix}_dataloader`'s sampler has shuffling enabled,"
                " it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.",
                category=PossibleUserWarning,
            )


@dataclass
class _DataLoaderSource:
    """Stores the information where the dataloaders come from.

    The source can be

    1. from a ``*_datalaoder()`` method on the :class:`~pytorch_lightning.core.module.LightningModule`,
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
            return self.instance.trainer._call_lightning_module_hook(self.name, pl_module=self.instance)

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
        """Returns whether the DataLoader source is a LightningModule or a LightningDataModule.

        It does not check whether ``*_dataloader`` methods are actually overridden.
        """
        from pytorch_lightning import LightningDataModule, LightningModule  # prevent cyclic import

        return isinstance(self.instance, (LightningModule, LightningDataModule))


@dataclass
class _DataHookSelector:
    """Stores the info about the shared DataHooks within ``LightningModule`` and ``LightningDataModule``.

    The hook source can be:

    1. the :class:`~pytorch_lightning.core.module.LightningModule`,
    2. the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`,

    Arguments:
        model: A ``LightningModule``
        datamodule: A ``LightningDataModule``
    """

    model: "pl.LightningModule"
    datamodule: Optional["pl.LightningDataModule"]
    _valid_hooks: Tuple[str] = field(
        default=("on_before_batch_transfer", "transfer_batch_to_device", "on_after_batch_transfer")
    )

    def get_instance(self, hook_name: str) -> Union["pl.LightningModule", "pl.LightningDataModule"]:
        if hook_name not in self._valid_hooks:
            raise ValueError(
                f"`{hook_name}` is not a shared hook within `LightningModule` and `LightningDataModule`."
                f" Valid hooks are {self._valid_hooks}."
            )

        if self.datamodule is None:
            return self.model

        if is_overridden(hook_name, self.datamodule):
            if is_overridden(hook_name, self.model):
                warning_cache.warn(
                    f"You have overridden `{hook_name}` in both `LightningModule` and `LightningDataModule`."
                    " It will use the implementation from `LightningDataModule` instance."
                )
            return self.datamodule

        if is_overridden(hook_name, self.model):
            warning_cache.warn(
                f"You have overridden `{hook_name}` in `LightningModule` but have passed in a"
                " `LightningDataModule`. It will use the implementation from `LightningModule` instance."
            )
        return self.model


def _check_dataloader_none(
    dataloader: Optional[Union[TRAIN_DATALOADERS, EVAL_DATALOADERS]],
    dataloader_source: _DataLoaderSource,
    trainer_fn: TrainerFn,
) -> None:
    # A prefix in the message to disambiguate between the train- and (optional) val dataloader that .fit() accepts
    prefix = "train_" if trainer_fn == TrainerFn.FITTING else ""
    if dataloader is None and not dataloader_source.is_defined():
        raise ValueError(
            f"An invalid dataloader was passed to `Trainer.{trainer_fn}({prefix}dataloaders=...)`."
            f" Either pass the dataloader to the `.{trainer_fn}()` method OR implement"
            f" `def {dataloader_source.name}(self):` in your LightningModule/LightningDataModule."
        )
