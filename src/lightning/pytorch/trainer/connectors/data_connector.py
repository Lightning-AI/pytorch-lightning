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
from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import lightning.pytorch as pl
from lightning.fabric.utilities.data import (
    _auto_add_worker_init_fn,
    _replace_dunder_methods,
    _set_sampler_epoch,
    has_iterable_dataset,
)
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from lightning.pytorch.overrides.distributed import UnrepeatedDistributedSamplerWrapper
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.data import _is_dataloader_shuffled, _update_dataloader
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _LIGHTNING_GRAPHCORE_AVAILABLE
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn, WarningCache
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lightning.pytorch.utilities.warnings import PossibleUserWarning

warning_cache = WarningCache()


class _DataConnector:
    def __init__(self, trainer: pl.Trainer):
        self.trainer = trainer
        self._datahook_selector: _DataHookSelector | None = None

    def on_trainer_init(
        self,
        val_check_interval: int | float | None,
        reload_dataloaders_every_n_epochs: int,
        check_val_every_n_epoch: int | None,
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

    def prepare_data(self) -> None:
        trainer = self.trainer

        # on multi-gpu jobs we only want to manipulate (download, etc) on node_rank=0, local_rank=0
        # or in the case where each node needs to do its own manipulation in which case just local_rank=0
        local_rank_zero = trainer.local_rank == 0
        global_rank_zero = trainer.local_rank == 0 and trainer.node_rank == 0

        datamodule = trainer.datamodule
        lightning_module = trainer.lightning_module
        # handle datamodule prepare data:
        # check for prepare_data_per_node & datamodule lifecycle properties before calling datamodule.prepare_data
        if datamodule is not None:
            dm_prepare_data_per_node = datamodule.prepare_data_per_node
            if (dm_prepare_data_per_node and local_rank_zero) or (not dm_prepare_data_per_node and global_rank_zero):
                call._call_lightning_datamodule_hook(trainer, "prepare_data")
        # handle lightning module prepare data:
        # check for prepare_data_per_node before calling lightning_module.prepare_data
        if lightning_module is not None:
            lm_prepare_data_per_node = lightning_module.prepare_data_per_node
            if (lm_prepare_data_per_node and local_rank_zero) or (not lm_prepare_data_per_node and global_rank_zero):
                call._call_lightning_module_hook(trainer, "prepare_data")

    def attach_data(
        self,
        model: pl.LightningModule,
        train_dataloaders: TRAIN_DATALOADERS | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        test_dataloaders: EVAL_DATALOADERS | None = None,
        predict_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: pl.LightningDataModule | None = None,
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

        # Attach the trainer to the LightningModule
        model.trainer = self.trainer

    def attach_dataloaders(
        self,
        model: pl.LightningModule,
        train_dataloaders: TRAIN_DATALOADERS | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        test_dataloaders: EVAL_DATALOADERS | None = None,
        predict_dataloaders: EVAL_DATALOADERS | None = None,
    ) -> None:
        trainer = self.trainer

        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.epoch_loop.val_loop._combined_loader = None
        trainer.validate_loop._combined_loader = None
        trainer.test_loop._combined_loader = None
        trainer.predict_loop._combined_loader = None

        trainer.fit_loop._data_source.instance = train_dataloaders if train_dataloaders is not None else model
        trainer.fit_loop.epoch_loop.val_loop._data_source.instance = (
            val_dataloaders if val_dataloaders is not None else model
        )
        trainer.validate_loop._data_source.instance = val_dataloaders if val_dataloaders is not None else model
        trainer.test_loop._data_source.instance = test_dataloaders if test_dataloaders is not None else model
        trainer.predict_loop._data_source.instance = predict_dataloaders if predict_dataloaders is not None else model

    def attach_datamodule(self, model: pl.LightningModule, datamodule: pl.LightningDataModule | None = None) -> None:
        # If we have a datamodule, attach necessary hooks + dataloaders
        self._datahook_selector = _DataHookSelector(model, datamodule)

        if datamodule is None:
            return

        trainer = self.trainer
        trainer.fit_loop._data_source.instance = datamodule
        trainer.fit_loop.epoch_loop.val_loop._data_source.instance = datamodule
        trainer.validate_loop._data_source.instance = datamodule
        trainer.test_loop._data_source.instance = datamodule
        trainer.predict_loop._data_source.instance = datamodule

        trainer.datamodule = datamodule
        datamodule.trainer = trainer

    def _requires_distributed_sampler(self, dataloader: DataLoader) -> bool:
        if _LIGHTNING_GRAPHCORE_AVAILABLE:
            from lightning_graphcore import IPUAccelerator

            # `DistributedSampler` is never used with `poptorch.DataLoader`
            is_ipu = isinstance(self.trainer.accelerator, IPUAccelerator)
        else:
            is_ipu = False
        return (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed
            and not isinstance(dataloader.sampler, DistributedSampler)
            and not has_iterable_dataset(dataloader)
            and not is_ipu
        )

    def _prepare_dataloader(self, dataloader: object, shuffle: bool, mode: RunningStage) -> object:
        """This function handles the following functionalities:

        - Injecting a `DistributedDataSamplerWrapper` into the `DataLoader` if on a distributed environment
        - Wrapping the dataloader based on strategy-specific logic
        """
        # don't do anything if it's not a dataloader
        if not isinstance(dataloader, DataLoader):
            return dataloader

        if _LIGHTNING_GRAPHCORE_AVAILABLE:
            from lightning_graphcore import IPUAccelerator

            # IPUs use a custom `poptorch.DataLoader` which we might need to convert to
            is_ipu = isinstance(self.trainer.accelerator, IPUAccelerator)
        else:
            is_ipu = False
        if (
            self._requires_distributed_sampler(dataloader)  # sets the distributed sampler
            or mode == RunningStage.PREDICTING  # to track indices for the predictions
            or is_ipu
        ):
            sampler = self._resolve_sampler(dataloader, shuffle=shuffle, mode=mode)
            return _update_dataloader(dataloader, sampler, mode=mode)

        return dataloader

    def _resolve_sampler(
        self, dataloader: DataLoader, shuffle: bool, mode: RunningStage | None = None
    ) -> Sampler | Iterable:
        if self._requires_distributed_sampler(dataloader):
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            assert distributed_sampler_kwargs is not None
            sampler = _get_distributed_sampler(
                dataloader,
                shuffle,
                mode=mode,
                overfit_batches=self.trainer.overfit_batches,
                **distributed_sampler_kwargs,
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


def _get_distributed_sampler(
    dataloader: DataLoader,
    shuffle: bool,
    overfit_batches: int | float,
    mode: RunningStage | None = None,
    **kwargs: Any,
) -> DistributedSampler:
    """This function is used to created the distributed sampler injected within the user DataLoader."""
    kwargs["shuffle"] = shuffle and not overfit_batches
    kwargs.setdefault("seed", int(os.getenv("PL_GLOBAL_SEED", 0)))
    if mode == RunningStage.PREDICTING:
        return UnrepeatedDistributedSamplerWrapper(dataloader.sampler, **kwargs)
    if isinstance(dataloader.sampler, (RandomSampler, SequentialSampler)):
        return DistributedSampler(dataloader.dataset, **kwargs)
    return DistributedSamplerWrapper(dataloader.sampler, **kwargs)


def _resolve_overfit_batches(combined_loader: CombinedLoader, mode: RunningStage) -> None:
    all_have_sequential_sampler = all(
        isinstance(dl.sampler, SequentialSampler) for dl in combined_loader.flattened if hasattr(dl, "sampler")
    )
    if all_have_sequential_sampler:
        return
    rank_zero_warn(
        f"You requested to overfit but enabled {mode.dataloader_prefix} dataloader shuffling."
        f" We are turning off the {mode.dataloader_prefix} dataloader shuffling for you."
    )
    updated = [
        _update_dataloader(dl, sampler=SequentialSampler(dl.dataset), mode=mode) if hasattr(dl, "dataset") else dl
        for dl in combined_loader.flattened
    ]
    combined_loader.flattened = updated


@dataclass
class _DataLoaderSource:
    """Stores the information where the dataloaders come from.

    The source can be

    1. from a ``*_dataloader()`` method on the :class:`~lightning.pytorch.core.module.LightningModule`,
    2. from a ``*_dataloader()`` method on the :class:`~lightning.pytorch.core.datamodule.LightningDataModule`,
    3. a direct instance of a :class:`~torch.utils.data.DataLoader` or supported collections thereof.

    Arguments:
        instance: A LightningModule, LightningDataModule, or (a collection of) iterable(s).
        name: A name for this dataloader source. If the instance is a module, the name corresponds to the hook
            that returns the desired dataloader(s).
    """

    instance: TRAIN_DATALOADERS | EVAL_DATALOADERS | pl.LightningModule | pl.LightningDataModule | None
    name: str

    def dataloader(self) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
        """Returns the dataloader from the source.

        If the source is a module, the method with the corresponding :attr:`name` gets called.
        """
        if isinstance(self.instance, pl.LightningModule):
            return call._call_lightning_module_hook(self.instance.trainer, self.name, pl_module=self.instance)
        if isinstance(self.instance, pl.LightningDataModule):
            assert self.instance.trainer is not None
            return call._call_lightning_datamodule_hook(self.instance.trainer, self.name)
        assert self.instance is not None
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
        return isinstance(self.instance, (pl.LightningModule, pl.LightningDataModule))


def _request_dataloader(data_source: _DataLoaderSource) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
    """Requests a dataloader by calling dataloader hooks corresponding to the given stage.

    Returns:
        The requested dataloader
    """
    with _replace_dunder_methods(DataLoader, "dataset"), _replace_dunder_methods(BatchSampler):
        # under this context manager, the arguments passed to `DataLoader.__init__` will be captured and saved as
        # attributes on the instance in case the dataloader needs to be re-instantiated later by Lightning.
        # Also, it records all attribute setting and deletion using patched `__setattr__` and `__delattr__`
        # methods so that the re-instantiated object is as close to the original as possible.
        return data_source.dataloader()


@dataclass
class _DataHookSelector:
    """Stores the info about the shared DataHooks within ``LightningModule`` and ``LightningDataModule``.

    The hook source can be:

    1. the :class:`~lightning.pytorch.core.module.LightningModule`,
    2. the :class:`~lightning.pytorch.core.datamodule.LightningDataModule`,

    Arguments:
        model: A ``LightningModule``
        datamodule: A ``LightningDataModule``
    """

    model: pl.LightningModule
    datamodule: pl.LightningDataModule | None
    _valid_hooks: tuple[str, ...] = field(
        default=("on_before_batch_transfer", "transfer_batch_to_device", "on_after_batch_transfer")
    )

    def get_instance(self, hook_name: str) -> pl.LightningModule | pl.LightningDataModule:
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


def _check_dataloader_iterable(
    dataloader: object,
    source: _DataLoaderSource,
    trainer_fn: TrainerFn,
) -> None:
    try:
        iter(dataloader)  # type: ignore[call-overload]
    except TypeError:
        # A prefix in the message to disambiguate between the train- and (optional) val dataloader that .fit() accepts
        prefix = "train_" if trainer_fn == TrainerFn.FITTING else ""
        if not source.is_module():
            raise TypeError(
                f"An invalid dataloader was passed to `Trainer.{trainer_fn}({prefix}dataloaders=...)`."
                f" Found {dataloader}."
            )
        if not is_overridden(source.name, source.instance):
            raise TypeError(
                f"An invalid dataloader was passed to `Trainer.{trainer_fn}({prefix}dataloaders=...)`."
                f" Found {dataloader}."
                f" Either pass the dataloader to the `.{trainer_fn}()` method OR implement"
                f" `def {source.name}(self):` in your LightningModule/LightningDataModule."
            )
        raise TypeError(
            f"An invalid dataloader was returned from `{type(source.instance).__name__}.{source.name}()`."
            f" Found {dataloader}."
        )


def _worker_check(dataloader: object, using_spawn: bool, name: str) -> None:
    if not isinstance(dataloader, DataLoader):
        return

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


def _parse_num_batches(stage: RunningStage, length: int | float, limit_batches: int | float) -> int | float:
    if length == 0:
        return int(length)

    num_batches = length
    # limit num batches either as a percent or num steps
    if isinstance(limit_batches, int):
        num_batches = min(length, limit_batches)
    elif isinstance(limit_batches, float) and length != float("inf"):
        num_batches = int(length * limit_batches)
    elif limit_batches != 1.0:
        raise MisconfigurationException(
            f"When using an `IterableDataset`, `Trainer(limit_{stage.dataloader_prefix}_batches)` must be"
            f" `1.0` or an int. An int specifies `num_{stage.dataloader_prefix}_batches` to use."
        )

    if num_batches == 0 and limit_batches > 0.0 and isinstance(limit_batches, float) and length != float("inf"):
        min_percentage = 1.0 / length
        raise MisconfigurationException(
            f"You requested to check {limit_batches} of the `{stage.dataloader_prefix}_dataloader` but"
            f" {limit_batches} * {length} < 1. Please increase the"
            f" `limit_{stage.dataloader_prefix}_batches` argument. Try at least"
            f" `limit_{stage.dataloader_prefix}_batches={min_percentage}`"
        )
    return num_batches


def _process_dataloader(trainer: pl.Trainer, trainer_fn: TrainerFn, stage: RunningStage, dataloader: object) -> object:
    if stage != RunningStage.TRAINING:
        is_shuffled = _is_dataloader_shuffled(dataloader)
        # limit this warning only for samplers assigned automatically when shuffle is set
        if is_shuffled:
            rank_zero_warn(
                f"Your `{stage.dataloader_prefix}_dataloader`'s sampler has shuffling enabled,"
                " it is strongly recommended that you turn shuffling off for val/test dataloaders.",
                category=PossibleUserWarning,
            )
    else:
        is_shuffled = True

    # automatically add samplers
    dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=is_shuffled, mode=stage)

    # let the strategy inject its logic
    strategy = trainer.strategy
    dataloader = strategy.process_dataloader(dataloader)

    # check the workers
    _worker_check(
        dataloader,
        isinstance(strategy, DDPStrategy) and strategy._start_method == "spawn",
        f"{stage.dataloader_prefix}_dataloader",
    )

    # add worker_init_fn for correct seeding in worker processes
    _auto_add_worker_init_fn(dataloader, trainer.global_rank)

    if trainer_fn != TrainerFn.FITTING:  # if we are fitting, we need to do this in the loop
        # some users want validation shuffling based on the training progress
        _set_sampler_epoch(dataloader, trainer.fit_loop.epoch_progress.current.processed)

    return dataloader
