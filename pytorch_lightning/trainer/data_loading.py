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
import multiprocessing
import os
from abc import ABC
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union

from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper, UnrepeatedDistributedSampler
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.auto_restart import (
    _capture_metadata_collate,
    CaptureIterableDataset,
    CaptureMapDataset,
    FastForwardSampler,
)
from pytorch_lightning.utilities.data import get_len, has_iterable_dataset, has_len_all_ranks
from pytorch_lightning.utilities.enums import DistributedType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.seed import pl_worker_init_function


class TrainerDataLoadingMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    val_check_interval: float
    tpu_local_core_rank: int
    train_dataloader: DataLoader
    num_training_batches: Union[int, float]
    val_check_batch: float
    val_dataloaders: Optional[List[DataLoader]]
    num_val_batches: List[Union[int, float]]
    test_dataloaders: Optional[List[DataLoader]]
    num_test_batches: List[Union[int, float]]
    limit_train_batches: Union[int, float]
    log_every_n_steps: int
    overfit_batches: Union[int, float]
    distributed_sampler_kwargs: dict
    accelerator: Accelerator
    accelerator_connector: AcceleratorConnector
    call_hook: Callable

    def _worker_check(self, dataloader: DataLoader, name: str) -> None:
        if not isinstance(dataloader, DataLoader):
            return

        using_spawn = self._accelerator_connector._distrib_type == DistributedType.DDP_SPAWN
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
                " in the `DataLoader` init to improve performance."
            )

    @staticmethod
    def _auto_add_worker_init_fn(dataloader: DataLoader, rank: int) -> None:
        if int(os.environ.get("PL_SEED_WORKERS", 0)) and dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = partial(pl_worker_init_function, rank=rank)

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
            dataloader = self._update_dataloader(dataloader, sampler, mode=mode)

        if cycle_iterator is not None:
            cycle_iterator.loader = dataloader
            return cycle_iterator

        return dataloader

    def _resolve_sampler(self, dataloader: DataLoader, shuffle: bool, mode: Optional[RunningStage] = None) -> Sampler:
        if self._requires_distributed_sampler(dataloader):
            if not isinstance(dataloader.sampler, (SequentialSampler, RandomSampler)):
                raise MisconfigurationException(
                    "You seem to have configured a sampler in your DataLoader. This will be replaced "
                    " by `DistributedSampler` since `replace_sampler_ddp` is True and you are using"
                    " distributed training. Either remove the sampler from your DataLoader or set"
                    " `replace_sampler_ddp=False` if you want to use your custom sampler."
                )
            return self._get_distributed_sampler(
                dataloader, shuffle, mode=mode, overfit_batches=self.overfit_batches, **self.distributed_sampler_kwargs
            )

        return dataloader.sampler

    @staticmethod
    def _dataloader_init_kwargs_resolve_sampler(
        dataloader: DataLoader, sampler: Optional[Sampler], mode: Optional[RunningStage] = None
    ) -> Dict[str, Any]:
        """This function is used to handle the sampler, batch_sampler arguments associated within a DataLoader for
        its re-instantiation.

        If the dataloader is being used for prediction, the sampler will be wrapped into an `IndexBatchSamplerWrapper`,
        so Lightning can keep track of its indices. If fault tolerant training is enabled, the sampler will be wrapped
        into a `FastForwardSampler`.
        """
        batch_sampler = getattr(dataloader, "batch_sampler")
        is_predicting = mode == RunningStage.PREDICTING
        # checking the batch sampler type is different than PyTorch default.
        if batch_sampler is not None and (type(batch_sampler) is not BatchSampler or is_predicting):
            batch_sampler = type(batch_sampler)(
                sampler,
                batch_size=batch_sampler.batch_size,
                drop_last=(False if is_predicting else batch_sampler.drop_last),
            )
            if is_predicting:
                batch_sampler = IndexBatchSamplerWrapper(batch_sampler)

            if _fault_tolerant_training():
                fast_forward_sampler = batch_sampler = FastForwardSampler(batch_sampler)
                fast_forward_sampler.setup(dataloader_batch_size=1)

            return {
                "sampler": None,
                "shuffle": False,
                "batch_sampler": batch_sampler,
                "batch_size": 1,
                "drop_last": False,
            }

        if _fault_tolerant_training():
            fast_forward_sampler = sampler = FastForwardSampler(sampler)
            fast_forward_sampler.setup(dataloader_batch_size=dataloader.batch_size)

        return {"sampler": sampler, "shuffle": False, "batch_sampler": None}

    @staticmethod
    def _get_dataloader_init_kwargs(
        dataloader: DataLoader, sampler: Optional[Sampler], mode: Optional[RunningStage] = None
    ) -> Dict[str, Any]:
        if not isinstance(dataloader, DataLoader):
            raise ValueError(f"The dataloader {dataloader} needs to subclass `torch.utils.data.DataLoader`")

        # get the dataloader instance attributes
        attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}
        # not part of `vars`
        attrs["multiprocessing_context"] = dataloader.multiprocessing_context

        # get the dataloader instance `__init__` parameters
        params = dict(inspect.signature(dataloader.__init__).parameters)
        has_variadic_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
        if has_variadic_kwargs:
            # if the signature takes **kwargs, assume they will be passed down with `super().__init__(**kwargs)`
            params.update(inspect.signature(DataLoader.__init__).parameters)
            del params["self"]

        # keep only the params whose default is different to the current attr value
        non_defaults = {name for name, p in params.items() if name in attrs and p.default != attrs[name]}
        # add `dataset` as it might have been replaced with `*args`
        non_defaults.add("dataset")

        # kwargs to re-construct the dataloader
        dl_kwargs = {k: v for k, v in attrs.items() if k in non_defaults}
        dl_kwargs.update(
            TrainerDataLoadingMixin._dataloader_init_kwargs_resolve_sampler(dataloader, sampler, mode=mode)
        )

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

        if isinstance(dl_kwargs["dataset"], IterableDataset):
            dl_kwargs["batch_sampler"] = None
            dl_kwargs["sampler"] = None

        if _fault_tolerant_training():
            dataset = dl_kwargs["dataset"]
            if isinstance(dataset, IterableDataset):
                # wrap the `IterableDataset` into a `CaptureIterableDataset` to record sampler states.
                dl_kwargs["dataset"] = CaptureIterableDataset(dataset=dl_kwargs["dataset"])
            elif get_len(dataset) != float("inf"):
                dl_kwargs["dataset"] = CaptureMapDataset(dataset=dl_kwargs["dataset"])
            else:
                raise MisconfigurationException(
                    "This shouldn't happen, please open an issue on Lightning Github repository."
                )

        return dl_kwargs

    @staticmethod
    def _update_dataloader(dataloader: DataLoader, sampler: Sampler, mode: Optional[RunningStage] = None) -> DataLoader:
        dl_kwargs = TrainerDataLoadingMixin._get_dataloader_init_kwargs(dataloader, sampler, mode=mode)
        dl_cls = type(dataloader)
        dataloader = dl_cls(**dl_kwargs)
        return dataloader

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
        apply_to_collection(self.train_dataloader, DataLoader, self._auto_add_worker_init_fn, rank=self.global_rank)

        # add collate_fn to collect metadata for fault tolerant training
        if _fault_tolerant_training():
            apply_to_collection(self.train_dataloader, DataLoader, self._add_sampler_metadata_collate)

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
                " you want to see logs for the training epoch."
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

        # when overfitting, use the training loader as val and test
        # duplicate it the numb of times needed to match the train loaders
        if self.overfit_batches > 0:
            train_dataloader = self.request_dataloader(RunningStage.TRAINING, model=model)
            dataloaders = [deepcopy(train_dataloader) for _ in range(len(dataloaders))]

        for loader_i in range(len(dataloaders)):
            loader = dataloaders[loader_i]

            if hasattr(loader, "sampler") and not isinstance(loader.sampler, SequentialSampler):
                # when overfitting, the dataloader should not have sampler
                if self.overfit_batches > 0 and mode.evaluating:
                    rank_zero_warn(
                        "You requested to overfit but enabled val/test dataloader shuffling."
                        " We are turning it off for you."
                    )
                    dataloaders[loader_i] = self._update_dataloader(
                        loader, SequentialSampler(loader.dataset), mode=mode
                    )
                else:
                    rank_zero_warn(
                        f"Your `{mode.dataloader_prefix}_dataloader` has `shuffle=True`,"
                        "it is strongly recommended that you turn this off for val/test/predict dataloaders."
                    )

        if any(dl is None for dl in dataloaders):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")

        # add samplers
        dataloaders = [self.prepare_dataloader(dl, False, mode=mode) for dl in dataloaders if dl is not None]

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(
            dataloaders, dtype=DataLoader, function=self._auto_add_worker_init_fn, rank=self.global_rank
        )

        loader_num_batches = []

        # determine number of batches
        # datasets could be none, 1 or 2+
        module = model or self.lightning_module or self.datamodule
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                num_batches = (
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
                        f" {limit_eval_batches}*{num_batches} < 1. Please increase the"
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
        self.call_hook("on_" + hook, pl_module=model)
        dataloader = source.dataloader()
        if isinstance(dataloader, tuple):
            dataloader = list(dataloader)
        self.training_type_plugin.barrier("get_dataloaders")
        return dataloader

    @staticmethod
    def _add_sampler_metadata_collate(dataloader: DataLoader) -> None:
        """Wrap default collate function to retrive ``FastForwardSampler`` state dict when fault tolerant is
        enabled."""
        dataloader.collate_fn = partial(
            _capture_metadata_collate, dataset=dataloader.dataset, default_collate=dataloader.collate_fn
        )

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
                return TrainerDataLoadingMixin._update_dataloader(
                    dataloader, SequentialSampler(dataloader.dataset), mode=RunningStage.TRAINING
                )

            dataloader = apply_to_collection(dataloader, DataLoader, replace_sampler)

        return dataloader
