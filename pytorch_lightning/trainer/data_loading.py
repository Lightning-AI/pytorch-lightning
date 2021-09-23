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
from copy import deepcopy
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.auto_restart import (
    _capture_metadata_collate,
)
from pytorch_lightning.utilities.dataloader_preparation import (
    _get_dataloader_init_kwargs, 
    _get_distributed_sampler,

)
from pytorch_lightning.utilities.data import has_iterable_dataset, has_len
from pytorch_lightning.utilities.debugging import InternalDebugger
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
    dev_debugger: InternalDebugger
    call_hook: Callable

    def _worker_check(self, dataloader: DataLoader, name: str) -> None:
        if not isinstance(dataloader, DataLoader):
            return

        using_spawn = self.accelerator_connector.distributed_backend == "ddp_spawn"
        num_cpus = multiprocessing.cpu_count()

        # ddp_spawn + num_workers > 0 don't mix! tell the user
        if dataloader.num_workers > 0 and using_spawn:
            # checks for the attr persistent_workers available in pytorch >= 1.7
            if hasattr(dataloader, "persistent_workers"):
                if not dataloader.persistent_workers:
                    rank_zero_warn(
                        "num_workers>0, persistent_workers=False, and accelerator=ddp_spawn"
                        " may result in data loading bottlenecks."
                        " Consider setting persistent_workers=True"
                        " (this is a limitation of Python .spawn() and PyTorch)"
                    )
            else:
                rank_zero_warn(
                    "num_workers>0 and accelerator=ddp_spawn do not mix well"
                    " and may result in data loading bottlenecks."
                    " Consider setting accelerator=ddp to use num_workers>0"
                    " (this is a limitation of Python .spawn() and PyTorch)"
                )

        elif dataloader.num_workers == 0 and using_spawn:
            # checks for the attr persistent_workers available in pytorch >= 1.7
            if hasattr(dataloader, "persistent_workers"):
                if not dataloader.persistent_workers:
                    rank_zero_warn(
                        "accelerator=ddp_spawn and num_workers=0 may result in data loading bottlenecks."
                        " Consider setting num_workers>0 and persistent_workers=True"
                    )
            else:
                rank_zero_warn(
                    "accelerator=ddp_spawn and num_workers=0 may result in data loading bottlenecks."
                    " Consider setting accelerator=ddp and set num_workers>0"
                )

        elif dataloader.num_workers <= 2 < num_cpus and not using_spawn:
            rank_zero_warn(
                f"The dataloader, {name}, does not have many workers which may be a bottleneck."
                " Consider increasing the value of the `num_workers` argument`"
                f" (try {num_cpus} which is the number of cpus on this machine)"
                " in the `DataLoader` init to improve performance."
            )

    def auto_add_worker_init_fn(self, dataloader: DataLoader) -> None:
        if int(os.environ.get("PL_SEED_WORKERS", 0)) and dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = partial(pl_worker_init_function, rank=self.global_rank)

    def _requires_distributed_sampler(self, dataloader) -> bool:
        return (
            self.accelerator_connector.replace_sampler_ddp
            and self.accelerator_connector.is_distributed
            and not isinstance(dataloader.sampler, DistributedSampler)
            and not has_iterable_dataset(dataloader)
        )

    def prepare_dataloader(self, dataloader: Any, shuffle: bool, mode: Optional[RunningStage] = None) -> Any:
        """
        This function handles to following functionalities:
            - Injecting a `DistributedDataSampler` within the `DataLoader if accelertor is is_distributed
            - Wrapping the datasets and samplers into Fault Tolerant Components   
        """
        if isinstance(dataloader, CombinedLoader):
            # apply `prepare_dataloader` on all the collection of loaders
            dataloader.loaders = apply_to_collection(
                dataloader.loaders, DataLoader, self.prepare_dataloader, shuffle, mode=mode
            )
            return dataloader

        # don't do anything if it's not a dataloader
        if not isinstance(dataloader, DataLoader):
            return dataloader

        if self._requires_distributed_sampler(dataloader):
            if not isinstance(dataloader.sampler, (SequentialSampler, RandomSampler)):
                raise MisconfigurationException(
                    "You seem to have configured a sampler in your DataLoader. This will be replaced "
                    " by `DistributedSampler` since `replace_sampler_ddp` is True and you are using"
                    " distributed training. Either remove the sampler from your DataLoader or set"
                    " `replace_sampler_ddp`=False if you want to use your custom sampler."
                )
            sampler = _get_distributed_sampler(
                dataloader, shuffle, mode=mode, overfit_batches=self.overfit_batches, **self.distributed_sampler_kwargs)
        else:
            sampler = dataloader.sampler
        
        # the DataLoader should be re-created only if we need to inject 
        # the fault tolerant components or the distributed sampler.
        if _fault_tolerant_training() or isinstance(sampler, DistributedSampler):
            dataloader = self._prepare_dataloader(dataloader, sampler, mode=mode)
        
        return dataloader

    @staticmethod
    def _prepare_dataloader(dataloader: DataLoader, sampler, mode: Optional[RunningStage] = None) -> DataLoader:
        dl_kwargs = _get_dataloader_init_kwargs(dataloader, sampler, mode=mode)
        dl_cls = type(dataloader)
        dataloader = dl_cls(**dl_kwargs)
        return dataloader

    def reset_train_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the train dataloader and initialises required variables (number of batches, when to validate,
        etc.).

        Args:
            model: The `LightningModule` if calling this outside of the trainer scope.
        """
        self.train_dataloader = self.request_dataloader(RunningStage.TRAINING, model=model)

        if self.overfit_batches > 0:
            if hasattr(self.train_dataloader, "sampler") and isinstance(self.train_dataloader.sampler, RandomSampler):
                rank_zero_warn(
                    "You requested to overfit but enabled training dataloader shuffling."
                    " We are turning off the training dataloader shuffling for you."
                )
                self.train_dataloader = self._prepare_dataloader(
                    self.train_dataloader, SequentialSampler(self.train_dataloader.dataset), mode=RunningStage.TRAINING
                )

        # debugging
        self.dev_debugger.track_load_dataloader_call("train_dataloader", dataloaders=[self.train_dataloader])

        # automatically add samplers
        self.train_dataloader = apply_to_collection(
            self.train_dataloader, DataLoader, self.prepare_dataloader, shuffle=True, mode=RunningStage.TRAINING
        )

        # check the workers recursively
        apply_to_collection(self.train_dataloader, DataLoader, self._worker_check, "train_dataloader")

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(self.train_dataloader, DataLoader, self.auto_add_worker_init_fn)

        # add collate_fn to collect metadata for fault tolerant training
        if _fault_tolerant_training():
            apply_to_collection(self.train_dataloader, DataLoader, self._add_sampler_metadata_collate)

        # wrap the sequence of train loaders to a CombinedLoader object for computing the num_training_batches
        self.train_dataloader = CombinedLoader(self.train_dataloader, self.data_connector.multiple_trainloader_mode)

        self.num_training_batches = len(self.train_dataloader) if has_len(self.train_dataloader) else float("inf")

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
            if not has_len(self.train_dataloader):
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
        loader_name = f"{mode.dataloader_prefix}_dataloader"
        dataloaders = self.request_dataloader(mode, model=model)

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        # when overfitting, use the training loader as val and test
        # duplicate it the numb of times needed to match the train loaders
        if self.overfit_batches > 0:
            train_dataloader = self.request_dataloader(RunningStage.TRAINING, model=model)
            dataloaders = [deepcopy(train_dataloader) for _ in range(len(dataloaders))]

        self.dev_debugger.track_load_dataloader_call(loader_name, dataloaders=dataloaders)

        for loader_i in range(len(dataloaders)):
            loader = dataloaders[loader_i]

            if hasattr(loader, "sampler") and isinstance(loader.sampler, RandomSampler):

                # when overfitting, the dataloader should not have sampler
                if self.overfit_batches > 0 and mode.evaluating:
                    rank_zero_warn(
                        "You requested to overfit but enabled val/test dataloader shuffling."
                        " We are turning it off for you."
                    )
                    dataloaders[loader_i] = self._prepare_dataloader(loader, SequentialSampler(loader.dataset), mode=mode)
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
        apply_to_collection(dataloaders, dtype=DataLoader, function=self.auto_add_worker_init_fn)

        loader_num_batches = []

        # determine number of batches
        # datasets could be none, 1 or 2+
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                num_batches = len(dataloader) if has_len(dataloader) else float("inf")
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
        pl_module = self.lightning_module or model
        has_loader = is_overridden("val_dataloader", pl_module)
        has_step = is_overridden("validation_step", pl_module)
        if has_loader and has_step:
            self.num_val_batches, self.val_dataloaders = self._reset_eval_dataloader(
                RunningStage.VALIDATING, model=pl_module
            )

    def reset_test_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the test dataloader and determines the number of batches.

        Args:
            model: The `LightningModule` if called outside of the trainer scope.
        """
        pl_module = self.lightning_module or model
        has_loader = is_overridden("test_dataloader", pl_module)
        has_step = is_overridden("test_step", pl_module)
        if has_loader and has_step:
            self.num_test_batches, self.test_dataloaders = self._reset_eval_dataloader(
                RunningStage.TESTING, model=pl_module
            )

    def reset_predict_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the predict dataloader and determines the number of batches.

        Args:
            model: The `LightningModule` if called outside of the trainer scope.
        """
        pl_module = self.lightning_module or model
        has_loader = is_overridden("predict_dataloader", pl_module)
        if has_loader:
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
        """Handles downloading data in the GPU or TPU case.

        Returns:
            The dataloader
        """
        hook = f"{stage.dataloader_prefix}_dataloader"
        self.call_hook("on_" + hook, pl_module=model)
        dataloader = self.call_hook(hook, pl_module=model)
        if isinstance(dataloader, tuple):
            dataloader = list(dataloader)
        self.accelerator.barrier("get_dataloaders")
        return dataloader

    @staticmethod
    def _add_sampler_metadata_collate(dataloader: DataLoader) -> None:
        """Wrap default collate function to retrive ``FastForwardSampler`` state dict when fault tolerant is
        enabled."""
        dataloader.collate_fn = partial(
            _capture_metadata_collate, dataset=dataloader.dataset, default_collate=dataloader.collate_fn
        )
