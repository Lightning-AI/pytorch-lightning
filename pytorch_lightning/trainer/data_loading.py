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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper, UnrepeatedDistributedSampler
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.auto_restart import _sampler_metadata_collate
from pytorch_lightning.utilities.data import has_iterable_dataset, has_len
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_enabled
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
                f" in the `DataLoader` init to improve performance."
            )

    def auto_add_worker_init_fn(self, dataloader: DataLoader) -> None:
        if int(os.environ.get("PL_SEED_WORKERS", 0)) and dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = partial(pl_worker_init_function, rank=self.global_rank)

    def auto_add_sampler(self, dataloader: Any, shuffle: bool, mode: Optional[RunningStage] = None) -> Any:
        # don't do anything if it's not a dataloader
        is_dataloader = isinstance(dataloader, DataLoader)
        # don't manipulate iterable datasets
        is_iterable_ds = has_iterable_dataset(dataloader)

        if isinstance(dataloader, CombinedLoader):
            dataloader.loaders = apply_to_collection(dataloader.loaders, DataLoader, self.auto_add_sampler, shuffle)
            return dataloader

        if not is_dataloader or is_iterable_ds:
            return dataloader

        need_dist_sampler = self.accelerator_connector.is_distributed and not isinstance(
            dataloader.sampler, DistributedSampler
        )
        if self.accelerator_connector.replace_sampler_ddp and need_dist_sampler:
            if not isinstance(dataloader.sampler, (SequentialSampler, RandomSampler)):
                raise MisconfigurationException(
                    "You seem to have configured a sampler in your DataLoader. This will be replaced "
                    " by `DistributedSampler` since `replace_sampler_ddp` is True and you are using"
                    " distributed training. Either remove the sampler from your DataLoader or set"
                    " `replace_sampler_ddp`=False if you want to use your custom sampler."
                )

            # replace with distributed sampler
            sampler = self._get_distributed_sampler(dataloader, shuffle, mode=mode)
            dataloader = self.replace_sampler(dataloader, sampler, mode=mode)

        return dataloader

    @staticmethod
    def _resolve_batch_sampler(dataloader, sampler, mode: Optional[RunningStage] = None) -> Dict[str, Any]:
        batch_sampler = getattr(dataloader, "batch_sampler")
        is_predicting = mode == RunningStage.PREDICTING
        # checking the batch sampler type is different than PyTorch default.
        if (batch_sampler is not None and type(batch_sampler) is not BatchSampler) or is_predicting:
            batch_sampler = type(batch_sampler)(
                sampler,
                batch_size=batch_sampler.batch_size,
                drop_last=(False if is_predicting else batch_sampler.drop_last),
            )
            if is_predicting:
                batch_sampler = IndexBatchSamplerWrapper(batch_sampler)
            return {
                "sampler": None,
                "shuffle": False,
                "batch_sampler": batch_sampler,
                "batch_size": 1,
                "drop_last": False,
            }
        return {"sampler": sampler, "shuffle": False, "batch_sampler": None}

    def replace_sampler(self, dataloader: DataLoader, sampler, mode: Optional[RunningStage] = None) -> DataLoader:
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
        dl_kwargs.update(self._resolve_batch_sampler(dataloader, sampler, mode=mode))

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
        dataloader = dl_cls(**dl_kwargs)
        return dataloader

    def _get_distributed_sampler(
        self, dataloader: DataLoader, shuffle: bool, mode: Optional[RunningStage] = None
    ) -> DistributedSampler:
        kwargs = self.distributed_sampler_kwargs
        kwargs["shuffle"] = shuffle and not self.overfit_batches
        kwargs.setdefault("seed", int(os.getenv("PL_GLOBAL_SEED", 0)))
        cls = UnrepeatedDistributedSampler if mode == RunningStage.PREDICTING else DistributedSampler
        sampler = cls(dataloader.dataset, **kwargs)
        return sampler

    def reset_train_dataloader(self, model: "pl.LightningModule") -> None:
        """Resets the train dataloader and initialises required variables
        (number of batches, when to validate, etc.).

        Args:
            model: The current `LightningModule`
        """
        self.train_dataloader = self.request_dataloader(model, "train")

        if self.overfit_batches > 0:
            if hasattr(self.train_dataloader, "sampler") and isinstance(self.train_dataloader.sampler, RandomSampler):
                rank_zero_warn(
                    "You requested to overfit but enabled training dataloader shuffling."
                    " We are turning off the training dataloader shuffling for you."
                )
                self.train_dataloader = self.replace_sampler(
                    self.train_dataloader, SequentialSampler(self.train_dataloader.dataset)
                )

        # debugging
        self.dev_debugger.track_load_dataloader_call("train_dataloader", dataloaders=[self.train_dataloader])

        # automatically add samplers
        self.train_dataloader = apply_to_collection(
            self.train_dataloader, DataLoader, self.auto_add_sampler, shuffle=True
        )

        # check the workers recursively
        apply_to_collection(self.train_dataloader, DataLoader, self._worker_check, "train dataloader")

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(self.train_dataloader, DataLoader, self.auto_add_worker_init_fn)

        # add collate_fn to collect metadata for fault tolerant training
        if _fault_tolerant_enabled():
            apply_to_collection(self.train_dataloader, DataLoader, self._add_sampler_metadata_collate)

        # wrap the sequence of train loaders to a CombinedLoader object for computing the num_training_batches
        self.train_dataloader = CombinedLoader(self.train_dataloader, self.data_connector.multiple_trainloader_mode)

        # allow accelerator to modify dataloader
        self.train_dataloader = self.accelerator.on_reset_train_dataloader(self.train_dataloader)

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
                f" you want to see logs for the training epoch."
            )

    def _reset_eval_dataloader(
        self, model: "pl.LightningModule", mode: str
    ) -> Tuple[List[Union[int, float]], List[DataLoader]]:
        """Generic method to reset a dataloader for evaluation.

        Args:
            model: The current `LightningModule`
            mode: Either `'val'`, `'test'` or `'predict'`

        Returns:
            Tuple (num_batches, dataloaders)
        """
        # always get the loaders first so we can count how many there are
        loader_name = f"{mode}_dataloader"
        dataloaders = self.request_dataloader(model, mode)

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        # when overfitting use the training loader as val and test
        # duplicate it the numb of times needed to match the train loaders
        if self.overfit_batches > 0:
            num_loaders = len(dataloaders)
            train_dataloader = self.request_dataloader(model, "train")
            dataloaders = [deepcopy(train_dataloader) for _ in range(num_loaders)]

        self.dev_debugger.track_load_dataloader_call(loader_name, dataloaders=dataloaders)

        for loader_i in range(len(dataloaders)):
            loader = dataloaders[loader_i]

            # shuffling in val and test set is bad practice
            modes = ("val", "test", "predict")
            if mode in modes and hasattr(loader, "sampler") and isinstance(loader.sampler, RandomSampler):

                # when overfitting, the dataloader should not have sampler
                if self.overfit_batches > 0 and mode != "predict":
                    rank_zero_warn(
                        "You requested to overfit but enabled val/test dataloader shuffling."
                        " We are turning it off for you."
                    )
                    dataloaders[loader_i] = self.replace_sampler(loader, SequentialSampler(loader.dataset))

                else:
                    rank_zero_warn(
                        f"Your {mode}_dataloader has `shuffle=True`, it is best practice to turn"
                        " this off for val/test/predict dataloaders."
                    )

        if any(dl is None for dl in dataloaders):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")

        # add samplers
        dataloaders = [
            self.auto_add_sampler(dl, shuffle=False, mode=self.state.stage) for dl in dataloaders if dl is not None
        ]

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(dataloaders, dtype=DataLoader, function=self.auto_add_worker_init_fn)

        # allow accelerator to modify dataloader
        hook_name = f"on_reset_{mode}_dataloader"
        dataloaders = getattr(self.accelerator, hook_name)(dataloaders)

        loader_num_batches = []

        # determine number of batches
        # datasets could be none, 1 or 2+
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                num_batches = len(dataloader) if has_len(dataloader) else float("inf")
                self._worker_check(dataloader, f"{mode} dataloader {i}")

                # percent or num_steps
                limit_eval_batches = getattr(self, f"limit_{mode}_batches")

                # limit num batches either as a percent or num steps
                if isinstance(limit_eval_batches, int) or limit_eval_batches == 0.0:
                    num_batches = min(num_batches, int(limit_eval_batches))
                elif num_batches != float("inf"):
                    num_batches = int(num_batches * limit_eval_batches)
                elif limit_eval_batches != 1.0:
                    raise MisconfigurationException(
                        "When using an IterableDataset for `limit_{mode}_batches`,"
                        f" `Trainer(limit_{mode}_batches)` must be `0.0`, `1.0` or an int. An int k specifies"
                        f" `num_{mode}_batches` to use."
                    )

                if num_batches == 0 and limit_eval_batches > 0.0 and isinstance(limit_eval_batches, float):
                    min_pct = 1.0 / len(dataloader)
                    raise MisconfigurationException(
                        f"you requested to check {limit_eval_batches} of the {mode} dataloader but"
                        f" {limit_eval_batches}*{num_batches} < 1. Please increase the limit_{mode}_batches."
                        f" Try at least limit_{mode}_batches={min_pct}"
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders

    def reset_val_dataloader(self, model: "pl.LightningModule") -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = is_overridden("val_dataloader", model)
        has_step = is_overridden("validation_step", model)
        if has_loader and has_step:
            self.num_val_batches, self.val_dataloaders = self._reset_eval_dataloader(model, "val")

    def reset_test_dataloader(self, model) -> None:
        """Resets the test dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = is_overridden("test_dataloader", model)
        has_step = is_overridden("test_step", model)
        if has_loader and has_step:
            self.num_test_batches, self.test_dataloaders = self._reset_eval_dataloader(model, "test")

    def reset_predict_dataloader(self, model) -> None:
        """Resets the predict dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = is_overridden("predict_dataloader", model)
        if has_loader:
            self.num_predict_batches, self.predict_dataloaders = self._reset_eval_dataloader(model, "predict")

    def reset_train_val_dataloaders(self, model) -> None:
        """
        Resets train and val dataloaders if none are attached to the trainer.

        The val dataloader must be initialized before training loop starts, as the training loop
        inspects the val dataloader to determine whether to run the evaluation loop.
        """
        if self.train_dataloader is None:
            self.reset_train_dataloader(model)

        if self.val_dataloaders is None:
            self.reset_val_dataloader(model)

    def request_dataloader(self, model: "pl.LightningModule", stage: str) -> Union[DataLoader, List[DataLoader]]:
        """Handles downloading data in the GPU or TPU case.

        Returns:
            The dataloader
        """
        self.call_hook(f"on_{stage}_dataloader")
        dataloader = getattr(model, f"{stage}_dataloader")()
        if isinstance(dataloader, tuple):
            dataloader = list(dataloader)
        self.accelerator.barrier("get_dataloaders")
        return dataloader

    @staticmethod
    def _add_sampler_metadata_collate(dataloader: DataLoader) -> None:
        """
        Wrap default collate function to retrive ``FastForwardSampler`` state dict when fault tolerant is enabled.
        """
        dataloader.collate_fn = partial(
            _sampler_metadata_collate, dataset=dataloader.dataset, default_collate=dataloader.collate_fn
        )
