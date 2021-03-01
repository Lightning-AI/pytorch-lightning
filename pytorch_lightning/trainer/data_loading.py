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
import platform
from abc import ABC
from copy import deepcopy
from typing import Callable, Iterable, List, Optional, Tuple, Union

from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.data import has_iterable_dataset, has_len
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden


class TrainerDataLoadingMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    global_rank: int
    shown_warnings:...
    val_check_interval: float
    tpu_local_core_rank: int
    train_dataloader: DataLoader
    num_training_batches: Union[int, float]
    val_check_batch:...
    val_dataloaders: List[DataLoader]
    num_val_batches: List[Union[int, float]]
    test_dataloaders: List[DataLoader]
    num_test_batches: List[Union[int, float]]
    limit_train_batches: Union[int, float]
    limit_val_batches: Union[int, float]
    limit_test_batches: Union[int, float]
    replace_sampler_ddp: bool
    accelerator: Accelerator
    num_nodes: int
    num_processes: int
    distributed_backend: Optional[str]
    dev_debugger: InternalDebugger

    def _worker_check(self, dataloader: DataLoader, name: str) -> None:
        on_windows = platform.system() == 'Windows'

        # ddp_spawn + num_workers > 0 don't mix! tell the user
        is_dataloader = isinstance(dataloader, DataLoader)
        using_spawn = self.accelerator_connector.distributed_backend == "ddp_spawn"
        if is_dataloader and not on_windows:
            if dataloader.num_workers > 0 and using_spawn:
                rank_zero_warn(
                    'Dataloader(num_workers>0) and ddp_spawn do not mix well!'
                    ' Your performance might suffer dramatically.'
                    ' Please consider setting accelerator=ddp to use num_workers > 0'
                    ' (this is a bottleneck of Python .spawn() and PyTorch'
                )

            elif dataloader.num_workers == 0 and using_spawn:
                rank_zero_warn(
                    'You are using `accelerator=ddp_spawn` with num_workers=0.'
                    ' For much faster performance, switch to `accelerator=ddp` and set `num_workers>0`'
                )

            elif dataloader.num_workers <= 2 and multiprocessing.cpu_count() > 2 and not using_spawn:
                num_cpus = multiprocessing.cpu_count()
                rank_zero_warn(
                    f'The dataloader, {name}, does not have many workers which may be a bottleneck.'
                    ' Consider increasing the value of the `num_workers` argument`'
                    f' (try {num_cpus} which is the number of cpus on this machine)'
                    f' in the `DataLoader` init to improve performance.'
                )

    def auto_add_sampler(self, dataloader: DataLoader, shuffle: bool) -> DataLoader:

        # don't do anything if it's not a dataloader
        is_dataloader = isinstance(dataloader, DataLoader)
        # don't manipulate iterable datasets
        is_iterable_ds = has_iterable_dataset(dataloader)

        if not is_dataloader or is_iterable_ds:
            return dataloader

        need_dist_sampler = self.accelerator_connector.is_distributed and not isinstance(
            dataloader.sampler, DistributedSampler
        )
        if self.accelerator_connector.replace_sampler_ddp and need_dist_sampler:
            if not isinstance(dataloader.sampler, (SequentialSampler, RandomSampler)):
                raise MisconfigurationException(
                    'You seem to have configured a sampler in your DataLoader. This will be replaced '
                    ' by `DistributedSampler` since `replace_sampler_ddp` is True and you are using'
                    ' distributed training. Either remove the sampler from your DataLoader or set'
                    ' `replace_sampler_ddp`=False if you want to use your custom sampler.'
                )

            # replace with distributed sampler
            sampler = self._get_distributed_sampler(dataloader, shuffle)
            dataloader = self.replace_sampler(dataloader, sampler)

        return dataloader

    @staticmethod
    def _resolve_batch_sampler(dl_args, dataloader, sampler):
        batch_sampler = getattr(dataloader, "batch_sampler")
        if batch_sampler is not None and type(batch_sampler) is not BatchSampler:
            batch_sampler = type(batch_sampler)(
                sampler,
                batch_size=batch_sampler.batch_size,
                drop_last=batch_sampler.drop_last,
            )
            dl_args['batch_sampler'] = batch_sampler
            dl_args['batch_size'] = 1
            dl_args['shuffle'] = False
            dl_args['sampler'] = None
            dl_args['drop_last'] = False
        else:
            dl_args['sampler'] = sampler
            dl_args['shuffle'] = False
            dl_args['batch_sampler'] = None

        return dl_args

    def replace_sampler(self, dataloader, sampler):
        skip_keys = ('sampler', 'batch_sampler', 'dataset_kind')
        skip_signature_keys = ('args', 'kwargs', 'self')

        attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}

        params = set(inspect.signature(dataloader.__init__).parameters)
        contains_dataset = True

        if type(dataloader) is not DataLoader:
            contains_dataset = "dataset" in params
            params.update(inspect.signature(DataLoader.__init__).parameters)

        dl_args = {name: attrs[name] for name in params if name in attrs and name not in skip_keys}

        dl_args = self._resolve_batch_sampler(dl_args, dataloader, sampler)

        multiprocessing_context = dataloader.multiprocessing_context
        dl_args['multiprocessing_context'] = multiprocessing_context

        missing_kwargs = params.difference(skip_signature_keys).difference(dl_args)
        if missing_kwargs:
            """
            Example:
            class CustomDataLoader(DataLoader):
                def __init__(self, num_features, dataset, *args, **kwargs):
                    self.num_features = num_features
                    super().__init__(dataset, *args, **kwargs)
            """
            dataloader_cls_name = dataloader.__class__.__name__
            raise MisconfigurationException(
                f"Trying to inject DistributedSampler within {dataloader_cls_name} class."
                "This would fail as your DataLoader doesn't expose all its __init__ parameters as attributes. "
                f"Missing attributes are {missing_kwargs}. "
                f"HINT: If you wrote the {dataloader_cls_name} class, add the `__init__` arguments as attributes or ",
                "manually add DistributedSampler as "
                f"{dataloader_cls_name}(dataset, ..., sampler=DistributedSampler(dataset, ...)).",
            )

        if not contains_dataset:
            dl_args.pop('dataset')

        dataloader = type(dataloader)(**dl_args)
        dataloader.multiprocessing_context = multiprocessing_context
        return dataloader

    def _get_distributed_sampler(self, dataloader, shuffle):
        kwargs = self.distributed_sampler_kwargs
        kwargs['shuffle'] = shuffle and not self.overfit_batches
        sampler = DistributedSampler(dataloader.dataset, **kwargs)
        return sampler

    def reset_train_dataloader(self, model: LightningModule) -> None:
        """Resets the train dataloader and initialises required variables
        (number of batches, when to validate, etc.).

        Args:
            model: The current `LightningModule`
        """
        self.train_dataloader = self.request_dataloader(model.train_dataloader)

        if self.overfit_batches > 0:
            if hasattr(self.train_dataloader, 'sampler') and isinstance(self.train_dataloader.sampler, RandomSampler):
                rank_zero_warn(
                    'You requested to overfit but enabled training dataloader shuffling.'
                    ' We are turning it off for you.'
                )
                self.train_dataloader = self.replace_sampler(
                    self.train_dataloader, SequentialSampler(self.train_dataloader.dataset)
                )

        # debugging
        self.dev_debugger.track_load_dataloader_call('train_dataloader', dataloaders=[self.train_dataloader])

        # automatically add samplers
        self.train_dataloader = apply_to_collection(
            self.train_dataloader, DataLoader, self.auto_add_sampler, shuffle=True
        )

        # check the workers recursively
        apply_to_collection(self.train_dataloader, DataLoader, self._worker_check, 'train dataloader')

        # wrap the sequence of train loaders to a CombinedLoader object for computing the num_training_batches
        self.train_dataloader = CombinedLoader(self.train_dataloader, self._multiple_trainloader_mode)

        self.num_training_batches = len(self.train_dataloader) if has_len(self.train_dataloader) else float('inf')

        if isinstance(self.limit_train_batches, int) or self.limit_train_batches == 0.0:
            self.num_training_batches = min(self.num_training_batches, int(self.limit_train_batches))
        elif self.num_training_batches != float('inf'):
            self.num_training_batches = int(self.num_training_batches * self.limit_train_batches)
        elif self.limit_train_batches != 1.0:
            raise MisconfigurationException(
                'When using an IterableDataset for `limit_train_batches`,'
                ' `Trainer(limit_train_batches)` must be `0.0`, `1.0` or an int. An int k specifies'
                ' `num_training_batches` to use.'
            )

        # determine when to check validation
        # if int passed in, val checks that often
        # otherwise, it checks in [0, 1.0] % range of a training epoch
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches:
                raise ValueError(
                    f'`val_check_interval` ({self.val_check_interval}) must be less than or equal '
                    f'to the number of the training batches ({self.num_training_batches}). '
                    'If you want to disable validation set `limit_val_batches` to 0.0 instead.'
                )
        else:
            if not has_len(self.train_dataloader):
                if self.val_check_interval == 1.0:
                    self.val_check_batch = float('inf')
                else:
                    raise MisconfigurationException(
                        'When using an IterableDataset for `train_dataloader`,'
                        ' `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies'
                        ' checking validation every k training batches.'
                    )
            else:
                self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
                self.val_check_batch = max(1, self.val_check_batch)

    def _reset_eval_dataloader(
        self,
        model: LightningModule,
        mode: str,
    ) -> Tuple[List[Union[int, float]], List[DataLoader]]:
        """Generic method to reset a dataloader for evaluation.

        Args:
            model: The current `LightningModule`
            mode: Either `'val'` or `'test'`

        Returns:
            Tuple (num_batches, dataloaders)
        """
        # always get the loaders first so we can count how many there are
        loader_name = f'{mode}_dataloader'
        dataloaders = self.request_dataloader(getattr(model, loader_name))

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        # when overfitting use the training loader as val and test
        # duplicate it the numb of times needed to match the train loaders
        if self.overfit_batches > 0:
            num_loaders = len(dataloaders)
            train_dataloader = self.request_dataloader(getattr(model, 'train_dataloader'))
            dataloaders = [deepcopy(train_dataloader) for _ in range(num_loaders)]

        self.dev_debugger.track_load_dataloader_call(loader_name, dataloaders=dataloaders)

        for loader_i in range(len(dataloaders)):
            loader = dataloaders[loader_i]

            # shuffling in val and test set is bad practice
            modes = ('val', 'test', 'predict')
            if mode in modes and hasattr(loader, 'sampler') and isinstance(loader.sampler, RandomSampler):

                # when overfitting, the dataloader should not have sampler
                if self.overfit_batches > 0:
                    rank_zero_warn(
                        'You requested to overfit but enabled test/val dataloader shuffling.'
                        ' We are turning it off for you.'
                    )
                    dataloaders[loader_i] = self.replace_sampler(loader, SequentialSampler(loader.dataset))

                else:
                    rank_zero_warn(
                        f'Your {mode}_dataloader has `shuffle=True`, it is best practice to turn'
                        ' this off for validation and test dataloaders.'
                    )

        if any([dl is None for dl in dataloaders]):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")

        # add samplers
        dataloaders = [self.auto_add_sampler(dl, shuffle=False) for dl in dataloaders if dl is not None]

        loader_num_batches = []

        # determine number of batches
        # datasets could be none, 1 or 2+
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                num_batches = len(dataloader) if has_len(dataloader) else float('inf')
                self._worker_check(dataloader, f'{mode} dataloader {i}')

                # percent or num_steps
                limit_eval_batches = getattr(self, f'limit_{mode}_batches')

                # limit num batches either as a percent or num steps
                if isinstance(limit_eval_batches, int) or limit_eval_batches == 0.0:
                    num_batches = min(num_batches, int(limit_eval_batches))
                elif num_batches != float('inf'):
                    num_batches = int(num_batches * limit_eval_batches)
                elif limit_eval_batches != 1.0:
                    raise MisconfigurationException(
                        'When using an IterableDataset for `limit_{mode}_batches`,'
                        f' `Trainer(limit_{mode}_batches)` must be `0.0`, `1.0` or an int. An int k specifies'
                        f' `num_{mode}_batches` to use.'
                    )

                if num_batches == 0 and limit_eval_batches > 0.0 and isinstance(limit_eval_batches, float):
                    min_pct = 1.0 / len(dataloader)
                    raise MisconfigurationException(
                        f'you requested to check {limit_eval_batches} of the {mode} dataloader but'
                        f' {limit_eval_batches}*{num_batches} < 1. Please increase the limit_{mode}_batches.'
                        f' Try at least limit_{mode}_batches={min_pct}'
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders

    def reset_val_dataloader(self, model: LightningModule) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = is_overridden('val_dataloader', model)
        has_step = is_overridden('validation_step', model)
        if has_loader and has_step:
            self.num_val_batches, self.val_dataloaders = self._reset_eval_dataloader(model, 'val')

    def reset_test_dataloader(self, model) -> None:
        """Resets the test dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = is_overridden('test_dataloader', model)
        has_step = is_overridden('test_step', model)
        if has_loader and has_step:
            self.num_test_batches, self.test_dataloaders =\
                self._reset_eval_dataloader(model, 'test')

    def reset_predict_dataloader(self, model) -> None:
        """Resets the predict dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = is_overridden('predict_dataloader', model)
        if has_loader:
            self.num_predict_batches, self.predict_dataloaders =\
                self._reset_eval_dataloader(model, 'predict')

    def request_dataloader(self, dataloader_fx: Callable) -> DataLoader:
        """Handles downloading data in the GPU or TPU case.

        Args:
            dataloader_fx: The bound dataloader getter

        Returns:
            The dataloader
        """
        dataloader = dataloader_fx()
        dataloader = self._flatten_dl_only(dataloader)

        self.accelerator.barrier('get_dataloaders')
        return dataloader

    def _flatten_dl_only(self, dataloaders):
        # handles user error when they return:
        # return dl1, dl2  vs  return (dl1, dl2)
        if isinstance(dataloaders, tuple):
            all_dls = [isinstance(x, Iterable) for x in dataloaders]
            all_dls = all(all_dls)
            if all_dls:
                dataloaders = list(dataloaders)

        return dataloaders
