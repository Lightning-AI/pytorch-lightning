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
import platform
from abc import ABC, abstractmethod
from distutils.version import LooseVersion
from typing import Union, List, Tuple, Callable, Optional

import torch
import torch.distributed as torch_distrib
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    from torch.utils.data import IterableDataset
    ITERABLE_DATASET_EXISTS = True
except ImportError:
    ITERABLE_DATASET_EXISTS = False

try:
    from apex import amp
except ImportError:
    amp = None

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


def _has_iterable_dataset(dataloader: DataLoader):
    return ITERABLE_DATASET_EXISTS and hasattr(dataloader, 'dataset') \
        and isinstance(dataloader.dataset, IterableDataset)


def _has_len(dataloader: DataLoader) -> bool:
    """ Checks if a given Dataloader has __len__ method implemented i.e. if
    it is a finite dataloader or infinite dataloader. """

    try:
        # try getting the length
        if len(dataloader) == 0:
            raise ValueError('`Dataloader` returned 0 length.'
                             ' Please make sure that your Dataloader at least returns 1 batch')
        has_len = True
    except TypeError:
        has_len = False
    except NotImplementedError:  # e.g. raised by torchtext if a batch_size_fn is used
        has_len = False

    if has_len and _has_iterable_dataset(dataloader) and LooseVersion(torch.__version__) >= LooseVersion("1.4.0"):
        rank_zero_warn(
            'Your `IterableDataset` has `__len__` defined.'
            ' In combination with multi-processing data loading (e.g. batch size > 1),'
            ' this can lead to unintended side effects since the samples will be duplicated.'
        )
    return has_len


class TrainerDataLoadingMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    global_rank: int
    use_ddp: bool
    use_ddp2: bool
    use_horovod: bool
    shown_warnings: ...
    val_check_interval: float
    use_tpu: bool
    tpu_local_core_rank: int
    train_dataloader: DataLoader
    num_training_batches: Union[int, float]
    val_check_batch: ...
    val_dataloaders: List[DataLoader]
    num_val_batches: List[Union[int, float]]
    test_dataloaders: List[DataLoader]
    num_test_batches: List[Union[int, float]]
    limit_train_batches: Union[int, float]
    limit_val_batches: Union[int, float]
    limit_test_batches: Union[int, float]
    replace_sampler_ddp: bool
    num_nodes: int
    num_processes: int
    distributed_backend: Optional[str]

    @abstractmethod
    def is_overridden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def _worker_check(self, dataloader: DataLoader, name: str) -> None:
        on_windows = platform.system() == 'Windows'

        # ddp_spawn + num_workers > 0 don't mix! tell the user
        is_dataloader = isinstance(dataloader, DataLoader)
        using_spawn = self.distributed_backend == 'ddp_spawn'
        if is_dataloader and not on_windows:
            if dataloader.num_workers > 0 and using_spawn:
                rank_zero_warn('Dataloader(num_workers>0) and ddp_spawn do not mix well!'
                               ' Your performance might suffer dramatically.'
                               ' Please consider setting distributed_backend=ddp to use num_workers > 0'
                               ' (this is a bottleneck of Python .spawn() and PyTorch')

            elif dataloader.num_workers == 0 and using_spawn:
                rank_zero_warn('You are using `distributed_backend=ddp_spawn` with num_workers=0.'
                               ' For much faster performance, switch to `distributed_backend=ddp`'
                               ' and set `num_workers>0`')

            elif dataloader.num_workers <= 2 and multiprocessing.cpu_count() > 2 and not using_spawn:
                num_cpus = multiprocessing.cpu_count()
                rank_zero_warn(f'The dataloader, {name}, does not have many workers which may be a bottleneck.'
                               ' Consider increasing the value of the `num_workers` argument`'
                               f' (try {num_cpus} which is the number of cpus on this machine)'
                               ' in the `DataLoader` init to improve performance.')

    def auto_add_sampler(self, dataloader: DataLoader, train: bool) -> DataLoader:

        # don't do anything if it's not a dataloader
        is_dataloader = isinstance(dataloader, DataLoader)
        # don't manipulate iterable datasets
        is_iterable_ds = _has_iterable_dataset(dataloader)

        if not is_dataloader or is_iterable_ds:
            return dataloader
        need_dist_sampler = (self.use_ddp or self.use_ddp2 or self.use_horovod or self.use_tpu)

        if self.replace_sampler_ddp and need_dist_sampler:
            if not isinstance(dataloader.sampler, (SequentialSampler, RandomSampler)):
                raise MisconfigurationException(
                    'You seem to have configured a sampler in your DataLoader. This will be replaced '
                    ' by `DistributedSampler` since `replace_sampler_ddp` is True and you are using'
                    ' distributed training. Either remove the sampler from your DataLoader or set'
                    ' `replace_sampler_ddp`=False if you want to use your custom sampler.')

            # replace with distributed sampler
            sampler = self._get_distributed_sampler(dataloader, train)
            dataloader = self.replace_sampler(dataloader, sampler)

        return dataloader

    def replace_sampler(self, dataloader, sampler):
        skip_keys = ['sampler', 'batch_sampler', 'dataset_kind']

        dl_args = {
            k: v for k, v in dataloader.__dict__.items() if not k.startswith('_') and k not in skip_keys
        }

        dl_args['sampler'] = sampler
        dataloader = type(dataloader)(**dl_args)
        return dataloader

    def _get_distributed_sampler(self, dataloader, train):
        if self.use_tpu:
            kwargs = dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        elif self.use_horovod:
            kwargs = dict(num_replicas=hvd.size(), rank=hvd.rank())
        else:
            world_size = {
                'ddp': self.num_nodes * self.num_processes,
                'ddp_spawn': self.num_nodes * self.num_processes,
                'ddp2': self.num_nodes,
                'ddp_cpu': self.num_processes * self.num_nodes
            }
            assert self.distributed_backend is not None
            kwargs = dict(num_replicas=world_size[self.distributed_backend], rank=self.global_rank)

        kwargs['shuffle'] = train
        sampler = DistributedSampler(dataloader.dataset, **kwargs)
        return sampler

    def reset_train_dataloader(self, model: LightningModule) -> None:
        """Resets the train dataloader and initialises required variables
        (number of batches, when to validate, etc.).

        Args:
            model: The current `LightningModule`
        """
        self.train_dataloader = self.request_dataloader(model.train_dataloader)

        self.num_training_batches = 0

        # automatically add samplers
        self.train_dataloader = self.auto_add_sampler(self.train_dataloader, train=True)

        self.num_training_batches = len(self.train_dataloader) if _has_len(self.train_dataloader) else float('inf')
        self._worker_check(self.train_dataloader, 'train dataloader')

        if isinstance(self.limit_train_batches, int) or self.limit_train_batches == 0.0:
            self.num_training_batches = min(self.num_training_batches, int(self.limit_train_batches))
        elif self.num_training_batches != float('inf'):
            self.num_training_batches = int(self.num_training_batches * self.limit_train_batches)
        elif self.limit_train_batches != 1.0:
            raise MisconfigurationException(
                'When using an IterableDataset for `limit_train_batches`,'
                ' `Trainer(limit_train_batches)` must be `0.0`, `1.0` or an int. An int k specifies'
                ' `num_training_batches` to use.')

        # determine when to check validation
        # if int passed in, val checks that often
        # otherwise, it checks in [0, 1.0] % range of a training epoch
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches:
                raise ValueError(
                    f'`val_check_interval` ({self.val_check_interval}) must be less than or equal '
                    f'to the number of the training batches ({self.num_training_batches}). '
                    'If you want to disable validation set `limit_val_batches` to 0.0 instead.')
        else:
            if not _has_len(self.train_dataloader):
                if self.val_check_interval == 1.0:
                    self.val_check_batch = float('inf')
                else:
                    raise MisconfigurationException(
                        'When using an IterableDataset for `train_dataloader`,'
                        ' `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies'
                        ' checking validation every k training batches.')
            else:
                self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
                self.val_check_batch = max(1, self.val_check_batch)

    def _reset_eval_dataloader(
            self,
            model: LightningModule,
            mode: str
    ) -> Tuple[List[Union[int, float]], List[DataLoader]]:
        """Generic method to reset a dataloader for evaluation.

        Args:
            model: The current `LightningModule`
            mode: Either `'val'` or `'test'`

        Returns:
            Tuple (num_batches, dataloaders)
        """
        # use the training loader as val and test when overfitting
        if self.overfit_batches > 0:
            dataloaders = self.request_dataloader(getattr(model, 'train_dataloader'))
        else:
            dataloaders = self.request_dataloader(getattr(model, f'{mode}_dataloader'))

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        for loader_i in range(len(dataloaders)):
            loader = dataloaders[loader_i]

            # shuffling in val and test set is bad practice
            if mode in ('val', 'test') and hasattr(loader, 'sampler') and isinstance(loader.sampler, RandomSampler):

                # when overfitting, the dataloader should not have sampler
                if self.overfit_batches > 0:
                    rank_zero_warn('You requested to overfit but enabled training dataloader shuffling.'
                                   ' We are turning it off for you.')
                    dataloaders[loader_i] = self.replace_sampler(loader, SequentialSampler(loader.dataset))

                else:
                    rank_zero_warn(f'Your {mode}_dataloader has `shuffle=True`, it is best practice to turn'
                                   ' this off for validation and test dataloaders.')

        if any([dl is None for dl in dataloaders]):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")

        # add samplers
        dataloaders = [self.auto_add_sampler(dl, train=False) for dl in dataloaders if dl is not None]

        loader_num_batches = []

        # determine number of batches
        # datasets could be none, 1 or 2+
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                num_batches = len(dataloader) if _has_len(dataloader) else float('inf')
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
                        f' `num_{mode}_batches` to use.')

                if num_batches == 0 and limit_eval_batches > 0.0 and isinstance(limit_eval_batches, float):
                    min_pct = 1.0 / len(dataloader)
                    raise MisconfigurationException(
                        f'you requested to check {limit_eval_batches} of the {mode} dataloader but'
                        f' {limit_eval_batches}*{num_batches} = 0. Please increase the limit_{mode}_batches.'
                        f' Try at least limit_{mode}_batches={min_pct}'
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders

    def reset_val_dataloader(self, model: LightningModule) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = self.is_overridden('val_dataloader', model)
        has_step = self.is_overridden('validation_step', model)
        if has_loader and has_step:
            self.num_val_batches, self.val_dataloaders = self._reset_eval_dataloader(model, 'val')

    def reset_test_dataloader(self, model) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = self.is_overridden('test_dataloader', model)
        has_step = self.is_overridden('test_step', model)
        if has_loader and has_step:
            self.num_test_batches, self.test_dataloaders =\
                self._reset_eval_dataloader(model, 'test')

    def request_dataloader(self, dataloader_fx: Callable) -> DataLoader:
        """Handles downloading data in the GPU or TPU case.

        Args:
            dataloader_fx: The bound dataloader getter

        Returns:
            The dataloader
        """
        dataloader = dataloader_fx()

        # get the function we'll use to get data
        if self.use_ddp or self.use_ddp2:
            # all processes wait until data download has happened
            torch_distrib.barrier()

        # data download/load on TPU
        elif self.use_tpu and XLA_AVAILABLE:
            # all processes wait until data download has happened
            torch_xla.core.xla_model.rendezvous('pl.TrainerDataLoadingMixin.get_dataloaders')

        elif self.use_horovod:
            # all processes wait until data download has happened
            hvd.join()

        return dataloader

    def determine_data_use_amount(self, overfit_batches: float) -> None:
        """Use less data for debugging purposes"""
        if overfit_batches > 0:
            self.limit_train_batches = overfit_batches
            self.limit_val_batches = overfit_batches
            self.limit_test_batches = overfit_batches
