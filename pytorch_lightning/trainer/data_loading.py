import platform
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Callable

import torch.distributed as torch_distrib
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


def _has_len(dataloader: DataLoader) -> bool:
    """ Checks if a given Dataloader has __len__ method implemented i.e. if
    it is a finite dataloader or infinite dataloader """
    try:
        # try getting the length
        if len(dataloader) == 0:
            raise ValueError('Dataloader returned 0 length. Please make sure'
                             ' that your Dataloader atleast returns 1 batch')
        return True
    except TypeError:
        return False


class TrainerDataLoadingMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    proc_rank: int
    use_ddp: bool
    use_ddp2: bool
    shown_warnings: ...
    val_check_interval: float
    use_tpu: bool
    tpu_local_core_rank: int
    train_dataloader: DataLoader
    num_training_batches: Union[int, float]
    val_check_batch: ...
    val_dataloaders: List[DataLoader]
    num_val_batches: Union[int, float]
    test_dataloaders: List[DataLoader]
    num_test_batches: Union[int, float]
    train_percent_check: float
    val_percent_check: float
    test_percent_check: float

    @abstractmethod
    def is_overriden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def _percent_range_check(self, name: str) -> None:
        value = getattr(self, name)
        msg = f'`{name}` must lie in the range [0.0, 1.0], but got {value:.3f}.'
        if name == 'val_check_interval':
            msg += ' If you want to disable validation set `val_percent_check` to 0.0 instead.'

        if not 0. <= value <= 1.:
            raise ValueError(msg)

    def _worker_check(self, dataloader: DataLoader, name: str) -> None:
        on_windows = platform.system() == 'Windows'

        if isinstance(dataloader, DataLoader) and dataloader.num_workers <= 2 and not on_windows:
            rank_zero_warn(f'The dataloader, {name}, does not have many workers which may be a bottleneck.'
                           ' Consider increasing the value of the `num_workers` argument`'
                           ' in the `DataLoader` init to improve performance.')

    def auto_add_sampler(self, dataloader: DataLoader, train: bool) -> DataLoader:

        # don't do anything if it's not a dataloader
        if not isinstance(dataloader, DataLoader):
            return dataloader

        need_dist_sampler = self.use_ddp or self.use_ddp2 or self.use_tpu

        if need_dist_sampler:

            skip_keys = ['sampler', 'batch_sampler', 'dataset_kind']

            dl_args = {
                k: v for k, v in dataloader.__dict__.items() if not k.startswith('_') and k not in skip_keys
            }

            if self.use_tpu:
                sampler = DistributedSampler(
                    dataloader.dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal()
                )
            else:
                sampler = DistributedSampler(dataloader.dataset)

            dl_args['sampler'] = sampler
            dataloader = type(dataloader)(**dl_args)

        return dataloader

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

        self._worker_check(self.train_dataloader, 'train dataloader')
        self._percent_range_check('train_percent_check')

        if not _has_len(self.train_dataloader):
            self.num_training_batches = float('inf')
        else:
            # try getting the length
            self.num_training_batches = len(self.train_dataloader)
            self.num_training_batches = int(self.num_training_batches * self.train_percent_check)

        # determine when to check validation
        # if int passed in, val checks that often
        # otherwise, it checks in [0, 1.0] % range of a training epoch
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches:
                raise ValueError(
                    f'`val_check_interval` ({self.val_check_interval}) must be less than or equal '
                    f'to the number of the training batches ({self.num_training_batches}). '
                    'If you want to disable validation set `val_percent_check` to 0.0 instead.')
        else:
            if not _has_len(self.train_dataloader):
                if self.val_check_interval == 1.0:
                    self.val_check_batch = float('inf')
                else:
                    raise MisconfigurationException(
                        'When using an infinite DataLoader (e.g. with an IterableDataset or when '
                        'DataLoader does not implement `__len__`) for `train_dataloader`, '
                        '`Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies '
                        'checking validation every k training batches.')
            else:
                self._percent_range_check('val_check_interval')

                self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
                self.val_check_batch = max(1, self.val_check_batch)

    def _reset_eval_dataloader(self, model: LightningModule,
                               mode: str) -> Tuple[int, List[DataLoader]]:
        """Generic method to reset a dataloader for evaluation.

        Args:
            model: The current `LightningModule`
            mode: Either `'val'` or `'test'`

        Returns:
            Tuple (num_batches, dataloaders)
        """
        dataloaders = self.request_dataloader(getattr(model, f'{mode}_dataloader'))

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        # add samplers
        dataloaders = [self.auto_add_sampler(dl, train=False) for dl in dataloaders if dl]

        num_batches = 0

        # determine number of batches
        # datasets could be none, 1 or 2+
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                self._worker_check(dataloader, f'{mode} dataloader {i}')
                if not _has_len(dataloader):
                    num_batches = float('inf')

            percent_check = getattr(self, f'{mode}_percent_check')

            if num_batches != float('inf'):
                self._percent_range_check(f'{mode}_percent_check')

                num_batches = sum(len(dataloader) for dataloader in dataloaders)
                num_batches = int(num_batches * percent_check)
            elif percent_check not in (0.0, 1.0):
                raise MisconfigurationException(
                    'When using an infinite DataLoader (e.g. with an IterableDataset or when '
                    f'DataLoader does not implement `__len__`) for `{mode}_dataloader`, '
                    f'`Trainer({mode}_percent_check)` must be `0.0` or `1.0`.')
        return num_batches, dataloaders

    def reset_val_dataloader(self, model: LightningModule) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        if self.is_overriden('validation_step'):
            self.num_val_batches, self.val_dataloaders =\
                self._reset_eval_dataloader(model, 'val')

    def reset_test_dataloader(self, model) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        if self.is_overriden('test_step'):
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

        return dataloader

    def determine_data_use_amount(self, train_percent_check: float, val_percent_check: float,
                                  test_percent_check: float, overfit_pct: float) -> None:
        """Use less data for debugging purposes
        """
        self.train_percent_check = train_percent_check
        self.val_percent_check = val_percent_check
        self.test_percent_check = test_percent_check
        if overfit_pct > 0:
            if overfit_pct > 1:
                raise ValueError(
                    f'`overfit_pct` must be not greater than 1.0, but got {overfit_pct:.3f}.')

            self.train_percent_check = overfit_pct
            self.val_percent_check = overfit_pct
            self.test_percent_check = overfit_pct
