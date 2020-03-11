from abc import ABC, abstractmethod

import torch.distributed as dist
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.utilities.debugging import MisconfigurationException

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
    num_training_batches: int
    val_check_batch: ...
    val_dataloaders: DataLoader
    num_val_batches: int
    test_dataloaders: DataLoader
    num_test_batches: int

    @abstractmethod
    def is_overriden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def _percent_range_check(self, name):
        value = getattr(self, name)
        msg = f"`{name}` must lie in the range [0.0, 1.0], but got {value:.3f}."
        if name == "val_check_interval":
            msg += " If you want to disable validation set `val_percent_check` to 0.0 instead."

        if not 0. <= value <= 1.:
            raise ValueError(msg)

    def auto_add_sampler(self, dataloader, train):
        if self.use_ddp or self.use_ddp2 or self.use_tpu:
            dl_args = {
                'dataset': dataloader.dataset,
                'batch_size': dataloader.batch_size,
                'shuffle': False,
                'num_workers': dataloader.num_workers,
                'collate_fn': dataloader.collate_fn,
                'pin_memory': dataloader.pin_memory,
                'drop_last': dataloader.drop_last,
                'timeout': dataloader.timeout,
                'worker_init_fn': dataloader.worker_init_fn
            }

            if self.use_tpu:
                sampler = DistributedSampler(
                    dataloader.dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal()
                )
                dl_args['shuffle'] = False
            else:
                if train:
                    sampler = DistributedSampler(dataloader.dataset)
                    dl_args['shuffle'] = False
                else:
                    sampler = SequentialSampler(dataloader.dataset)

            dl_args['sampler'] = sampler

            dataloader = DataLoader(**dl_args)
        return dataloader

    def reset_train_dataloader(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """

        self.train_dataloader = self.request_data_loader(model.train_dataloader)
        self.num_training_batches = 0

        # automatically add samplers
        self.train_dataloader = self.auto_add_sampler(self.train_dataloader, train=True)

        self._percent_range_check('train_percent_check')

        if self.is_infinite_dataloader(self.train_dataloader):
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
                    f"`val_check_interval` ({self.val_check_interval}) must be less than or equal "
                    f"to the number of the training batches ({self.num_training_batches}). "
                    f"If you want to disable validation set `val_percent_check` to 0.0 instead.")
        else:
            if self.is_infinite_dataloader(self.train_dataloader):
                m = '''
                When using an infinite DataLoader (e.g. with an IterableDataset or when DataLoader
                does not implement `__len__`) for `train_dataloader`, `Trainer(val_check_interval)`
                must be an int. An int k specifies checking validation every k training batches.
                '''
                raise MisconfigurationException(m)

            self._percent_range_check('val_check_interval')

            self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
            self.val_check_batch = max(1, self.val_check_batch)

    def is_infinite_dataloader(self, dataloader):
        try:
            # try getting the length
            _ = len(dataloader)
            return False
        except TypeError as e:
            return True

    def reset_val_dataloader(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        if not self.is_overriden('validation_step'):
            return

        self.val_dataloaders = self.request_data_loader(model.val_dataloader)
        if not isinstance(self.val_dataloaders, list):
            self.val_dataloaders = [self.val_dataloaders]
        self.num_val_batches = 0

        # add samplers
        self.val_dataloaders = [self.auto_add_sampler(dl, train=False)
                                for dl in self.val_dataloaders if dl]

        # determine number of validation batches
        # val datasets could be none, 1 or 2+
        if self.val_dataloaders is not None:
            self._percent_range_check('val_percent_check')

            self.num_val_batches = sum(len(dataloader) for dataloader in self.val_dataloaders)
            self.num_val_batches = int(self.num_val_batches * self.val_percent_check)

    def reset_test_dataloader(self, model):
        """Dataloaders are provided by the model.

        :param model:
        """
        if not self.is_overriden('test_step'):
            return

        # get actual loader
        self.test_dataloaders = self.request_data_loader(model.test_dataloader)
        if not isinstance(self.test_dataloaders, list):
            self.test_dataloaders = [self.test_dataloaders]
        self.num_test_batches = 0

        # add samplers
        self.test_dataloaders = [self.auto_add_sampler(dl, train=False)
                                 for dl in self.test_dataloaders if dl]

        # determine number of test batches
        if self.test_dataloaders is not None:
            self._percent_range_check('test_percent_check')

            len_sum = sum(len(dataloader) for dataloader in self.test_dataloaders)
            self.num_test_batches = len_sum
            self.num_test_batches = int(self.num_test_batches * self.test_percent_check)

    def request_data_loader(self, data_loader_fx):
        """
        Handles downloading data in the GPU or TPU case.

        :param data_loader_fx:
        :return:
        """
        # get the function we'll use to get data
        if self.use_ddp or self.use_ddp2:
            data_loader = data_loader_fx()

            # all processes wait until data download has happened
            dist.barrier()

        # data download/load on TPU
        elif self.use_tpu and XLA_AVAILABLE:
            data_loader = data_loader_fx()

            # all processes wait until data download has happened
            torch_xla.core.xla_model.rendezvous("pl.TrainerDataLoadingMixin.get_dataloaders")

        # regular start
        else:
            data_loader = data_loader_fx()

        return data_loader

    def determine_data_use_amount(self, train_percent_check, val_percent_check,
                                  test_percent_check, overfit_pct):
        """
        Use less data for debugging purposes
        """
        self.train_percent_check = train_percent_check
        self.val_percent_check = val_percent_check
        self.test_percent_check = test_percent_check
        if overfit_pct > 0:
            if overfit_pct > 1:
                raise ValueError(f"`overfit_pct` must be not greater than 1.0, but got "
                                 f"{overfit_pct:.3f}.")

            self.train_percent_check = overfit_pct
            self.val_percent_check = overfit_pct
            self.test_percent_check = overfit_pct
