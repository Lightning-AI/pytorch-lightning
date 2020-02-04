import warnings
from abc import ABC

import torch.distributed as dist

try:
    # loading for pyTorch 1.3
    from torch.utils.data import IterableDataset
except ImportError:
    # loading for pyTorch 1.1
    import torch
    warnings.warn('Your version of pyTorch %s does not support `IterableDataset`,'
                  ' please upgrade to 1.2+' % torch.__version__, ImportWarning)
    EXIST_ITER_DATASET = False
else:
    EXIST_ITER_DATASET = True
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


class TrainerDataLoadingMixin(ABC):
    def __init__(self):
        # this is just a summary on variables used in this abstract class,
        #  the proper values/initialisation should be done in child class
        self.proc_rank = None
        self.use_ddp = None
        self.use_ddp2 = None
        self.shown_warnings = None
        self.val_check_interval = None
        self.use_tpu = None
        self.tpu_local_core_rank = None

    def _percent_range_check(self, name):
        value = getattr(self, name)
        msg = f"`{name}` must lie in the range [0.0, 1.0], but got {value:.3f}."
        if name == "val_check_interval":
            msg += " If you want to disable validation set `val_percent_check` to 0.0 instead."

        if not 0. <= value <= 1.:
            raise ValueError(msg)

    def init_train_dataloader(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        self.get_train_dataloader = model.train_dataloader

        # determine number of training batches
        if DALI_AVAILABLE and isinstance(self.get_train_dataloader(), DALIGenericIterator):
            self.num_training_batches = self._get_dali_batch_count(self.get_train_dataloader())
        elif EXIST_ITER_DATASET and isinstance(self.get_train_dataloader().dataset, IterableDataset):
            self.num_training_batches = float('inf')
        else:
            self._percent_range_check('train_percent_check')

            self.num_training_batches = len(self.get_train_dataloader())
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
            self._percent_range_check('val_check_interval')

            self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
            self.val_check_batch = max(1, self.val_check_batch)

        self._check_ddp_loader(self.get_train_dataloader())

        # support IterableDataset for train data
        self.is_iterable_train_dataloader = (
            isinstance(self.get_train_dataloader(), DataLoader) and
            EXIST_ITER_DATASET and isinstance(self.get_train_dataloader().dataset, IterableDataset))
        if self.is_iterable_train_dataloader and not isinstance(self.val_check_interval, int):
            m = '''
            When using an iterableDataset for `train_dataloader`,
            `Trainer(val_check_interval)` must be an int.
            An int k specifies checking validation every k training batches
            '''
            raise MisconfigurationException(m)

    def init_val_dataloader(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        self.get_val_dataloaders = model.val_dataloader
        self.num_val_batches = 0

        # determine number of validation batches
        # val datasets could be none, 1 or 2+
        if self.get_val_dataloaders() is not None:
            for dataloader in self.get_val_dataloaders():
                if DALI_AVAILABLE and isinstance(dataloader, DALIGenericIterator):
                    self.num_val_batches += self._get_dali_batch_count(dataloader)
                else:
                    self.num_val_batches += len(dataloader)

            self._percent_range_check('val_percent_check')
            self.num_val_batches = int(self.num_val_batches * self.val_percent_check)

        on_ddp = self.use_ddp or self.use_ddp2
        needs_sampler = on_ddp or self.use_tpu
        if needs_sampler and self.get_val_dataloaders() is not None:
            for dataloader in self.get_val_dataloaders():
                if self._check_ddp_loader(dataloader, "val"):
                    break

    def init_test_dataloader(self, model):
        """Dataloaders are provided by the model.

        :param model:
        """
        self.get_test_dataloaders = model.test_dataloader
        self.num_test_batches = 0

        # determine number of test batches
        if self.get_test_dataloaders() is not None:
            for dataloader in self.get_test_dataloaders():
                if DALI_AVAILABLE and isinstance(dataloader, DALIGenericIterator):
                    self.num_test_batches += self._get_dali_batch_count(dataloader)
                else:
                    self.num_test_batches += len(dataloader)

            self._percent_range_check('test_percent_check')
            self.num_test_batches = int(self.num_test_batches * self.test_percent_check)

        on_ddp = self.use_ddp or self.use_ddp2
        needs_sampler = on_ddp or self.use_tpu
        if needs_sampler and self.get_test_dataloaders() is not None:
            for dataloader in self.get_test_dataloaders():
                if self._check_ddp_loader(dataloader, "test"):
                    break

    def get_dataloaders(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        self.init_train_dataloader(model)
        self.init_test_dataloader(model)
        self.init_val_dataloader(model)

        if self.use_ddp or self.use_ddp2:
            # wait for all processes to catch up
            dist.barrier()

            # load each dataloader
            self.get_train_dataloader()
            self.get_test_dataloaders()
            self.get_val_dataloaders()

        # on TPUs load each dataloader only on process 0
        # this will trigger the data downloads
        if self.use_tpu:
            if self.tpu_local_core_rank == 0:
                self.get_train_dataloader()
                self.get_test_dataloaders()
                self.get_val_dataloaders()

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

    def _check_ddp_loader(self, data_loader, mode="train"):
        """
        Check if sampler is used correctly in ddp mode
        :param data_loader:
        :param mode:
        :return:
        """
        on_ddp = self.use_ddp or self.use_ddp2
        needs_sampler = on_ddp or self.use_tpu
        if needs_sampler and isinstance(data_loader, DataLoader) \
                and not isinstance(data_loader.sampler, DistributedSampler):
            mode_msg = "Your '{}_loader(s)' don't use DistributedSampler.\n".format(mode)
            msg = """
            You're using multiple gpus and multiple nodes, or TPUs without using a
            to assign a subset of your data to each process. To silence this warning, pass a
            DistributedSampler to your DataLoader.

            ie: this:
            dataset = myDataset()
            dataloader = Dataloader(dataset)

            becomes:
            dataset = myDataset()
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = Dataloader(dataset, sampler=dist_sampler)

            If you want each process to load the full dataset, ignore this warning.
            """
            if msg not in self.shown_warnings and self.proc_rank == 0:
                self.shown_warnings.add(msg)
                warnings.warn(mode_msg + msg)
            return True
        return False

    def _get_dali_batch_count(self, data_loader):
        """
        Calculate the batch count in dali iterator
        :param data_loader: DALIGenericIterator
        :return:
        """
        last_batch = 1 if data_loader._fill_last_batch else 0
        return int(data_loader._size / (data_loader._num_gpus * data_loader.batch_size)) + last_batch
