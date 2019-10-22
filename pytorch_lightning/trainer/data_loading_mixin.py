import warnings

import torch.distributed as dist
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class TrainerDataLoadingMixin(object):

    def layout_bookeeping(self):

        # determine number of training batches
        if isinstance(self.get_train_dataloader(), IterableDataset):
            self.nb_training_batches = float('inf')
        else:
            self.nb_training_batches = len(self.get_train_dataloader())
            self.nb_training_batches = int(self.nb_training_batches * self.train_percent_check)

        # determine number of validation batches
        # val datasets could be none, 1 or 2+
        if self.get_val_dataloaders() is not None:
            self.nb_val_batches = sum(len(dataloader) for dataloader in self.get_val_dataloaders())
            self.nb_val_batches = int(self.nb_val_batches * self.val_percent_check)
            self.nb_val_batches = max(1, self.nb_val_batches)

        # determine number of test batches
        if self.get_test_dataloaders() is not None:
            self.nb_test_batches = sum(
                len(dataloader) for dataloader in self.get_test_dataloaders()
            )
            self.nb_test_batches = int(self.nb_test_batches * self.test_percent_check)
            self.nb_test_batches = max(1, self.nb_test_batches)

        # determine when to check validation
        # if int passed in, val checks that often
        # otherwise, it checks in [0, 1.0] % range of a training epoch
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
        else:
            self.val_check_batch = int(self.nb_training_batches * self.val_check_interval)
            self.val_check_batch = max(1, self.val_check_batch)

    def get_dataloaders(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        self.get_train_dataloader = model.train_dataloader
        self.get_test_dataloaders = model.test_dataloader
        self.get_val_dataloaders = model.val_dataloader

        # call warnings from proc zero only which triggers dataloaders
        # if those have to download data it will only happen on proc 0
        if self.proc_rank == 0:
            on_ddp = self.use_ddp or self.use_ddp2
            if on_ddp and not isinstance(self.get_train_dataloader().sampler, DistributedSampler):
                msg = """
                You're using multiple gpus and multiple nodes without using a DistributedSampler
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
                warnings.warn(msg)

            if on_ddp and self.get_val_dataloaders() is not None:
                for dataloader in self.get_val_dataloaders():
                    if not isinstance(dataloader.sampler, DistributedSampler):
                        msg = """
                        Your val_dataloader(s) don't use DistributedSampler.

                        You're using multiple gpus and multiple nodes without using a
                        DistributedSampler to assign a subset of your data to each process.
                        To silence this warning, pass a DistributedSampler to your DataLoader.

                        ie: this:
                        dataset = myDataset()
                        dataloader = Dataloader(dataset)

                        becomes:
                        dataset = myDataset()
                        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                        dataloader = Dataloader(dataset, sampler=dist_sampler)

                        If you want each process to load the full dataset, ignore this warning.
                        """
                        warnings.warn(msg)
                        break

            if on_ddp and self.get_test_dataloaders() is not None:
                for dataloader in self.get_test_dataloaders():
                    if not isinstance(dataloader.sampler, DistributedSampler):
                        msg = """
                        Your test_dataloader(s) don't use DistributedSampler.

                        You're using multiple gpus and multiple nodes without using a
                        DistributedSampler to assign a subset of your data to each process.
                        To silence this warning, pass a DistributedSampler to your DataLoader.

                        ie: this:
                        dataset = myDataset()
                        dataloader = Dataloader(dataset)

                        becomes:
                        dataset = myDataset()
                        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                        dataloader = Dataloader(dataset, sampler=dist_sampler)

                        If you want each process to load the full dataset, ignore this warning.
                        """
                        warnings.warn(msg)
                        break

        if self.use_ddp or self.use_ddp2:
            # wait for all processes to catch up
            dist.barrier()

            # load each dataloader
            self.get_train_dataloader()
            self.get_test_dataloaders()
            self.get_val_dataloaders()

        # support IterableDataset for train data
        self.is_iterable_train_dataloader = isinstance(self.get_train_dataloader(), IterableDataset)
        if self.is_iterable_train_dataloader and not isinstance(self.val_check_interval, int):
            m = '''
            When using an iterableDataset for train_dataloader,
            Trainer(val_check_interval) must be an int.
            An int k specifies checking validation every k training batches
            '''
            raise MisconfigurationException('when using ')

    def determine_data_use_amount(self, train_percent_check, val_percent_check,
                                  test_percent_check, overfit_pct):
        """
        Use less data for debugging purposes
        """
        self.train_percent_check = train_percent_check
        self.val_percent_check = val_percent_check
        self.test_percent_check = test_percent_check
        if overfit_pct > 0:
            self.train_percent_check = overfit_pct
            self.val_percent_check = overfit_pct
            self.test_percent_check = overfit_pct
