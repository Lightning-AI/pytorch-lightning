from threading import Thread
from queue import Queue

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AsynchronousLoader(object):
    """
    Class for asynchronously loading from CPU memory to device memory with DataLoader

    Note that this only works for single GPU training, multiGPU uses PyTorch's DataParallel or
    DistributedDataParallel which uses its own code for transferring data across GPUs. This could just
    break or make things slower with DataParallel or DistributedDataParallel

    Parameters
    ----------
    dataset: PyTorch Dataset
        The PyTorch dataset we're loading.
        Exactly one of dataset or dataloader must be specified
        This must also be finite and of constant length
    dataloader: PyTorch DataLoader
        The PyTorch DataLoader we're using to load.
        Exactly one of dataset or dataloader must be specified
        This must also be finite and of constant length
    device: PyTorch Device
        The PyTorch device we are loading to
    q_size: Integer
        Size of the queue used to store the data loaded to the device
    **kwargs:
        Any additional arguments to pass to the dataloader if we're constructing one here
    """

    def __init__(self, dataset=None, dataloader=None, device=torch.device('cuda', 0), q_size=10, **kwargs):

        if dataset is not None and dataloader is None:
            self.dataloader = DataLoader(dataset, **kwargs)
        elif dataloader is not None and dataset is None:
            self.dataloader = dataloader
        else:
            raise Exception("Exactly one of dataset or dataloader must be specified")

        self.device = device
        self.q_size = q_size

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.q_size)

        self.idx = 0

    def load_loop(self):  # The loop that will load into the queue in the background
        for i, sample in enumerate(self.dataloader):
            self.queue.put(self.load_instance(sample))

    # Recursive loading for each instance based on torch.utils.data.default_collate
    def load_instance(self, sample):
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                # Can only do asynchronous transfer if we use pin_memory
                if not sample.is_pinned():
                    sample = sample.pin_memory()
                return sample.to(self.device, non_blocking=True)
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        assert self.idx == 0, 'idx must be 0 at the beginning of __iter__.'
        self.worker = Thread(target=self.load_loop)
        self.worker.daemon = True
        self.worker.start()
        return self

    def __next__(self):
        # If we've reached the number of batches to return
        # or the queue is empty and the worker is dead then exit
        done = not self.worker.is_alive() and self.queue.empty()
        done = done or self.idx >= super(AsynchronousLoader, self).__len__()
        if done:
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        else:  # Otherwise return the next batch
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def __len__(self):
        return len(self.dataloader)
