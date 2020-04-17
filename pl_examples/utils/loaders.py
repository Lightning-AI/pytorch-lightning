from threading import Thread
from queue import Queue

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AsynchronousLoader(DataLoader):
    """
    Class for asynchronously loading from CPU memory to device memory

    Parameters
    ----------
    dataset: PyTorch Dataset
        The PyTorch dataset we're loading
    device: PyTorch Device
        The PyTorch device we are loading to
    num_workers: Integer
        Number of worker processes to use for loading from storage and collating the batches in CPU memory
    queue_size: Integer
        Size of the queue used to store the data loaded to the device
    """

    def __init__(self, dataset, device, num_workers=8, queue_size=10, **kwargs):
        super(AsynchronousLoader, self).__init__(dataset=dataset, pin_memory=True, num_workers=num_workers, **kwargs)
        self.device = device
        self.queue_size = queue_size

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0

    def load_loop(self):  # The loop that will load into the queue in the background
        for i, sample in enumerate(super(AsynchronousLoader, self).__iter__()):
            self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):  # Recursive loading for each instance based on torch.utils.data.default_collate
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
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
        # If we've reached the number of batches to return or the queue is empty and the worker is dead then exit
        if (not self.worker.is_alive() and self.queue.empty()) or self.idx >= super(AsynchronousLoader, self).__len__():
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
        return super(AsynchronousLoader, self).__len__()
