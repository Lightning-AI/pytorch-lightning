Fault-tolerant Training
=======================

.. warning:: Fault-tolerant Training is currently an experimental feature within Lightning.

Fault-tolerant Training is an internal mechanism that enables PyTorch Lightning to recover from a hardware or software failure.
This is particularly interesting while training in the cloud with preemptive instances which can shutdown at any time.

Until now, a ``Trainer.fit()`` failing in the middle of an epoch during training or validation
would require the user to restart that epoch completely, losing any progress made during the epoch.
This would make benchmarking non-reproducible as optimization has been interrupted and only partially restored.

With Fault Tolerant Training, when ``Trainer.fit()`` fails in the middle of an epoch during training or validation,
Lightning will restart exactly where it failed, and everything will be restored.

Fault Tolerance requires PyTorch 1.7 or higher and can be enabled as follows:

.. code-block:: bash

    PL_FAULT_TOLERANT_TRAINING=1 python script.py


Under The Hood
--------------

Lightning keeps track of the following state updates during training:

* Samplers indices and random states across multiple processes and workers: This enables restoring random transforms and batch fetching to the exact state as it was right before the failure.
* Optimizers, learning rate schedulers, callbacks, etc..
* Loop progression
* Logging internal states such that metric reductions on epoch end are not getting affected by the failure and model selection can continue as expected.

Currently Supported
-------------------

If you are using a single map-based dataset by sub-classing :class:`~torch.utils.data.Dataset`, everything should work as expected.

.. code-block:: python

    from torch.utils.data import Dataset, DataLoader


    class RandomDataset(Dataset):
        def __init__(self, size: int, length: int):
            self.len = length
            self.data = torch.randn(length, size)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len

If you are using a single iterable-based dataset, there are some limitations. To support fault-tolerance, you will need to use and expose a sampler within your dataset.

For example, the following implementation for an iterable dataset sub-classing :class:`~torch.utils.data.IterableDataset` won't be supported.

.. code-block:: python

    from torch.utils.data import IterableDataset, DataLoader


    # does not support fault tolerance training!
    class RandomIterableDataset(IterableDataset):
        def __init__(self, size: int, count: int):
            self.count = count
            self.size = size

        def __iter__(self):
            for _ in range(self.count):
                yield torch.randn(self.size)


There are two primary reasons why Lightning can't support the previous implementation.

* Lightning cannot infer what you are iterating over, making it difficult to restart training. Lightning Fault Tolerant Training requires a :class:`~torch.utils.data.distributed.Sampler` to be used to encapsulate the fetching logic, requiring both the sampler and an iterator to be made available as attributes within the dataset, so Lightning can access them to track progress.
* Implementing the `__next__` method is required as it separates iterator creation from its consumption, which is essential for Lightning to wrap the iterator before their consumption.

If your iterable dataset are implemented in the following way, everything should works as expected.

.. code-block:: python

    import torch
    from torch.utils.data import IterableDataset, DataLoader


    class RandomIterableDataset(IterableDataset):
        def __init__(self, size: int, length: int):
            self.data = torch.randn(length, size)

            # expose the sampler as an attribute
            self.sampler = RandomSampler(range(length))

        def __iter__(self) -> "RandomIterableDataset":
            # expose the generator from the sampler as an attribute
            # the ``sampler_iter`` will be wrapped by Lightning to ensure
            # we can capture random seeds and iteration count for fast-forward samplers
            # while restarting.
            self.sampler_iter = iter(self.sampler)
            return self

        def __next__(self) -> torch.Tensor:
            # call next on the iterator and get the associated data.
            # the logic here can become more complex but the sampler
            # should be the central piece for fetching the next sample
            index = next(self.sampler_iter)
            return self.data[index]


Current Known Limitations
-------------------------

If you are using multiple training dataloaders, Lightning won't be able to restore the random state properly.

.. testcode::

    class LitModel(LightningModule):
        def train_dataloader(self):
            loader_a = torch.utils.data.DataLoader(range(8), batch_size=4)
            loader_b = torch.utils.data.DataLoader(range(16), batch_size=4)
            return {"loader_a": loader_a, "loader_b": loader_b}

        def training_step(self, batch, batch_idx):
            # access the data in the same format as the collection of dataloaders.
            # dict, list are supported.
            loader_a = batch["loader_a"]
            loader_b = batch["loader_b"]


If you believe this to be useful, please open a `feature request <https://github.com/PyTorchLightning/pytorch-lightning/issues>`_.


Performance Impacts
-------------------

Fault-tolerant Training was tested on common and worst-case scenarios in order to measure the impact of the internal state tracking on the total training time.
On tiny models like the `BoringModel and RandomDataset <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/bug_report/bug_report_model.py>`_
which has virtually no data loading and processing overhead, we noticed up to 50% longer training time with fault tolerance enabled.
In this worst-case scenario, fault-tolerant adds an overhead that is noticeable in comparison to the compute time for dataloading itself.
However, for more realistic training workloads where data loading and preprocessing is more expensive, the constant overhead that fault tolerance adds becomes less noticeable or not noticeable at all.
For example, when training with ResNet50 on CIFAR 10 we have observed a 0.5% to 1% increase in training time depending on ``batch size`` or ``number of workers``.

More detailed benchmarks will be shared in the future.

.. note::

    The extra time is coming from several parts:

    - Capturing the iteration count + random states for each sample within each DataLoader workers and pass it through the data_queue
    - Extra logic to handle / store the dataloader's states from each batch.
