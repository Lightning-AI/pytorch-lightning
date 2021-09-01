Fault Tolerant Training
=======================

.. warning:: Fault Tolerant Training is currently an experimental feature within Lightning.

Fault Tolerance Training is an internal mechanism that enables PyTorch Lightning to recover from a failure in hardware or software.

This is particularly interesting while training in the cloud with preemptive instance which can fail at any time.

Fault Tolerance Training requires PyTorch 1.7 or higher and can be enabled as follows:

.. code-block:: bash

    PL_FAULT_TOLERANT_TRAINING=1 python script.py


Before, when your fitting was failing in the middle of an epoch either in training or validation,
PyTorch Lightning would restart at the next epoch and any progress made in the previous one would be lost.
This would make benchmarking non reproducible as optimization has been interrupted and only partially restored.

With Fault Tolerant Training enabled, when your fitting fails in the middle of an epoch either in training or validation,
Lightning will restart exactly where it fails and everything will be restored.

What does Lightning do exactly ?
--------------------------------

* Lightning keeps track of your samplers indices and random seeds across multiple processes and workers. This enables random transforms and batch fetching to be done in the exact same as it would have without the failure.
* Lightning keeps track of optimizers, lr_schedulers, callbacks, etc..
* Lightning keep tracks of logging internal states, so your metric reduction on epoch end isn't affected by the failure and model selection can continue as expected.

Currently supported
-------------------

If you are using a single map-based dataset, everything should work as expected.

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

If you are using a single iterable-based dataset, there is some limitations. You need to use and expose a sampler within your dataset.

This won't be supported as Lightning can't infer you are iterating over count.

.. code-block:: python

    from torch.utils.data import IterableDataset, DataLoader


    class RandomIterableDataset(IterableDataset):
        def __init__(self, size: int, count: int):
            self.count = count
            self.size = size

        def __iter__(self):
            for _ in range(self.count):
                yield torch.randn(self.size)


A :class:`~torch.utils.data.distributed.Sampler` should always be used to encapsulate the fetching logic
and both the sampler and its iterator should be made available as attributes on the dataset,
so Lightning can access them to track progress.

Here is the recommended way to implement your iterable dataset:

.. code-block:: python

    import torch
    from torch.utils.data import IterableDataset, DataLoader


    class RandomIterableDataset(IterableDataset):
        def __init__(self, size: int, length: int):
            self.data = torch.randn(length, size)
            self.sampler = RandomSampler(range(length))

        def __iter__(self) -> "RandomIterableDataset":
            self.sampler_iter = iter(self.sampler)
            return self

        def __next__(self) -> torch.Tensor:
            index = next(self.sampler_iter)
            return self.data[index]

.. note::

    Implementing the `__next__` method has the advantage of separating the iterator creation from its consumption,
    which is essential for Lightning to properly capture progress.


The current known limitations
-----------------------------

If you are using multiple a collection of train dataloaders, Lightning won't be able to restore the random state properly.

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

Fault Tolerant Training was tested on common and worse case scenarios in the term of performance impacts.

Using the `BoringModel and RandomDataset <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/bug_report_model.py>`_

which represents the worse case scenario as highly optimized for speed due in-memory tensors and single multilayer perceptron layer,
we noticed a 50 % performance drop.

For more traditional training such as a Resnet18 on CIFAR 10, we usually observe a 5% to 15 % range depending on `batch size` or `number of workers`.

More detailed benchmark would be shared in the future.
