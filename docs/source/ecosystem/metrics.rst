TorchMetrics
============

`TorchMetrics <https://torchmetrics.readthedocs.io>`_ is a collection of machine learning metrics for distributed,
scalable PyTorch models and an easy-to-use API to create custom metrics. It has a collection of 60+ PyTorch metrics implementations and
is rigorously tested for all edge cases.

.. code-block:: bash

    pip install torchmetrics

In TorchMetrics, we offer the following benefits:

- A standardized interface to increase reproducibility
- Reduced Boilerplate
- Distributed-training compatible
- Rigorously tested
- Automatic accumulation over batches
- Automatic synchronization across multiple devices

-----------------

Example 1: Functional Metrics
-----------------------------

Below is a simple example for calculating the accuracy using the functional interface:

.. code-block:: python

    import torch
    import torchmetrics

    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))

    acc = torchmetrics.functional.accuracy(preds, target)

------------

Example 2: Module Metrics
-------------------------

The example below shows how to use the class-based interface:

.. code-block:: python

    import torch
    import torchmetrics

    # initialize metric
    metric = torchmetrics.Accuracy()

    n_batches = 10
    for i in range(n_batches):
        # simulate a classification problem
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
        # metric on current batch
        acc = metric(preds, target)
        print(f"Accuracy on batch {i}: {acc}")

    # metric on all batches using custom accumulation
    acc = metric.compute()
    print(f"Accuracy on all data: {acc}")

    # Reseting internal state such that metric ready for new data
    metric.reset()

------------

Example 3: TorchMetrics with Lightning
--------------------------------------

The example below shows how to use a metric in your :doc:`LightningModule <../common/lightning_module>`:

.. code-block:: python

    class MyModel(LightningModule):
        def __init__(self):
            ...
            self.accuracy = torchmetrics.Accuracy()

        def training_step(self, batch, batch_idx):
            x, y = batch
            preds = self(x)
            ...
            # log step metric
            self.accuracy(preds, y)
            self.log("train_acc_step", self.accuracy, on_epoch=True)
            ...
