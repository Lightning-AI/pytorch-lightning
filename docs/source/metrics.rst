.. testsetup:: *

    import torch
    from torch.nn import Module
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.metrics import Metric

.. _metrics:

Metrics
=======

``pytorch_lightning.metrics`` is a Metrics API created for easy metric development and usage in
PyTorch and PyTorch Lightning. It is rigorously tested for all edge cases and includes a growing list of
common metric implementations.

The metrics API provides ``update()``, ``compute()``, ``reset()`` functions to the user. The metric base class inherits
``nn.Module`` which allows us to call ``metric(...)`` directly. The ``forward()`` method of the base ``Metric`` class
serves the dual purpose of calling ``update()`` on its input and simultanously returning the value of the metric over the
provided input.

These metrics work with DDP in PyTorch and PyTorch Lightning by default. When ``.compute()`` is called in
distributed mode, the internal state of each metric is synced and reduced across each process, so that the
logic present in ``.compute()`` is applied to state information from all processes.

The example below shows how to use a metric in your ``LightningModule``:

.. note::

    For v0.10.0 the user is expected to call ``.compute()`` on the metric at the end each epoch.
    This has been shown in the example below. For v1.0 release, we will integrate metrics
    with logging and ``.compute()`` will be called automatically by PyTorch Lightning.

.. code-block:: python

    def __init__(self):
        ...
        self.accuracy = pl.metrics.Accuracy()
   
    def training_step(self, batch, batch_idx):
        logits = self(x)
        ...
        # log step metric
        self.log('train_acc_step', self.accuracy(logits, y))
        ...
   
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())


This metrics API is independent of PyTorch Lightning. Metrics can directly be used in PyTorch as shown in the example:

.. code-block:: python

    from pytorch_lightning import metrics

    train_accuracy = metrics.Accuracy()
    valid_accuracy = metrics.Accuracy(compute_on_step=False)

    for epoch in range(epochs):
        for x, y in train_data:
            y_hat = model(x)
            
            # training step accuracy
            batch_acc = train_accuracy(y_hat, y)
            
        for x, y in valid_data:
            y_hat = model(x)
            valid_accuracy(y_hat, y)
            
    # total accuracy over all training batches
    total_train_accuracy = train_accuracy.compute()
    
    # total accuracy over all validation batches
    total_valid_accuracy = train_accuracy.compute()

Implementing a Metric
---------------------

To implement your custom metric, subclass the base ``Metric`` class and implement the following methods:

- ``__init__()``: Each state variable should be called using ``self.add_state(...)``.
- ``update()``: Any code needed to update the state given any inputs to the metric.
- ``compute()``: Computes a final value from the state of the metric.

All you need to do is call add_state correctly to implement a custom metric with DDP.
``reset()`` is called on metric state variables added using ``add_state()``.

To see how metric states are synchronized across distributed processes, refer to ``add_state()`` docs
from the base ``Metric`` class.

Example implementation:

.. code-block:: python

    from pytorch_lightning.metrics import Metric

    class MyAccuracy(Metric):
        def __init__(self, ddp_sync_on_step=False):
            super().__init__(ddp_sync_on_step=ddp_sync_on_step)

            self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            preds, target = self._input_format(preds, target)
            assert preds.shape == target.shape

            self.correct += torch.sum(preds == target)
            self.total += target.numel()

        def compute(self):
            return self.correct.float() / self.total

Metric
^^^^^^

.. autoclass:: pytorch_lightning.metrics.Metric
    :noindex:

Classification Metrics
----------------------

Accuracy
^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.Accuracy
    :noindex:

Regression Metrics
------------------

MeanSquaredError
^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.regression.MeanSquaredError
    :noindex:


MeanAbsoluteError
^^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.regression.MeanAbsoluteError
    :noindex:


MeanSquaredLogError
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.regression.MeanSquaredLogError
    :noindex:
