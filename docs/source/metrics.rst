.. testsetup:: *

    import torch
    from torch.nn import Module
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.metrics import Metric

.. _metrics:

#######
Metrics
#######

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


``Metric`` objects can also be directly logged, in which case Lightning will log
the metric based on ``on_step`` and ``on_epoch`` flags present in ``self.log(...)``.
If ``on_epoch`` is True, the logger automatically logs the end of epoch metric value by calling
``.compute()``.

.. note::
    ``sync_dist``, ``sync_dist_op``, ``sync_dist_group``, ``reduce_fx`` and ``tbptt_reduce_fx``
    flags from ``self.log(...)`` don't affect the metric logging in any manner. The metric class
    contains its own distributed synchronization logic.

    This however is only true for metrics that inherit the base class ``Metric``,
    and thus the functional metric API provides no support for in-built distributed synchronization
    or reduction functions.


.. code-block:: python

    def __init__(self):
        ...
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        logits = self(x)
        ...
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        logits = self(x)
        ...
        self.valid_acc(logits, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

.. note::
    If using metrics in data parallel mode (dp), the metric update/logging should be done
    in the ``<mode>_step_end`` method (where ``<mode>`` is either ``training``, ``validation``
    or ``test``). This is due to metric states else being destroyed after each forward pass,
    leading to wrong accumulation. In practice do the following:

    .. code-block:: python

        def training_step(self, batch, batch_idx):
            data, target = batch
            pred = self(data)
            ...
            return {'loss' : loss, 'preds' : preds, 'target' : target}

        def training_step_end(self, outputs):
            #update and log
            self.metric(outputs['preds'], outputs['target'])
            self.log('metric', self.metric)


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
    total_valid_accuracy = valid_accuracy.compute()

.. note::

    Metrics contain internal states that keep track of the data seen so far.
    Do not mix metric states across training, validation and testing.
    It is highly recommended to re-initialize the metric per mode as
    shown in the examples above.

.. note::

    Metric states are **not** added to the models ``state_dict`` by default.
    To change this, after initializing the metric, the method ``.persistent(mode)`` can
    be used to enable (``mode=True``) or disable (``mode=False``) this behaviour.

*********************
Implementing a Metric
*********************

To implement your custom metric, subclass the base ``Metric`` class and implement the following methods:

- ``__init__()``: Each state variable should be called using ``self.add_state(...)``.
- ``update()``: Any code needed to update the state given any inputs to the metric.
- ``compute()``: Computes a final value from the state of the metric.

All you need to do is call ``add_state`` correctly to implement a custom metric with DDP.
``reset()`` is called on metric state variables added using ``add_state()``.

.. note::

    ``reset()`` is automatically called whenever ``compute()`` is called. Therefore
    you should very rearly have to call ``reset()`` yourself.

To see how metric states are synchronized across distributed processes,
refer to ``add_state()`` docs from the base ``Metric`` class.

Example implementation:

.. code-block:: python

    from pytorch_lightning.metrics import Metric

    class MyAccuracy(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)

            self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            preds, target = self._input_format(preds, target)
            assert preds.shape == target.shape

            self.correct += torch.sum(preds == target)
            self.total += target.numel()

        def compute(self):
            return self.correct.float() / self.total

Metrics support backpropagation, if all computations involved in the metric calculation
are differentiable. However, note that the cached state is detached from the computational
graph and cannot be backpropagated. Not doing this would mean storing the computational
graph for each update call, which can lead to out-of-memory errors.
In practise this means that:

.. code-block:: python

    metric = MyMetric()
    val = metric(pred, target) # this value can be backpropagated
    val = metric.compute() # this value cannot be backpropagated


**********
Metric API
**********

.. autoclass:: pytorch_lightning.metrics.Metric
    :noindex:


****************
Class metrics
****************

.. currentmodule:: pytorch_lightning.metrics


Classification
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Accuracy
    Precision
    Recall
    Fbeta
    ConfusionMatrix

Regression
----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    MeanSquaredError
    MeanAbsoluteError
    MeanSquaredLogError
    ExplainedVariance
    PSNR
    SSIM


******************
Functional Metrics
******************

.. currentmodule:: pytorch_lightning.metrics.functional

The functional metrics follow the simple paradigm: input in, output out. This means,
they don't provide any advanced mechanisms for syncing across DDP nodes or aggregation
over batches. They simply compute the metric value based on the given inputs.

Also the integration within other parts of PyTorch Lightning will never be as tight
as with the class-based interface. If you look for just computing the values,
the functional metrics are the way to go. However, if you are looking for the
best integration and user experience, please consider also to use the class interface.

Classification [func]
---------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    accuracy
    auc
    auroc
    multiclass_auroc
    average_precision
    confusion_matrix
    dice_score
    f1_score
    fbeta_score
    iou
    multiclass_roc
    precision
    precision_recall
    precision_recall_curve
    recall
    roc
    stat_scores
    stat_scores_multiple_classes
    to_categorical
    to_onehot

Regression [func]
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    explained_variance
    mean_absolute_error
    mean_squared_error
    psnr
    mean_squared_log_error
    ssim

NLP [func]
----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    bleu_score


Pairwise [func]
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: functiontemplate.rst

    embedding_similarity [func]
