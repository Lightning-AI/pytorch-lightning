.. testsetup:: *

    import torch
    from torch.nn import Module
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.metrics import TensorMetric, NumpyMetric

.. _metrics:

Metrics
=======
Metrics are quantitative measures that can be used to monitor model performance,
and they therefore are essential ingridient in any machine learning project. They
can be used to measure performance during training, comparing different models and
asses pitfalls such as overfitting.

The `pytorch lightning metric package` offers a general package for PyTorch Metrics.
This means that they can be used with regular non-lightning PyTorch code.

In this package, we provide three major pieces of functionality.

1. A Metric class you can use to implement metrics with built-in distributed (ddp) support which are device agnostic.

2. A collection of ready to use popular metrics. These comes with both a functional and class
   based interface.

3. An interface to call `sklearns metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_ .

Example::
    
    # calculate accuracy between two tensors
    from pytorch_lightning.metrics.functional import accuracy

    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 2])

    # calculates accuracy
    accuracy(pred, target)

.. warning::
    The metrics package is still in development! If we're missing a metric or you find a mistake, please send a PR!

----------------

Implement a metric
------------------
While lighning provides a collection of standard used metrics, it is also possible
to implement your own metric using our base interface. All metrics are subclasses 
from our base class ``Metric`` which automatically implements device agnostics
and DDP syncing. That said we recommend that users either subclass from either 

* :class:`TensorMetric` to implement native PyTorch metrics. Will automatically 
  convert all input and output to tensors.

* :class:`NumpyMetric` to implement numpy metrics. Will automatically convert all 
  input between numpy arrays and torch tensors.

It is recommended to use PyTorch metrics when possible, since Numpy metrics slow 
down training because data needs to be converted back and forth between numpy arrays 
and torch tensors.

----------------

TensorMetric
^^^^^^^^^^^^
Here's an example showing how to implement a TensorMetric

.. testcode::

    from pytorch_lightning.metrics import TensorMetric
    class MSE(TensorMetric):
        def forward(self, x, y):
            return torch.mean(torch.pow(x-y, 2.0))

.. autoclass:: pytorch_lightning.metrics.metric.TensorMetric
    :noindex:

----------------

NumpyMetric
^^^^^^^^^^^
Here's an example showing how to implement a NumpyMetric

.. testcode::

    from pytorch_lightning.metrics import NumpyMetric
    class MSE(NumpyMetric):
        def forward(self, x, y):
            return np.mean(np.power(x-y, 2.0))

.. autoclass:: pytorch_lightning.metrics.metric.NumpyMetric
    :noindex:

Metric hooks
^^^^^^^^^^^^

Similar to a standard `torch.nn.Module`, the only *nessesary* method that should
be implemented for a specific metric is ``forward`` method. In this case, output
we automatically be collected and averaged. That said, to gain fine control over 
metric calculation a number of `hooks` can be overridden. The order of evaluation 
is the following:

* ``input_convert``
* ``forward``
* ``output_convert``
* ``ddp_reduce``
    - ``ddp_sync``
    - ``aggregate``
* ``compute``

Note that all hooks are ``@staticmethod``s as default. Additionally, each metric 
has the ``aggregated`` property implemented 

input_convert
"""""""""""""

.. code-block::

    @staticmethod
    def input_convert(self, data: Any):

Pre-hook that implements how input should be converted before passing it to ``forward`` 
The default for ``TensorMetric`` is to convert everything to tensors and ``NumpyMetric``
will convert everything to numpy arrays.


output_convert
""""""""""""""

.. code-block::

    @staticmethod
    def output_convert(self, data: Any, output: Any):

Post-hook that implements how output from ``forward`` should be casted. The default
for both ``TensorMetric`` and ``NumpyMetric`` is do convert to tensors.

ddp_reduce
""""""""""

.. code-block::
    
    @staticmethod
    def ddp_reduce(self, data: Any, output: Any):

Post-hook that implements how output from multiple devices should be collected and
aggregated. We do not recommend overriding this, but instead consider overriding
the two sub-methods called inside this hook: ``ddp_sync`` and ``aggregate``.

ddp_sync
""""""""

.. code-block::

    def ddp_sync(self, tensor: Any):

Method for implementing how output from different devices should be synced. As
default we do a ``gather_all`` such that output from each device is broadcast
to all other devices.

aggregate
"""""""""

.. code-block::

    def aggregate(self, *tensors: torch.Tensor) -> torch.Tensor:

Method on how aggregation should work. As default input will be summed together
over all devices or/and over all batches.

compute
"""""""

.. code-block::

    @staticmethod
    def compute(self, data: Any, output: Any):
    
Post-hook that can be used to implement computations that needs to happen after
output has been synced between devices. As default this will output the average
of the aggregated values.

-------

To summaries, in most cases it should be sufficient to implement ``forward`` (pre-ddp)
and ``compute`` (post-ddp) computations, and the remaining hooks are for special cases.
Below is shown an example of implementing root mean squared error (RMSE) metric where
the root need to be taken after syncing the output to get the right result:

.. testcode::

    from pytorch_lightning.metrics import TensorMetric
    class RMSE(TensorMetric):
        def forward(self, x, y):
            return {'sum_squared_error': torch.pow(x-y, 2.0).sum(),
                    'n_observations': x.numel()}
                    
        @staticmethod
        def compute(self, data, output):
            # sse and n has automatically be synced (summed) over all devices
            sse, n = output['sum_squared_error'], output['n_observations']
            return torch.sqrt(sse / n)


Class Metrics
-------------
Class metrics can be instantiated as part of a module definition (even with just
plain PyTorch). Class metrics are device agnostic, meaning that similar to any
`torch.nn.Module` they will move to the correct device when defined as part of the
model definition.

.. testcode::

    from pytorch_lightning.metrics import Accuracy  

    # Plain PyTorch
    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.metric = Accuracy()

        def forward(self, x, y):
            y_hat = ...
            acc = self.metric(y_hat, y)

    # PyTorch Lightning
    class MyModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.metric = Accuracy()

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = ...
            acc = self.metric(y_hat, y)

These metrics even work when using distributed training:

.. code-block:: python

    model = MyModule()
    trainer = Trainer(gpus=8, num_nodes=2)

    # any metric automatically reduces across GPUs (even the ones you implement using Lightning)
    trainer.fit(model)
    
Class metrics aggregate both over multi device and multiple batches. The aggregated
value can be access through the `metric.aggregated` property. When this property is 
called the internal state is reset.

.. testcode::

    # Plain PyTorch
    metric = Accuracy()
    for pred, target in zip(predictions, target):
        batch_val = metric(pred, target)
    aggregated_val = metric.aggregated
    
    # Pytorch Lightning (evaluation loop)
    class MyModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.metric = Accuracy()
        
        def validation_step(self, batch, batch_idx):
            data, target = batch
            pred = self(data)
            batch_val = self.metric(pred, target)
            
        def validation_epoch_end(self, outputs):
            acc = self.metric.aggregated
            return acc # this will be the aggregated value over the hole validation set

Accuracy
^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.Accuracy
    :noindex:

AveragePrecision
^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.AveragePrecision
    :noindex:

AUROC
^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.AUROC
    :noindex:

BLEUScore
^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.nlp.BLEUScore
    :noindex:

ConfusionMatrix
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.ConfusionMatrix
    :noindex:

DiceCoefficient
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.DiceCoefficient
    :noindex:

EmbeddingSimilarity
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.self_supervised.EmbeddingSimilarity
    :noindex:
    
F1
^^

.. autoclass:: pytorch_lightning.metrics.classification.F1
    :noindex:

FBeta
^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.FBeta
    :noindex:

PrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.PrecisionRecallCurve
    :noindex:

Precision
^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.Precision
    :noindex:

Recall
^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.Recall
    :noindex:

ROC
^^^

.. autoclass:: pytorch_lightning.metrics.classification.ROC
    :noindex:

MAE
^^^

.. autoclass:: pytorch_lightning.metrics.regression.MAE
    :noindex:

MSE
^^^

.. autoclass:: pytorch_lightning.metrics.regression.MSE
    :noindex:

MulticlassROC
^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.MulticlassROC
    :noindex:

MulticlassPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.MulticlassPrecisionRecallCurve
    :noindex:

IoU
^^^

.. autoclass:: pytorch_lightning.metrics.classification.IoU
    :noindex:

RMSE
^^^^

.. autoclass:: pytorch_lightning.metrics.regression.RMSE
    :noindex:

RMSLE
^^^^^

.. autoclass:: pytorch_lightning.metrics.regression.RMSLE
    :noindex:

SSIM
^^^^

.. autoclass:: pytorch_lightning.metrics.regression.SSIM
    :noindex:

----------------

Functional Metrics
------------------
Functional metrics can be called anywhere (even used with just plain PyTorch).

.. code-block:: python

    from pytorch_lightning.metrics.functional import accuracy

    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 2])

    # calculates accuracy across all GPUs and all Nodes used in training
    accuracy(pred, target)


accuracy (F)
^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.accuracy
    :noindex:

auc (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.auc
    :noindex:

auroc (F)
^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.auroc
    :noindex:

average_precision (F)
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.average_precision
    :noindex:

bleu_score (F)
^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.bleu_score
    :noindex:

confusion_matrix (F)
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.confusion_matrix
    :noindex:

dice_score (F)
^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.dice_score
    :noindex:

embedding_similarity (F)
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.embedding_similarity
    :noindex:

f1_score (F)
^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.f1_score
    :noindex:

fbeta_score (F)
^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.fbeta_score
    :noindex:

multiclass_precision_recall_curve (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.multiclass_precision_recall_curve
    :noindex:

multiclass_roc (F)
^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.multiclass_roc
    :noindex:

precision (F)
^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.precision
    :noindex:

precision_recall (F)
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.precision_recall
    :noindex:

precision_recall_curve (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.precision_recall_curve
    :noindex:

recall (F)
^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.recall
    :noindex:

roc (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.roc
    :noindex:

stat_scores (F)
^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.stat_scores
    :noindex:

iou (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.iou
    :noindex:

mse (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.mse
    :noindex:

rmse (F)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.rmse
    :noindex:

mae (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.mae
    :noindex:

rmsle (F)
^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.rmsle
    :noindex:

psnr (F)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.psnr
    :noindex:

ssim (F)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.ssim
    :noindex:

stat_scores_multiple_classes (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.stat_scores_multiple_classes
    :noindex:

----------------

Metric pre-processing
---------------------

We supply a couple of utility functions that may be usefull for converting tensors
to correct input format.

to_categorical (F)
^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.to_categorical
    :noindex:

to_onehot (F)
^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.to_onehot
    :noindex:

----------------

Sklearn interface
-----------------

Lightning supports `sklearns metrics module <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
as a backend for calculating metrics. Sklearns metrics are well tested and robust,
but requires conversion between pytorch and numpy thus may slow down your computations.

To use the sklearn backend of metrics simply import as

.. code-block:: python

    import pytorch_lightning.metrics.sklearns import plm
    metric = plm.Accuracy(normalize=True)
    val = metric(pred, target)

Each converted sklearn metric comes has the same interface as its
original counterpart (e.g. accuracy takes the additional `normalize` keyword).
Like the native Lightning metrics, these converted sklearn metrics also come
with built-in distributed (ddp) support.

SklearnMetric (sk)
^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.SklearnMetric
    :noindex:

Accuracy (sk)
^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Accuracy
    :noindex:

AUC (sk)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.AUC
    :noindex:

AveragePrecision (sk)
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.AveragePrecision
    :noindex:

BalancedAccuracy (sk)
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.BalancedAccuracy
    :noindex:

CohenKappaScore (sk)
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.CohenKappaScore
    :noindex:

ConfusionMatrix (sk)
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.ConfusionMatrix
    :noindex:

DCG (sk)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.DCG
    :noindex:

F1 (sk)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.F1
    :noindex:

FBeta (sk)
^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.FBeta
    :noindex:

Hamming (sk)
^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Hamming
    :noindex:

Hinge (sk)
^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Hinge
    :noindex:

Jaccard (sk)
^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Jaccard
    :noindex:

Precision (sk)
^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Precision
    :noindex:

Recall (sk)
^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Recall
    :noindex:

PrecisionRecallCurve (sk)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.PrecisionRecallCurve
    :noindex:

ROC (sk)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.ROC
    :noindex:

AUROC (sk)
^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.AUROC
    :noindex:

ExplainedVariance (sk)
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.ExplainedVariance
    :noindex:

MeanAbsoluteError (sk)
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.MeanAbsoluteError
    :noindex:

MeanSquaredError (sk)
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.MeanSquaredError
    :noindex:

MeanSquaredLogError (sk)
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.MeanSquaredLogError
    :noindex:

MedianAbsoluteError (sk)
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.MedianAbsoluteError
    :noindex:

R2Score (sk)
^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.R2Score
    :noindex:

MeanPoissonDeviance (sk)
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.MeanPoissonDeviance
    :noindex:

MeanGammaDeviance (sk)
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.MeanGammaDeviance
    :noindex:

MeanTweedieDeviance (sk)
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.MeanTweedieDeviance
    :noindex:
