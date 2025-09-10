.. testsetup:: *

    from lightning.pytorch.callbacks import StochasticWeightAveraging

.. _training_tricks:

#############################
Effective Training Techniques
#############################

Lightning implements various techniques to help during training that can help make the training smoother.

----------

********************
Accumulate Gradients
********************

.. include:: ../common/gradient_accumulation.rst

----------

*****************
Gradient Clipping
*****************

Gradient clipping can be enabled to avoid exploding gradients. By default, this will clip the gradient norm by calling
:func:`torch.nn.utils.clip_grad_norm_` computed over all model parameters together.
If the Trainer's ``gradient_clip_algorithm`` is set to ``'value'`` (``'norm'`` by default), this will use instead
:func:`torch.nn.utils.clip_grad_value_` for each parameter instead.

.. note::
    If using mixed precision, the ``gradient_clip_val`` does not need to be changed as the gradients are unscaled
    before applying the clipping function.

.. seealso:: :class:`~lightning.pytorch.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients' global norm to <=0.5 using gradient_clip_algorithm='norm' by default
    trainer = Trainer(gradient_clip_val=0.5)

    # clip gradients' maximum magnitude to <=0.5
    trainer = Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="value")

Read more about :ref:`Configuring Gradient Clipping <configure_gradient_clipping>` for advanced use cases.

----------

****************
Weight Averaging
****************

Weight averaging methods such as Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA) can make your
models generalize better at virtually no additional cost. Averaging smooths the loss landscape thus making it harder to
end up in a local minimum during optimization.

Lightning provides two callbacks to facilitate weight averaging. :class:`~lightning.pytorch.callbacks.WeightAveraging`
is a generic callback that wraps the
`AveragedModel <https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html>`__ class from
PyTorch. It allows SWA, EMA, or a custom averaging strategy to be used. By default, it updates the weights after every
step, but it can be customized to update at specific steps or epochs by overriding the `should_update()` method.

The older :class:`~lightning.pytorch.callbacks.StochasticWeightAveraging` callback is specific to SWA. It starts the SWA
procedure after a certain number of epochs and always runs on every epoch. Additionally, it switches to a constant
learning rate schedule (`SWALR <https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.SWALR.html>`__) when the
procedure starts.

.. seealso::
    For a more detailed explanation of SWA and how it works, read
    `this post <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging>`__ by the PyTorch team.

.. seealso::
    The :class:`~lightning.pytorch.callbacks.WeightAveraging` callback and
    :class:`~lightning.pytorch.callbacks.StochasticWeightAveraging` callback

.. testcode::

    from lightning.pytorch.callbacks import StochasticWeightAveraging, WeightAveraging
    from torch.optim.swa_utils import get_ema_avg_fn

    # Enable Exponential Moving Average after 100 steps
    class EMAWeightAveraging(WeightAveraging):
        def __init__(self):
            super().__init__(avg_fn=get_ema_avg_fn())
        def should_update(self, step_idx=None, epoch_idx=None):
            return (step_idx is not None) and (step_idx >= 100)
    trainer = Trainer(callbacks=EMAWeightAveraging())

    # Enable Stochastic Weight Averaging after 10 epochs with learning rate 0.01
    trainer = Trainer(callbacks=StochasticWeightAveraging(swa_epoch_start=10, swa_lrs=0.01))

----------

.. _batch_size_finder:

*****************
Batch Size Finder
*****************

Auto-scaling of batch size can be enabled to find the largest batch size that fits into
memory. Large batch size often yields a better estimation of the gradients, but may also result in
longer training time. Inspired by https://github.com/BlackHC/toma.

.. seealso:: :class:`~lightning.pytorch.tuner.tuning.Tuner`

.. code-block:: python

    from lightning.pytorch.tuner import Tuner

    # Create a tuner for the trainer
    trainer = Trainer(...)
    tuner = Tuner(trainer)

    # Auto-scale batch size by growing it exponentially (default)
    tuner.scale_batch_size(model, mode="power")

    # Auto-scale batch size with binary search
    tuner.scale_batch_size(model, mode="binsearch")

    # Fit as normal with new batch size
    trainer.fit(model)


Currently, this feature supports two modes ``'power'`` scaling and ``'binsearch'``
scaling. In ``'power'`` scaling, starting from a batch size of 1 keeps doubling
the batch size until an out-of-memory (OOM) error is encountered. Setting the
argument to ``'binsearch'`` will initially also try doubling the batch size until
it encounters an OOM, after which it will do a binary search that will finetune the
batch size. Additionally, it should be noted that the batch size scaler cannot
search for batch sizes larger than the size of the training dataset.

.. note::

    This feature expects that a ``batch_size`` field is either located as a model attribute
    i.e. ``model.batch_size`` or as a field in your ``hparams`` i.e. ``model.hparams.batch_size``.
    Similarly it can work with datamodules too. The field should exist and will be updated by
    the results of this algorithm. Additionally, your ``train_dataloader()`` method should depend
    on this field for this feature to work i.e.

    .. code-block:: python

        # using LightningModule
        class LitModel(LightningModule):
            def __init__(self, batch_size):
                super().__init__()
                self.save_hyperparameters()
                # or
                self.batch_size = batch_size

            def train_dataloader(self):
                return DataLoader(train_dataset, batch_size=self.batch_size | self.hparams.batch_size)


        model = LitModel(batch_size=32)
        trainer = Trainer(...)
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model)


        # using LightningDataModule
        class LitDataModule(LightningDataModule):
            def __init__(self, batch_size):
                super().__init__()
                self.save_hyperparameters()
                # or
                self.batch_size = batch_size

            def train_dataloader(self):
                return DataLoader(train_dataset, batch_size=self.batch_size | self.hparams.batch_size)


        model = MyModel()
        datamodule = LitDataModule(batch_size=32)

        trainer = Trainer(...)
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=datamodule)

    Note that the ``train_dataloader`` can be either part of
    the ``LightningModule`` or ``LightningDataModule``
    as shown above. If both the ``LightningModule``
    and the ``LightningDataModule`` contain a ``train_dataloader``,
    the ``LightningDataModule`` takes precedence.

The algorithm in short works by:
    1. Dumping the current state of the model and trainer
    2. Iteratively until convergence or maximum number of tries ``max_trials`` (default 25) has been reached:
        - Call ``fit()`` method of trainer. This evaluates ``steps_per_trial`` (default 3) number of
          optimization steps. Each training step can trigger an OOM error if the tensors
          (training batch, weights, gradients, etc.) allocated during the steps have a
          too large memory footprint.
        - If an OOM error is encountered, decrease batch size else increase it.
          How much the batch size is increased/decreased is determined by the chosen
          strategy.
    3. The found batch size is saved to either ``model.batch_size`` or ``model.hparams.batch_size``
    4. Restore the initial state of model and trainer

.. warning:: Batch size finder is not yet supported for DDP or any of its variations, it is coming soon.


Customizing Batch Size Finder
=============================

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

1. You can also customize the :class:`~lightning.pytorch.callbacks.batch_size_finder.BatchSizeFinder` callback to run
   at different epochs. This feature is useful while fine-tuning models since you can't always use the same batch size after
   unfreezing the backbone.

.. code-block:: python

    from lightning.pytorch.callbacks import BatchSizeFinder


    class FineTuneBatchSizeFinder(BatchSizeFinder):
        def __init__(self, milestones, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.milestones = milestones

        def on_fit_start(self, *args, **kwargs):
            return

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                self.scale_batch_size(trainer, pl_module)


    trainer = Trainer(callbacks=[FineTuneBatchSizeFinder(milestones=(5, 10))])
    trainer.fit(...)


2. Run batch size finder for ``validate``/``test``/``predict``.

.. code-block:: python

    from lightning.pytorch.callbacks import BatchSizeFinder


    class EvalBatchSizeFinder(BatchSizeFinder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def on_fit_start(self, *args, **kwargs):
            return

        def on_test_start(self, trainer, pl_module):
            self.scale_batch_size(trainer, pl_module)


    trainer = Trainer(callbacks=[EvalBatchSizeFinder()])
    trainer.test(...)


----------

.. _learning_rate_finder:

********************
Learning Rate Finder
********************

For training deep neural networks, selecting a good learning rate is essential
for both better performance and faster convergence. Even optimizers such as
:class:`~torch.optim.Adam` that are self-adjusting the learning rate can benefit from more optimal
choices.

To reduce the amount of guesswork concerning choosing a good initial learning
rate, a `learning rate finder` can be used. As described in `this paper <https://arxiv.org/abs/1506.01186>`_
a learning rate finder does a small run where the learning rate is increased
after each processed batch and the corresponding loss is logged. The result of
this is a ``lr`` vs. ``loss`` plot that can be used as guidance for choosing an optimal
initial learning rate.

.. warning::

    For the moment, this feature only works with models having a single optimizer.


.. note::

    With DDP: Since all the processes run in isolation, only process with ``global_rank=0`` will make the decision to stop the
    learning rate finder and broadcast its results to all other ranks. That means, at the end of LR finder, each process will be running with
    the learning rate found on ``global_rank=0``.


Using Lightning's built-in LR finder
====================================

To enable the learning rate finder, your :doc:`lightning module <../common/lightning_module>` needs to
have a ``learning_rate`` or ``lr`` attribute (or as a field in your ``hparams`` i.e.
``hparams.learning_rate`` or ``hparams.lr``). Then, create the :class:`~lightning.pytorch.tuner.tuning.Tuner` via ``tuner = Tuner(trainer)``
and call ``tuner.lr_find(model)`` to run the LR finder.
The suggested ``learning_rate`` will be written to the console and will be automatically
set to your :doc:`lightning module <../common/lightning_module>`, which can be accessed
via ``self.learning_rate`` or ``self.lr``.

.. code-block:: python

    from lightning.pytorch.tuner import Tuner


    class LitModel(LightningModule):
        def __init__(self, learning_rate):
            super().__init__()
            self.learning_rate = learning_rate
            self.model = Model(...)

        def configure_optimizers(self):
            return Adam(self.parameters(), lr=(self.lr or self.learning_rate))


    model = LitModel()
    trainer = Trainer(...)

    # Create a Tuner
    tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    tuner.lr_find(model)


If your model is using an arbitrary value instead of ``self.lr`` or ``self.learning_rate``, set that value in ``lr_find``:

.. code-block:: python

    model = LitModel()
    trainer = Trainer(...)
    tuner = Tuner(trainer)

    # to set to your own hparams.my_value
    tuner.lr_find(model, attr_name="my_value")


You can also inspect the results of the learning rate finder or just play around
with the parameters of the algorithm. A typical example of this would look like:

.. code-block:: python

    model = MyModelClass(hparams)
    trainer = Trainer()
    tuner = Tuner(trainer)

    # Run learning rate finder
    lr_finder = tuner.lr_find(model)

    # Results can be found in
    print(lr_finder.results)

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.hparams.lr = new_lr

    # Fit model
    trainer.fit(model)

The figure produced by ``lr_finder.plot()`` should look something like the figure
below. It is recommended to not pick the learning rate that achieves the lowest
loss, but instead something in the middle of the sharpest downward slope (red point).
This is the point returned py ``lr_finder.suggestion()``.


Customizing Learning Rate Finder
================================

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

You can also customize the :class:`~lightning.pytorch.callbacks.lr_finder.LearningRateFinder` callback to run at different epochs. This feature is useful while fine-tuning models.

.. code-block:: python

    from lightning.pytorch.callbacks import LearningRateFinder


    class FineTuneLearningRateFinder(LearningRateFinder):
        def __init__(self, milestones, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.milestones = milestones

        def on_fit_start(self, *args, **kwargs):
            return

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                self.lr_find(trainer, pl_module)


    trainer = Trainer(callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))])
    trainer.fit(...)


.. figure:: ../_static/images/trainer/lr_finder.png

----------

**************************
Advanced GPU Optimizations
**************************

When training on single or multiple GPU machines, Lightning offers a host of advanced optimizations to improve throughput, memory efficiency, and model scaling.
Refer to :doc:`Advanced GPU Optimized Training <../advanced/model_parallel>` for more details.

----------


.. _ddp_spawn_shared_memory:

******************************************
Sharing Datasets Across Process Boundaries
******************************************

The :class:`~lightning.pytorch.core.datamodule.LightningDataModule` class provides an organized way to decouple data loading from training logic, with :meth:`~lightning.pytorch.core.hooks.DataHooks.prepare_data` being used for downloading and pre-processing the dataset on a single process, and :meth:`~lightning.pytorch.core.hooks.DataHooks.setup` loading the pre-processed data for each process individually:

.. code-block:: python

    class MNISTDataModule(L.LightningDataModule):
        def prepare_data(self):
            MNIST(self.data_dir, download=True)

        def setup(self, stage: str):
            self.mnist = MNIST(self.data_dir)

        def train_loader(self):
            return DataLoader(self.mnist, batch_size=128)

However, for in-memory datasets, that means that each process will hold a (redundant) replica of the dataset in memory, which may be impractical when using many processes while utilizing datasets that nearly fit into CPU memory, as the memory consumption will scale up linearly with the number of processes.
For example, when training Graph Neural Networks, a common strategy is to load the entire graph into CPU memory for fast access to the entire graph structure and its features, and to then perform neighbor sampling to obtain mini-batches that fit onto the GPU.

A simple way to prevent redundant dataset replicas is to rely on :obj:`torch.multiprocessing` to share the `data automatically between spawned processes via shared memory <https://pytorch.org/docs/stable/notes/multiprocessing.html>`_.
For this, all data pre-loading should be done on the main process inside :meth:`DataModule.__init__`. As a result, all tensor-data will get automatically shared when using the ``'ddp_spawn'`` strategy.

.. warning::

    :obj:`torch.multiprocessing` will send a handle of each individual tensor to other processes.
    In order to prevent any errors due to too many open file handles, try to reduce the number of tensors to share, *e.g.*, by stacking your data into a single tensor.

.. code-block:: python

    class MNISTDataModule(L.LightningDataModule):
        def __init__(self, data_dir: str):
            self.mnist = MNIST(data_dir, download=True, transform=T.ToTensor())

        def train_loader(self):
            return DataLoader(self.mnist, batch_size=128)


    model = Model(...)
    datamodule = MNISTDataModule("data/MNIST")

    trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp_spawn")
    trainer.fit(model, datamodule)

See the `graph-level <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pytorch_lightning/gin.py>`_ and `node-level <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pytorch_lightning/graph_sage.py>`_ prediction examples in PyTorch Geometric for practical use-cases.
