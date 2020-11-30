.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

.. _training_tricks:

Training Tricks
================
Lightning implements various tricks to help during training

----------

Accumulate gradients
--------------------
Accumulated gradients runs K small batches of size N before doing a backwards pass.
The effect is a large effective batch size of size KxN.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT (ie: no accumulated grads)
    trainer = Trainer(accumulate_grad_batches=1)

----------

Gradient Clipping
-----------------
Gradient clipping may be enabled to avoid exploding gradients. Specifically, this will `clip the gradient
norm <https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_>`_ computed over all model parameters together.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients with norm above 0.5
    trainer = Trainer(gradient_clip_val=0.5)

----------

Auto scaling of batch size
--------------------------
Auto scaling of batch size may be enabled to find the largest batch size that fits into
memory. Larger batch size often yields better estimates of gradients, but may also result in
longer training time. Inspired by https://github.com/BlackHC/toma.

.. seealso:: :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. code-block:: python

    # DEFAULT (ie: don't scale batch size automatically)
    trainer = Trainer(auto_scale_batch_size=None)

    # Autoscale batch size
    trainer = Trainer(auto_scale_batch_size=None|'power'|'binsearch')

    # find the batch size
    trainer.tune(model)

Currently, this feature supports two modes `'power'` scaling and `'binsearch'`
scaling. In `'power'` scaling, starting from a batch size of 1 keeps doubling
the batch size until an out-of-memory (OOM) error is encountered. Setting the
argument to `'binsearch'` will initially also try doubling the batch size until
it encounters an OOM, after which it will do a binary search that will finetune the
batch size. Additionally, it should be noted that the batch size scaler cannot
search for batch sizes larger than the size of the training dataset.


.. note::

    This feature expects that a `batch_size` field is either located as a model attribute
    i.e. `model.batch_size` or as a field in your `hparams` i.e. `model.hparams.batch_size`.
    The field should exist and will be overridden by the results of this algorithm.
    Additionally, your `train_dataloader()` method should depend on this field
    for this feature to work i.e.

    .. code-block:: python

        def train_dataloader(self):
            return DataLoader(train_dataset, batch_size=self.batch_size|self.hparams.batch_size)

.. warning::

    Due to these constraints, this features does *NOT* work when passing dataloaders directly
    to `.fit()`.

The scaling algorithm has a number of parameters that the user can control by
invoking the trainer method `.scale_batch_size` themself (see description below).

.. code-block:: python

    # Use default in trainer construction
    trainer = Trainer()
    tuner = Tuner(trainer)

    # Invoke method
    new_batch_size = tuner.scale_batch_size(model, *extra_parameters_here)

    # Override old batch size
    model.hparams.batch_size = new_batch_size

    # Fit as normal
    trainer.fit(model)

The algorithm in short works by:
    1. Dumping the current state of the model and trainer
    2. Iteratively until convergence or maximum number of tries `max_trials` (default 25) has been reached:
        - Call `fit()` method of trainer. This evaluates `steps_per_trial` (default 3) number of
          training steps. Each training step can trigger an OOM error if the tensors
          (training batch, weights, gradients ect.) allocated during the steps have a
          too large memory footprint.
        - If an OOM error is encountered, decrease batch size else increase it.
          How much the batch size is increased/decreased is determined by the choosen
          stratrgy.
    3. The found batch size is saved to either `model.batch_size` or `model.hparams.batch_size`
    4. Restore the initial state of model and trainer

.. autoclass:: pytorch_lightning.tuner.tuning.Tuner
   :noindex:
   :members: scale_batch_size

.. warning:: Batch size finder is not supported for DDP yet, it is coming soon.


Pipeline Parallelism with Checkpointing to reduce peak memory (beta feature)
----------------------------------------------------------------------------

Pipe Pipeline is a lightning integration of Pipeline Parallelism provided by [Fairscale](https://github.com/facebookresearch/fairscale)

It is one of the component of [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) and [ZeRO-2](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/)

Pipe combines pipeline parallelism with checkpointing to reduce peak memory required to train while minimizing device under-utilization.

Before running, install Fairscale using the command below or install all extras using pip install pytorch-lightning["extra"].

or

```
pip install https://github.com/facebookresearch/fairscale/archive/master.zip
```

.. note:: This feature is supported only with `Trainer(automatic_optimzation=False)`

We except the nn.Sequential model to be set as `.layers` attribute to your LightningModule.

.. code-block:: bash

    from pytorch_lightning.plugins.pipe_plugin import PipePlugin

    class MyModel(LightningModule):

        def __init__(...):

            self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)) # 3 layers

        ....

    model = MyModel()

    # train by balancing your 2 first layers on gpu 0 and last layer gpu 1
    trainer = Trainer(accelerator='ddp', plugins=PipePlugin(balance=[2, 1]))

    trainer.fit(model)


With auto-balancing.

By setting `example_input_array` to your model, we can infer automatically the right balance for your model.

.. code-block:: bash

    from pytorch_lightning.plugins.pipe_plugin import PipePlugin

    class MyModel(LightningModule):

        def __init__(...):

            self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)) # 3 layers

            # used to make an inference and find best balancing for your model
            self._example_input_array = torch.randn((1, 32))

        ....

    model = MyModel()

    # train by balancing your 2 first layers on gpu 0 and last layer gpu 1
    trainer = Trainer(accelerator='ddp', plugins='pipe')

    trainer.fit(model)

Choice your balance either by size or time with `balance_by_size` (default) or `balance_by_time`.

.. code-block:: bash

    from pytorch_lightning.plugins.pipe_plugin import PipePlugin

    class MyModel(LightningModule):

        def __init__(...):

            self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)) # 3 layers

            # used to make an inference and find best balancing for your model
            self._example_input_array = torch.randn((1, 32))

        ....

    model = MyModel()

    # train by balancing your 2 first layers on gpu 0 and last layer gpu 1
    trainer = Trainer(accelerator='ddp', plugins=PipePlugin(balance_mode = "balance_by_time"))

    trainer.fit(model)
