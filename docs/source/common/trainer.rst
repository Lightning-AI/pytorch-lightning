.. role:: hidden
    :class: hidden-section

.. testsetup:: *

    import os
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.utilities.seed import seed_everything

.. _trainer:

Trainer
=======

Once you've organized your PyTorch code into a LightningModule,
the Trainer automates everything else.

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pt_trainer_mov.m4v"></video>

|

This abstraction achieves the following:

1. You maintain control over all aspects via PyTorch code without an added abstraction.

2. The trainer uses best practices embedded by contributors and users
   from top AI labs such as Facebook AI Research, NYU, MIT, Stanford, etc...

3. The trainer allows overriding any key part that you don't want automated.

|

-----------

Basic use
---------

This is the basic use of the trainer:

.. code-block:: python

    model = MyLightningModule()

    trainer = Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)

--------

Under the hood
--------------
Under the hood, the Lightning Trainer handles the training loop details for you, some examples include:

- Automatically enabling/disabling grads
- Running the training, validation and test dataloaders
- Calling the Callbacks at the appropriate times
- Putting batches and computations on the correct devices

Here's the pseudocode for what the trainer does under the hood (showing the train loop only)

.. code-block:: python

    # put model in train mode
    model.train()
    torch.set_grad_enabled(True)

    losses = []
    for batch in train_dataloader:
        # calls hooks like this one
        on_train_batch_start()

        # train step
        loss = training_step(batch)

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

        losses.append(loss)


--------

Trainer in Python scripts
-------------------------
In Python scripts, it's recommended you use a main function to call the Trainer.

.. code-block:: python

    from argparse import ArgumentParser

    def main(hparams):
        model = LightningModule()
        trainer = Trainer(gpus=hparams.gpus)
        trainer.fit(model)

    if __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument('--gpus', default=None)
        args = parser.parse_args()

        main(args)

So you can run it like so:

.. code-block:: bash

    python main.py --gpus 2

.. note::

    Pro-tip: You don't need to define all flags manually. Lightning can add them automatically

.. code-block:: python

    from argparse import ArgumentParser

    def main(args):
        model = LightningModule()
        trainer = Trainer.from_argparse_args(args)
        trainer.fit(model)

    if __name__ == '__main__':
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        args = parser.parse_args()

        main(args)

So you can run it like so:

.. code-block:: bash

    python main.py --gpus 2 --max_steps 10 --limit_train_batches 10 --any_trainer_arg x

.. note::
    If you want to stop a training run early, you can press "Ctrl + C" on your keyboard.
    The trainer will catch the ``KeyboardInterrupt`` and attempt a graceful shutdown, including
    running accelerator callback ``on_train_end`` to clean up memory. The trainer object will also set
    an attribute ``interrupted`` to ``True`` in such cases. If you have a callback which shuts down compute
    resources, for example, you can conditionally run the shutdown logic for only uninterrupted runs.

------------

Validation
----------
You can perform an evaluation epoch over the validation set, outside of the training loop,
using :meth:`pytorch_lightning.trainer.trainer.Trainer.validate`. This might be
useful if you want to collect new metrics from a model right at its initialization
or after it has already been trained.

.. code-block:: python

    trainer.validate(val_dataloaders=val_dataloaders)

------------

Testing
-------
Once you're done training, feel free to run the test set!
(Only right before publishing your paper or pushing to production)

.. code-block:: python

    trainer.test(test_dataloaders=test_dataloaders)

------------

Reproducibility
---------------

To ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
and set ``deterministic`` flag in ``Trainer``.

Example::

    from pytorch_lightning import Trainer, seed_everything

    seed_everything(42, workers=True)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    model = Model()
    trainer = Trainer(deterministic=True)


By setting ``workers=True`` in :func:`~pytorch_lightning.utilities.seed.seed_everything`, Lightning derives
unique seeds across all dataloader workers and processes for :mod:`torch`, :mod:`numpy` and stdlib
:mod:`random` number generators. When turned on, it ensures that e.g. data augmentations are not repeated across workers.

-------

Trainer flags
-------------

accelerator
^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/distributed_backend.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/distributed_backend.mp4"></video>

|

The accelerator backend to use (previously known as distributed_backend).

- (``'dp'``) is DataParallel (split batch among GPUs of same machine)
- (``'ddp'``) is DistributedDataParallel (each gpu on each node trains, and syncs grads)
- (``'ddp_cpu'``) is DistributedDataParallel on CPU (same as ``'ddp'``, but does not use GPUs.
  Useful for multi-node CPU training or single-node debugging. Note that this will **not** give
  a speedup on a single node, since Torch already makes efficient use of multiple CPUs on a single
  machine.)
- (``'ddp2'``) dp on node, ddp across nodes. Useful for things like increasing
    the number of negative samples

.. testcode::

    # default used by the Trainer
    trainer = Trainer(accelerator=None)

Example::

    # dp = DataParallel
    trainer = Trainer(gpus=2, accelerator='dp')

    # ddp = DistributedDataParallel
    trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp')

    # ddp2 = DistributedDataParallel + dp
    trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp2')

.. note:: This option does not apply to TPU. TPUs use ``'ddp'`` by default (over each core)

You can also modify hardware behavior by subclassing an existing accelerator to adjust for your needs.

Example::

    class MyOwnAcc(Accelerator):
        ...

    Trainer(accelerator=MyOwnAcc())

.. warning:: Passing in custom accelerators is experimental but work is in progress to enable full compatibility.

accumulate_grad_batches
^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/accumulate_grad_batches.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/accumulate_grad_batches.mp4"></video>

|

Accumulates grads every k batches or as set up in the dict.
Trainer also calls ``optimizer.step()`` for the last indivisible step number.

.. testcode::

    # default used by the Trainer (no accumulation)
    trainer = Trainer(accumulate_grad_batches=1)

Example::

    # accumulate every 4 batches (effective batch size is batch*4)
    trainer = Trainer(accumulate_grad_batches=4)

    # no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
    trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})

amp_backend
^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/amp_backend.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/amp_backend.mp4"></video>

|

Use PyTorch AMP ('native') (available PyTorch 1.6+), or NVIDIA apex ('apex').

.. testcode::

    # using PyTorch built-in AMP, default used by the Trainer
    trainer = Trainer(amp_backend='native')

    # using NVIDIA Apex
    trainer = Trainer(amp_backend='apex')

amp_level
^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/amp_level.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/amp_level.mp4"></video>

|

The optimization level to use (O1, O2, etc...)
for 16-bit GPU precision (using NVIDIA apex under the hood).

Check `NVIDIA apex docs <https://nvidia.github.io/apex/amp.html#opt-levels>`_ for level

Example::

    # default used by the Trainer
    trainer = Trainer(amp_level='O2')

auto_scale_batch_size
^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/auto_scale%E2%80%A8_batch_size.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/auto_scale_batch_size.mp4"></video>

|

Automatically tries to find the largest batch size that fits into memory,
before any training.

.. code-block::

    # default used by the Trainer (no scaling of batch size)
    trainer = Trainer(auto_scale_batch_size=None)

    # run batch size scaling, result overrides hparams.batch_size
    trainer = Trainer(auto_scale_batch_size='binsearch')

    # call tune to find the batch size
    trainer.tune(model)

auto_select_gpus
^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/auto_select+_gpus.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/auto_select_gpus.mp4"></video>

|

If enabled and `gpus` is an integer, pick available gpus automatically.
This is especially useful when GPUs are configured to be in "exclusive mode",
such that only one process at a time can access them.

Example::

    # no auto selection (picks first 2 gpus on system, may fail if other process is occupying)
    trainer = Trainer(gpus=2, auto_select_gpus=False)

    # enable auto selection (will find two available gpus on system)
    trainer = Trainer(gpus=2, auto_select_gpus=True)

    # specifies all GPUs regardless of its availability
    Trainer(gpus=-1, auto_select_gpus=False)

    # specifies all available GPUs (if only one GPU is not occupied, uses one gpu)
    Trainer(gpus=-1, auto_select_gpus=True)

auto_lr_find
^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/auto_lr_find.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/auto_lr_find.mp4"></video>

|

Runs a learning rate finder algorithm (see this `paper <https://arxiv.org/abs/1506.01186>`_)
when calling trainer.tune(), to find optimal initial learning rate.

.. code-block:: python

    # default used by the Trainer (no learning rate finder)
    trainer = Trainer(auto_lr_find=False)

Example::

    # run learning rate finder, results override hparams.learning_rate
    trainer = Trainer(auto_lr_find=True)

    # call tune to find the lr
    trainer.tune(model)

Example::

    # run learning rate finder, results override hparams.my_lr_arg
    trainer = Trainer(auto_lr_find='my_lr_arg')

    # call tune to find the lr
    trainer.tune(model)

.. note::
    See the :doc:`learning rate finder guide <../advanced/lr_finder>`.

benchmark
^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/benchmark.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/benchmark.mp4"></video>

|

If true enables cudnn.benchmark.
This flag is likely to increase the speed of your system if your
input sizes don't change. However, if it does, then it will likely
make your system slower.

The speedup comes from allowing the cudnn auto-tuner to find the best
algorithm for the hardware `[see discussion here]
<https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936>`_.

Example::

    # default used by the Trainer
    trainer = Trainer(benchmark=False)

deterministic
^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/deterministic.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/deterministic.mp4"></video>

|

If true enables cudnn.deterministic.
Might make your system slower, but ensures reproducibility.
Also sets ``$HOROVOD_FUSION_THRESHOLD=0``.

For more info check `[pytorch docs]
<https://pytorch.org/docs/stable/notes/randomness.html>`_.

Example::

    # default used by the Trainer
    trainer = Trainer(deterministic=False)

callbacks
^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/callbacks.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/callbacks.mp4"></video>

|

Add a list of :class:`~pytorch_lightning.callbacks.Callback`. Callbacks run sequentially in the order defined here
with the exception of :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callbacks which run
after all others to ensure all states are saved to the checkpoints.

.. code-block:: python

    # a list of callbacks
    callbacks = [PrintCallback()]
    trainer = Trainer(callbacks=callbacks)

Example::

    from pytorch_lightning.callbacks import Callback

    class PrintCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is started!")
        def on_train_end(self, trainer, pl_module):
            print("Training is done.")


Model-specific callbacks can also be added inside the ``LightningModule`` through
:meth:`~pytorch_lightning.core.lightning.LightningModule.configure_callbacks`.
Callbacks returned in this hook will extend the list initially given to the ``Trainer`` argument, and replace
the trainer callbacks should there be two or more of the same type.
:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callbacks always run last.


check_val_every_n_epoch
^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/check_val_every_n_epoch.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/check_val_every_n_epoch.mp4"></video>

|

Check val every n train epochs.

Example::

    # default used by the Trainer
    trainer = Trainer(check_val_every_n_epoch=1)

    # run val loop every 10 training epochs
    trainer = Trainer(check_val_every_n_epoch=10)

checkpoint_callback
^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/checkpoint_callback.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/checkpoint_callback.mp4"></video>

|

By default Lightning saves a checkpoint for you in your current working directory, with the state of your last training epoch,
Checkpoints capture the exact value of all parameters used by a model.
To disable automatic checkpointing, set this to `False`.

.. code-block:: python

    # default used by Trainer
    trainer = Trainer(checkpoint_callback=True)

    # turn off automatic checkpointing
    trainer = Trainer(checkpoint_callback=False)


You can override the default behavior by initializing the :class:`~pytorch_lightning.callbacks.ModelCheckpoint`
callback, and adding it to the :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks` list.
See :doc:`Saving and Loading Weights <../common/weights_loading>` for how to customize checkpointing.

.. testcode::

    from pytorch_lightning.callbacks import ModelCheckpoint
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    # Add your callback to the callbacks list
    trainer = Trainer(callbacks=[checkpoint_callback])


.. warning:: Passing a ModelCheckpoint instance to this argument is deprecated since
    v1.1 and will be unsupported from v1.3. Use `callbacks` argument instead.


default_root_dir
^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/default%E2%80%A8_root_dir.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/default_root_dir.mp4"></video>

|

Default path for logs and weights when no logger or
:class:`pytorch_lightning.callbacks.ModelCheckpoint` callback passed.  On
certain clusters you might want to separate where logs and checkpoints are
stored. If you don't then use this argument for convenience. Paths can be local
paths or remote paths such as `s3://bucket/path` or 'hdfs://path/'. Credentials
will need to be set up to use remote filepaths.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(default_root_dir=os.getcwd())

distributed_backend
^^^^^^^^^^^^^^^^^^^
Deprecated: This has been renamed ``accelerator``.

fast_dev_run
^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/fast_dev_run.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/fast_dev_run.mp4"></video>

|

Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test
to find any bugs (ie: a sort of unit test).

Under the hood the pseudocode looks like this when running *fast_dev_run* with a single batch:

.. code-block:: python

    # loading
    __init__()
    prepare_data

    # test training step
    training_batch = next(train_dataloader)
    training_step(training_batch)

    # test val step
    val_batch = next(val_dataloader)
    out = validation_step(val_batch)
    validation_epoch_end([out])

.. testcode::

    # default used by the Trainer
    trainer = Trainer(fast_dev_run=False)

    # runs 1 train, val, test batch and program ends
    trainer = Trainer(fast_dev_run=True)

    # runs 7 train, val, test batches and program ends
    trainer = Trainer(fast_dev_run=7)

.. note::

    This argument is a bit different from ``limit_train/val/test_batches``. Setting this argument will
    disable tuner, checkpoint callbacks, early stopping callbacks, loggers and logger callbacks like
    ``LearningRateLogger`` and runs for only 1 epoch. This must be used only for debugging purposes.
    ``limit_train/val/test_batches`` only limits the number of batches and won't disable anything.

flush_logs_every_n_steps
^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/flush_logs%E2%80%A8_every_n_steps.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/flush_logs_every_n_steps.mp4"></video>

|

Writes logs to disk this often.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(flush_logs_every_n_steps=100)

See Also:
    - :doc:`logging <../extensions/logging>`

gpus
^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/gpus.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/gpus.mp4"></video>

|

- Number of GPUs to train on (int)
- or which GPUs to train on (list)
- can handle strings

.. testcode::

    # default used by the Trainer (ie: train on CPU)
    trainer = Trainer(gpus=None)

    # equivalent
    trainer = Trainer(gpus=0)

Example::

    # int: train on 2 gpus
    trainer = Trainer(gpus=2)

    # list: train on GPUs 1, 4 (by bus ordering)
    trainer = Trainer(gpus=[1, 4])
    trainer = Trainer(gpus='1, 4') # equivalent

    # -1: train on all gpus
    trainer = Trainer(gpus=-1)
    trainer = Trainer(gpus='-1') # equivalent

    # combine with num_nodes to train on multiple GPUs across nodes
    # uses 8 gpus in total
    trainer = Trainer(gpus=2, num_nodes=4)

    # train only on GPUs 1 and 4 across nodes
    trainer = Trainer(gpus=[1, 4], num_nodes=4)

See Also:
    - :doc:`Multi-GPU training guide <../advanced/multi_gpu>`.

gradient_clip_val
^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/gradient+_clip_val.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/gradient_clip_val.mp4"></video>

|

Gradient clipping value

- 0 means don't clip.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(gradient_clip_val=0.0)

limit_train_batches
^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/limit_train_batches.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/limit_batches.mp4"></video>

|

How much of training dataset to check.
Useful when debugging or testing something that happens at the end of an epoch.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(limit_train_batches=1.0)

Example::

    # default used by the Trainer
    trainer = Trainer(limit_train_batches=1.0)

    # run through only 25% of the training set each epoch
    trainer = Trainer(limit_train_batches=0.25)

    # run through only 10 batches of the training set each epoch
    trainer = Trainer(limit_train_batches=10)

limit_test_batches
^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/limit_test_batches.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/limit_batches.mp4"></video>

|

How much of test dataset to check.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(limit_test_batches=1.0)

    # run through only 25% of the test set each epoch
    trainer = Trainer(limit_test_batches=0.25)

    # run for only 10 batches
    trainer = Trainer(limit_test_batches=10)

In the case of multiple test dataloaders, the limit applies to each dataloader individually.

limit_val_batches
^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/limit_val_batches.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/limit_batches.mp4"></video>

|

How much of validation dataset to check.
Useful when debugging or testing something that happens at the end of an epoch.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(limit_val_batches=1.0)

    # run through only 25% of the validation set each epoch
    trainer = Trainer(limit_val_batches=0.25)

    # run for only 10 batches
    trainer = Trainer(limit_val_batches=10)

In the case of multiple validation dataloaders, the limit applies to each dataloader individually.

log_every_n_steps
^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/log_every_n_steps.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/log_every_n_steps.mp4"></video>

|


How often to add logging rows (does not write to disk)

.. testcode::

    # default used by the Trainer
    trainer = Trainer(log_every_n_steps=50)

See Also:
    - :doc:`logging <../extensions/logging>`

log_gpu_memory
^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/log_gpu_memory.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/log_gpu_memory.mp4"></video>

|

Options:

- None
- 'min_max'
- 'all'

.. testcode::

    # default used by the Trainer
    trainer = Trainer(log_gpu_memory=None)

    # log all the GPUs (on master node only)
    trainer = Trainer(log_gpu_memory='all')

    # log only the min and max memory on the master node
    trainer = Trainer(log_gpu_memory='min_max')

.. note:: Might slow performance because it uses the output of ``nvidia-smi``.

logger
^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/logger.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/logger.mp4"></video>

|

:doc:`Logger <../common/loggers>` (or iterable collection of loggers) for experiment tracking. A ``True`` value uses the default ``TensorBoardLogger`` shown below. ``False`` will disable logging.

.. testcode::

    from pytorch_lightning.loggers import TensorBoardLogger

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )
    Trainer(logger=logger)

max_epochs
^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/max_epochs.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/min_max_epochs.mp4"></video>

|

Stop training once this number of epochs is reached

.. testcode::

    # default used by the Trainer
    trainer = Trainer(max_epochs=1000)

min_epochs
^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/min_epochs.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/min_max_epochs.mp4"></video>

|

Force training for at least these many epochs

.. testcode::

    # default used by the Trainer
    trainer = Trainer(min_epochs=1)

max_steps
^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/max_steps.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/min_max_steps.mp4"></video>

|

Stop training after this number of steps
Training will stop if max_steps or max_epochs have reached (earliest).

.. testcode::

    # Default (disabled)
    trainer = Trainer(max_steps=None)

    # Stop after 100 steps
    trainer = Trainer(max_steps=100)

min_steps
^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/min_steps.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/min_max_steps.mp4"></video>

|

Force training for at least these number of steps.
Trainer will train model for at least min_steps or min_epochs (latest).

.. testcode::

    # Default (disabled)
    trainer = Trainer(min_steps=None)

    # Run at least for 100 steps (disable min_epochs)
    trainer = Trainer(min_steps=100, min_epochs=0)

max_time
^^^^^^^^

Set the maximum amount of time for training. Training will get interrupted mid-epoch.
For customizable options use the :class:`~pytorch_lightning.callbacks.timer.Timer` callback.

.. testcode::

    # Default (disabled)
    trainer = Trainer(max_time=None)

    # Stop after 12 hours of training or when reaching 10 epochs (string)
    trainer = Trainer(max_time="00:12:00:00", max_epochs=10)

    # Stop after 1 day and 5 hours (dict)
    trainer = Trainer(max_time={"days": 1, "hours": 5})

In case ``max_time`` is used together with ``min_steps`` or ``min_epochs``, the ``min_*`` requirement
always has precedence.

num_nodes
^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/num_nodes.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/num_nodes.mp4"></video>

|

Number of GPU nodes for distributed training.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(num_nodes=1)

    # to train on 8 nodes
    trainer = Trainer(num_nodes=8)

num_processes
^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/num_processes.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/num_processes.mp4"></video>

|

Number of processes to train with. Automatically set to the number of GPUs
when using ``accelerator="ddp"``. Set to a number greater than 1 when
using ``accelerator="ddp_cpu"`` to mimic distributed training on a
machine without GPUs. This is useful for debugging, but **will not** provide
any speedup, since single-process Torch already makes efficient use of multiple
CPUs.

.. testcode::

    # Simulate DDP for debugging on your GPU-less laptop
    trainer = Trainer(accelerator="ddp_cpu", num_processes=2)

num_sanity_val_steps
^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/num_sanity%E2%80%A8_val_steps.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/num_sanity_val_steps.mp4"></video>

|

Sanity check runs n batches of val before starting the training routine.
This catches any bugs in your validation without having to wait for the first validation check.
The Trainer uses 2 steps by default. Turn it off or modify it here.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(num_sanity_val_steps=2)

    # turn it off
    trainer = Trainer(num_sanity_val_steps=0)

    # check all validation data
    trainer = Trainer(num_sanity_val_steps=-1)


This option will reset the validation dataloader unless ``num_sanity_val_steps=0``.

overfit_batches
^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/overfit_batches.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/overfit_batches.mp4"></video>

|

Uses this much data of the training set. If nonzero, will use the same training set for validation and testing.
If the training dataloaders have `shuffle=True`, Lightning will automatically disable it.

Useful for quickly debugging or trying to overfit on purpose.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(overfit_batches=0.0)

    # use only 1% of the train set (and use the train set for val and test)
    trainer = Trainer(overfit_batches=0.01)

    # overfit on 10 of the same batches
    trainer = Trainer(overfit_batches=10)

plugins
^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/cluster_environment.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/cluster_environment.mp4"></video>

|

:ref:`Plugins` allow you to connect arbitrary backends, precision libraries, clusters etc. For example:

- :ref:`DDP <multi_gpu>`
- `TorchElastic <https://pytorch.org/elastic/0.2.2/index.html>`_
- :ref:`Apex <amp>`

To define your own behavior, subclass the relevant class and pass it in. Here's an example linking up your own
:class:`~pytorch_lightning.plugins.environments.ClusterEnvironment`.

.. code-block:: python

    from pytorch_lightning.plugins.environments import ClusterEnvironment

    class MyCluster(ClusterEnvironment):

        def master_address(self):
            return your_master_address

        def master_port(self):
            return your_master_port

        def world_size(self):
            return the_world_size

    trainer = Trainer(plugins=[MyCluster()], ...)


prepare_data_per_node
^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/prepare_data_per_node.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/prepare_data_per_node.mp4"></video>

|

If True will call `prepare_data()` on LOCAL_RANK=0 for every node.
If False will only call from NODE_RANK=0, LOCAL_RANK=0

.. testcode::

    # default
    Trainer(prepare_data_per_node=True)

    # use only NODE_RANK=0, LOCAL_RANK=0
    Trainer(prepare_data_per_node=False)

precision
^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/precision.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/precision.mp4"></video>

|

Double precision (64), full precision (32) or half precision (16).
Can all be used on GPU or TPUs. Only double (64) and full precision (32) available on CPU.

If used on TPU will use torch.bfloat16 but tensor printing
will still show torch.float32.

.. testcode::
    :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

    # default used by the Trainer
    trainer = Trainer(precision=32)

    # 16-bit precision
    trainer = Trainer(precision=16, gpus=1)

    # 64-bit precision
    trainer = Trainer(precision=64)

Example::

    # one day
    trainer = Trainer(precision=8|4|2)

process_position
^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/process_position.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/process_position.mp4"></video>

|

Orders the progress bar. Useful when running multiple trainers on the same node.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(process_position=0)

.. note:: This argument is ignored if a custom callback is passed to :paramref:`~Trainer.callbacks`.

profiler
^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/profiler.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/profiler.mp4"></video>

|

To profile individual steps during training and assist in identifying bottlenecks.

See the :doc:`profiler documentation <../advanced/profiler>`. for more details.

.. testcode::

    from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

    # default used by the Trainer
    trainer = Trainer(profiler=None)

    # to profile standard training events, equivalent to `profiler=SimpleProfiler()`
    trainer = Trainer(profiler="simple")

    # advanced profiler for function-level stats, equivalent to `profiler=AdvancedProfiler()`
    trainer = Trainer(profiler="advanced")

progress_bar_refresh_rate
^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/progress_bar%E2%80%A8_refresh_rate.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/progress_bar_refresh_rate.mp4"></video>

|

How often to refresh progress bar (in steps).

.. testcode::

    # default used by the Trainer
    trainer = Trainer(progress_bar_refresh_rate=1)

    # disable progress bar
    trainer = Trainer(progress_bar_refresh_rate=0)

Note:
    - In Google Colab notebooks, faster refresh rates (lower number) is known to crash them because of their screen refresh rates.
      Lightning will set it to 20 in these environments if the user does not provide a value.
    - This argument is ignored if a custom callback is passed to :paramref:`~Trainer.callbacks`.

reload_dataloaders_every_epoch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/reload_%E2%80%A8dataloaders_%E2%80%A8every_epoch.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/reload_dataloaders_every_epoch.mp4"></video>

|

Set to True to reload dataloaders every epoch.

.. code-block:: python

    # if False (default)
    train_loader = model.train_dataloader()
    for epoch in epochs:
        for batch in train_loader:
            ...

    # if True
    for epoch in epochs:
        train_loader = model.train_dataloader()
        for batch in train_loader:

.. _replace-sampler-ddp:

replace_sampler_ddp
^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/replace_sampler_ddp.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/replace_sampler_ddp.mp4"></video>

|

Enables auto adding of :class:`~torch.utils.data.distributed.DistributedSampler`. In PyTorch, you must use it in
distributed settings such as TPUs or multi-node. The sampler makes sure each GPU sees the appropriate part of your data.
By default it will add ``shuffle=True`` for train sampler and ``shuffle=False`` for val/test sampler.
If you want to customize it, you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.
If ``replace_sampler_ddp=True`` and a distributed sampler was already added,
Lightning will not replace the existing one.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(replace_sampler_ddp=True)

By setting to False, you have to add your own distributed sampler:

.. code-block:: python


    # in your LightningModule or LightningDataModule
    def train_dataloader(self):
        # default used by the Trainer
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
        return dataloader

.. note:: For iterable datasets, we don't do this automatically.

resume_from_checkpoint
^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/resume_from_checkpoint.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/resume_from_checkpoint.mp4"></video>

|

To resume training from a specific checkpoint pass in the path here. If resuming from a mid-epoch
checkpoint, training will start from the beginning of the next epoch.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(resume_from_checkpoint=None)

    # resume from a specific checkpoint
    trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

sync_batchnorm
^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/sync_batchnorm.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/sync_batchnorm.mp4"></video>

|

Enable synchronization between batchnorm layers across all GPUs.

.. testcode::

    trainer = Trainer(sync_batchnorm=True)

track_grad_norm
^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/track_grad_norm.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/track_grad_norm.mp4"></video>

|

- no tracking (-1)
- Otherwise tracks that norm (2 for 2-norm)

.. testcode::

    # default used by the Trainer
    trainer = Trainer(track_grad_norm=-1)

    # track the 2-norm
    trainer = Trainer(track_grad_norm=2)

tpu_cores
^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/tpu_cores.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/tpu_cores.mp4"></video>

|

- How many TPU cores to train on (1 or 8).
- Which TPU core to train on [1-8]

A single TPU v2 or v3 has 8 cores. A TPU pod has
up to 2048 cores. A slice of a POD means you get as many cores
as you request.

Your effective batch size is batch_size * total tpu cores.

This parameter can be either 1 or 8.

Example::

    # your_trainer_file.py

    # default used by the Trainer (ie: train on CPU)
    trainer = Trainer(tpu_cores=None)

    # int: train on a single core
    trainer = Trainer(tpu_cores=1)

    # list: train on a single selected core
    trainer = Trainer(tpu_cores=[2])

    # int: train on all cores few cores
    trainer = Trainer(tpu_cores=8)

    # for 8+ cores must submit via xla script with
    # a max of 8 cores specified. The XLA script
    # will duplicate script onto each TPU in the POD
    trainer = Trainer(tpu_cores=8)

To train on more than 8 cores (ie: a POD),
submit this script using the xla_dist script.

Example::

    python -m torch_xla.distributed.xla_dist
    --tpu=$TPU_POD_NAME
    --conda-env=torch-xla-nightly
    --env=XLA_USE_BF16=1
    -- python your_trainer_file.py

truncated_bptt_steps
^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/truncated_bptt_steps.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/truncated_bptt_steps.mp4"></video>

|

Truncated back prop breaks performs backprop every k steps of
a much longer sequence.

If this is enabled, your batches will automatically get truncated
and the trainer will apply Truncated Backprop to it.

(`Williams et al. "An efficient gradient-based algorithm for on-line training of
recurrent network trajectories."
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.7941&rep=rep1&type=pdf>`_)

.. testcode::

    # default used by the Trainer (ie: disabled)
    trainer = Trainer(truncated_bptt_steps=None)

    # backprop every 5 steps in a batch
    trainer = Trainer(truncated_bptt_steps=5)

.. note::  Make sure your batches have a sequence dimension.

Lightning takes care to split your batch along the time-dimension.

.. code-block:: python

    # we use the second as the time dimension
    # (batch, time, ...)
    sub_batch = batch[0, 0:t, ...]

Using this feature requires updating your LightningModule's
:meth:`pytorch_lightning.core.LightningModule.training_step` to include a `hiddens` arg
with the hidden

.. code-block:: python

        # Truncated back-propagation through time
        def training_step(self, batch, batch_idx, hiddens):
            # hiddens are the hiddens from the previous truncated backprop step
            out, hiddens = self.lstm(data, hiddens)
            return {
                "loss": ...,
                "hiddens": hiddens
            }

To modify how the batch is split,
override :meth:`pytorch_lightning.core.LightningModule.tbptt_split_batch`:

.. testcode::

    class LitMNIST(LightningModule):
        def tbptt_split_batch(self, batch, split_size):
            # do your own splitting on the batch
            return splits

val_check_interval
^^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/val_check_interval.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/val_check_interval.mp4"></video>

|

How often within one training epoch to check the validation set.
Can specify as float or int.

- use (float) to check within a training epoch
- use (int) to check every n steps (batches)

.. testcode::

    # default used by the Trainer
    trainer = Trainer(val_check_interval=1.0)

    # check validation set 4 times during a training epoch
    trainer = Trainer(val_check_interval=0.25)

    # check validation set every 1000 training batches
    # use this when using iterableDataset and your dataset has no length
    # (ie: production cases with streaming data)
    trainer = Trainer(val_check_interval=1000)


weights_save_path
^^^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/weights_save_path.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/weights_save_path.mp4"></video>

|

Directory of where to save weights if specified.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(weights_save_path=os.getcwd())

    # save to your custom path
    trainer = Trainer(weights_save_path='my/path')

Example::

    # if checkpoint callback used, then overrides the weights path
    # **NOTE: this saves weights to some/path NOT my/path
    checkpoint = ModelCheckpoint(dirpath='some/path')
    trainer = Trainer(
        callbacks=[checkpoint],
        weights_save_path='my/path'
    )

weights_summary
^^^^^^^^^^^^^^^

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/weights_summary.jpg"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/weights_summary.mp4"></video>

|

Prints a summary of the weights when training begins.
Options: 'full', 'top', None.

.. testcode::

    # default used by the Trainer (ie: print summary of top level modules)
    trainer = Trainer(weights_summary='top')

    # print full summary of all modules and submodules
    trainer = Trainer(weights_summary='full')

    # don't print a summary
    trainer = Trainer(weights_summary=None)

-----

Trainer class API
-----------------

Methods
^^^^^^^

init
****

.. automethod:: pytorch_lightning.trainer.Trainer.__init__
   :noindex:

fit
****

.. automethod:: pytorch_lightning.trainer.Trainer.fit
   :noindex:

validate
********

.. automethod:: pytorch_lightning.trainer.Trainer.validate
   :noindex:

test
****

.. automethod:: pytorch_lightning.trainer.Trainer.test
   :noindex:

predict
*******

.. automethod:: pytorch_lightning.trainer.Trainer.predict
   :noindex:

tune
****

.. automethod:: pytorch_lightning.trainer.Trainer.tune
   :noindex:

Properties
^^^^^^^^^^

callback_metrics
****************

The metrics available to callbacks. These are automatically set when you log via `self.log`

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log('a_val', 2)


    callback_metrics = trainer.callback_metrics
    assert callback_metrics['a_val'] == 2

current_epoch
*************

The current epoch

.. code-block:: python

    def training_step(self, batch, batch_idx):
        current_epoch = self.trainer.current_epoch
        if current_epoch > 100:
            # do something
            pass


logger (p)
**********

The current logger being used. Here's an example using tensorboard

.. code-block:: python

    def training_step(self, batch, batch_idx):
        logger = self.trainer.logger
        tensorboard = logger.experiment


logged_metrics
**************

The metrics sent to the logger (visualizer).

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log('a_val', 2, log=True)


    logged_metrics = trainer.logged_metrics
    assert logged_metrics['a_val'] == 2

log_dir
*******
The directory for the current experiment. Use this to save images to, etc...

.. code-block:: python

    def training_step(self, batch, batch_idx):
        img = ...
        save_img(img, self.trainer.log_dir)



is_global_zero
**************

Whether this process is the global zero in multi-node training

.. code-block:: python

    def training_step(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            print('in node 0, accelerator 0')

progress_bar_metrics
********************

The metrics sent to the progress bar.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log('a_val', 2, prog_bar=True)


    progress_bar_metrics = trainer.progress_bar_metrics
    assert progress_bar_metrics['a_val'] == 2
