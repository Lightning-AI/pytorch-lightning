"""
Once you've organized your PyTorch code into a LightningModule,
the Trainer automates everything else.

.. figure:: /_images/lightning_module/pt_trainer.png
   :alt: Convert from PyTorch to Lightning

This abstraction achieves the following:

    1. You maintain control over all aspects via PyTorch code without an added abstraction.

    2. The trainer uses best practices embedded by contributors and users
       from top AI labs such as Facebook AI Research, NYU, MIT, Stanford, etc...

    3. The trainer allows overriding any key part that you don't want automated.

-----------

Basic use
---------

This is the basic use of the trainer:

.. code-block:: python

    from pytorch_lightning import Trainer

    model = MyLightningModule()

    trainer = Trainer()
    trainer.fit(model)


--------

Best Practices
--------------
For cluster computing, it's recommended you structure your
main.py file this way

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

So you can run it like so:distributed_backend

.. code-block:: bash

    python main.py --gpus 2


.. note::
    If you want to stop a training run early, you can press "Ctrl + C" on your keyboard.
    The trainer will catch the `KeyboardInterrupt` and attempt a graceful shutdown, including
    running callbacks such as `on_train_end`. The trainer object will also set an attribute
    `interrupted` to `True` in such cases. If you have a callback which shuts down compute
    resources, for example, you can conditionally run the shutdown logic for only uninterrupted runs.

------------

Testing
-------
Once you're done training, feel free to run the test set!
(Only right before publishing your paper or pushing to production)

.. code-block:: python

    trainer.test()

------------

Deployment / prediction
-----------------------
You just trained a LightningModule which is also just a torch.nn.Module.
Use it to do whatever!

.. code-block:: python

    # load model
    pretrained_model = LightningModule.load_from_checkpoint(PATH)
    pretrained_model.freeze()

    # use it for finetuning
    def forward(self, x):
        features = pretrained_model(x)
        classes = classifier(features)

    # or for prediction
    out = pretrained_model(x)
    api_write({'response': out}

------------

Reproducibility
---------------

To ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
and set ``deterministic``` flag in ``Trainer``.

.. code-block:: python

    from pytorch-lightning import Trainer, seed_everything

    seed_everything(42)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    model = Model()
    trainer = Trainer(deterministic=True)


-------

Trainer flags
-------------

accumulate_grad_batches
^^^^^^^^^^^^^^^^^^^^^^^
Accumulates grads every k batches or as set up in the dict.

.. code-block:: python

    # default used by the Trainer (no accumulation)
    trainer = Trainer(accumulate_grad_batches=1)

Example::

    # accumulate every 4 batches (effective batch size is batch*4)
    trainer = Trainer(accumulate_grad_batches=4)

    # no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
    trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})

amp_level
^^^^^^^^^
The optimization level to use (O1, O2, etc...)
for 16-bit GPU precision (using NVIDIA apex under the hood).

Check `NVIDIA apex docs <https://nvidia.github.io/apex/amp.html#opt-levels>`_ for level

Example::

    # default used by the Trainer
    trainer = Trainer(amp_level='O1')

auto_scale_batch_size
^^^^^^^^^^^^^^^^^^^^^
Automatically tries to find the largest batch size that fits into memory,
before any training.

.. code-block:: python

    # default used by the Trainer (no scaling of batch size)
    trainer = Trainer(auto_scale_batch_size=None)

    # run batch size scaling, result overrides hparams.batch_size
    trainer = Trainer(auto_scale_batch_size='binsearch')

auto_lr_find
^^^^^^^^^^^^
Runs a learning rate finder algorithm (see this `paper <https://arxiv.org/abs/1506.01186>`_)
before any training, to find optimal initial learning rate.

.. code-block:: python

    # default used by the Trainer (no learning rate finder)
    trainer = Trainer(auto_lr_find=False)

Example::

    # run learning rate finder, results override hparams.learning_rate
    trainer = Trainer(auto_lr_find=True)

    # run learning rate finder, results override hparams.my_lr_arg
    trainer = Trainer(auto_lr_find='my_lr_arg')

.. note::
    See the `learning rate finder guide <lr_finder.rst>`_

benchmark
^^^^^^^^^

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

Add a list of user defined callbacks. These callbacks DO NOT replace the explicit callbacks
(loggers, EarlyStopping or ModelCheckpoint).

.. note:: Only user defined callbacks (ie: Not EarlyStopping or ModelCheckpoint)

.. code-block:: python

    # a list of callbacks
    callbacks = [PrintCallback()]
    trainer = Trainer(callbacks=callbacks)

Example::

    from pytorch_lightning.callbacks import Callback

    class PrintCallback(Callback):
        def on_train_start(self):
            print("Training is started!")
        def on_train_end(self):
            print(f"Training is done. The logs are: {self.trainer.logs}")

check_val_every_n_epoch
^^^^^^^^^^^^^^^^^^^^^^^

Check val every n train epochs.

Example::

    # default used by the Trainer
    trainer = Trainer(check_val_every_n_epoch=1)

    # run val loop every 10 training epochs
    trainer = Trainer(check_val_every_n_epoch=10)

checkpoint_callback
^^^^^^^^^^^^^^^^^^^
Callback for checkpointing.

.. code-block:: python

    trainer = Trainer(checkpoint_callback=checkpoint_callback)

Example::

    from pytorch_lightning.callbacks import ModelCheckpoint

    # default used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

default_root_dir
^^^^^^^^^^^^^^^^^

Default path for logs and weights when no logger
or :class:`pytorch_lightning.callbacks.ModelCheckpoint` callback passed.
On certain clusters you might want to separate where logs and checkpoints
are stored. If you don't then use this method for convenience.

Example::

    # default used by the Trainer
    trainer = Trainer(default_root_path=os.getcwd())

distributed_backend
^^^^^^^^^^^^^^^^^^^
The distributed backend to use.

- (```dp```) is DataParallel (split batch among GPUs of same machine)
- (```ddp```) is DistributedDataParallel (each gpu on each node trains, and syncs grads)
- (```ddp_cpu```) is DistributedDataParallel on CPU (same as `ddp`, but does not use GPUs.
  Useful for multi-node CPU training or single-node debugging. Note that this will **not** give
  a speedup on a single node, since Torch already makes effient use of multiple CPUs on a single
  machine.)
- (```ddp2```) dp on node, ddp across nodes. Useful for things like increasing
    the number of negative samples

.. code-block:: python

    # default used by the Trainer
    trainer = Trainer(distributed_backend=None)

Example::

    # dp = DataParallel
    trainer = Trainer(gpus=2, distributed_backend='dp')

    # ddp = DistributedDataParallel
    trainer = Trainer(gpus=2, num_nodes=2, distributed_backend='ddp')

    # ddp2 = DistributedDataParallel + dp
    trainer = Trainer(gpus=2, num_nodes=2, distributed_backend='ddp2')

.. note:: this option does not apply to TPU. TPUs use ```ddp``` by default (over each core)

early_stop_callback
^^^^^^^^^^^^^^^^^^^

Callback for early stopping.
early_stop_callback (:class:`pytorch_lightning.callbacks.EarlyStopping`)

- ``True``: A default callback monitoring ``'val_loss'`` is created.
   Will raise an error if ``'val_loss'`` is not found.
- ``False``: Early stopping will be disabled.
- ``None``: The default callback monitoring ``'val_loss'`` is created.
- Default: ``None``.

.. code-block:: python

    trainer = Trainer(early_stop_callback=early_stop_callback)

Example::

    from pytorch_lightning.callbacks import EarlyStopping

    # default used by the Trainer
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )

.. note:: If ``'val_loss'`` is not found will work as if early stopping is disabled.

fast_dev_run
^^^^^^^^^^^^

Runs 1 batch of train, test  and val to find any bugs (ie: a sort of unit test).

Under the hood the pseudocode looks like this:

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

Example::

    # default used by the Trainer
    trainer = Trainer(fast_dev_run=False)

    # runs 1 train, val, test  batch and program ends
    trainer = Trainer(fast_dev_run=True)

gpus
^^^^

- Number of GPUs to train on
- or Which GPUs to train on
- can handle strings

Example::

    # default used by the Trainer (ie: train on CPU)
    trainer = Trainer(gpus=None)

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

.. note:: See the `multi-gpu computing guide <multi_gpu.rst>`_

gradient_clip_val
^^^^^^^^^^^^^^^^^
Gradient clipping value

- 0 means don't clip.

Example::

    # default used by the Trainer
    trainer = Trainer(gradient_clip_val=0.0)


gradient_clip:

.. warning:: .. deprecated:: 0.5.0

    Use `gradient_clip_val` instead. Will remove 0.8.0.

log_gpu_memory
^^^^^^^^^^^^^^
Options:

- None
- 'min_max'
- 'all'

Example::

    # default used by the Trainer
    trainer = Trainer(log_gpu_memory=None)

    # log all the GPUs (on master node only)
    trainer = Trainer(log_gpu_memory='all')

    # log only the min and max memory on the master node
    trainer = Trainer(log_gpu_memory='min_max')

.. note:: Might slow performance because it uses the output of nvidia-smi.

log_save_interval
^^^^^^^^^^^^^^^^^

Writes logs to disk this often.

Example::

    # default used by the Trainer
    trainer = Trainer(log_save_interval=100)

logger
^^^^^^

`Logger <loggers.rst>`_ (or iterable collection of loggers) for experiment tracking.

.. code-block:: python

    Trainer(logger=logger)

Example::

    from pytorch_lightning.loggers import TensorBoardLogger

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=self.slurm_job_id,
        name='lightning_logs'
    )

max_epochs
^^^^^^^^^^
Stop training once this number of epochs is reached

Example::

    # default used by the Trainer
    trainer = Trainer(max_epochs=1000)

max_nb_epochs:

.. warning:: .. deprecated:: 0.5.0

    Use `max_epochs` instead. Will remove 0.8.0.

min_epochs
^^^^^^^^^^
Force training for at least these many epochs

Example::

    # default used by the Trainer
    trainer = Trainer(min_epochs=1)

min_nb_epochs:

.. warning:: deprecated:: 0.5.0
    Use `min_epochs` instead. Will remove 0.8.0.

max_steps
^^^^^^^^^
Stop training after this number of steps
Training will stop if max_steps or max_epochs have reached (earliest).

.. code-block:: python

    # Default (disabled)
    trainer = Trainer(max_steps=None)

Example::

    # Stop after 100 steps
    trainer = Trainer(max_steps=100)

min_steps
^^^^^^^^^

Force training for at least these number of steps.
Trainer will train model for at least min_steps or min_epochs (latest).

.. code-block:: python

    # Default (disabled)
    trainer = Trainer(min_steps=None)

Example::

    # Run at least for 100 steps (disable min_epochs)
    trainer = Trainer(min_steps=100, min_epochs=0)

num_nodes
^^^^^^^^^

Number of GPU nodes for distributed training.

Example::

    # default used by the Trainer
    trainer = Trainer(num_nodes=1)

    # to train on 8 nodes
    trainer = Trainer(num_nodes=8)

nb_gpu_nodes:

.. warning:: .. deprecated:: 0.5.0

    Use `num_nodes` instead. Will remove 0.8.0.

num_processes
^^^^^^^^^^^^^

Number of processes to train with. Automatically set to the number of GPUs
when using ``distrbuted_backend="ddp"``. Set to a number greater than 1 when
using ``distributed_backend="ddp_cpu"`` to mimic distributed training on a
machine without GPUs. This is useful for debugging, but **will not** provide
any speedup, since single-process Torch already makes effient use of multiple
CPUs.

Example::

    # Simulate DDP for debugging on your GPU-less laptop
    trainer = Trainer(distributed_backend="ddp_cpu", num_processes=2)

num_sanity_val_steps
^^^^^^^^^^^^^^^^^^^^

Sanity check runs n batches of val before starting the training routine.
This catches any bugs in your validation without having to wait for the first validation check.
The Trainer uses 5 steps by default. Turn it off or modify it here.

Example::

    # default used by the Trainer
    trainer = Trainer(num_sanity_val_steps=5)

    # turn it off
    trainer = Trainer(num_sanity_val_steps=0)

nb_sanity_val_steps:

.. warning:: .. deprecated:: 0.5.0

    Use `num_sanity_val_steps` instead. Will remove 0.8.0.

num_tpu_cores
^^^^^^^^^^^^^
.. warning:: .. deprecated:: 0.7.6

    Use `tpu_cores` instead. Will remove 0.9.0.

Example::

    python -m torch_xla.distributed.xla_dist
    --tpu=$TPU_POD_NAME
    --conda-env=torch-xla-nightly
    --env=XLA_USE_BF16=1
    -- python your_trainer_file.py

tpu_cores
^^^^^^^^^
- How many TPU cores to train on (1 or 8).
- Which TPU core to train on [1-8]

A single TPU v2 or v3 has 8 cores. A TPU pod has
up to 2048 cores. A slice of a POD means you get as many cores
as you request.

Your effective batch size is batch_size * total tpu cores.

.. note:: No need to add a DistributedDataSampler, Lightning automatically does it for you.

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

overfit_pct
^^^^^^^^^^^
Uses this much data of all datasets (training, validation, test).
Useful for quickly debugging or trying to overfit on purpose.

Example::

    # default used by the Trainer
    trainer = Trainer(overfit_pct=0.0)

    # use only 1% of the train, test, val datasets
    trainer = Trainer(overfit_pct=0.01)

    # equivalent:
    trainer = Trainer(
        train_percent_check=0.01,
        val_percent_check=0.01,
        test_percent_check=0.01
    )

See Also:
    - `train_percent_check`_
    - `val_percent_check`_
    - `test_percent_check`_


precision
^^^^^^^^^
Full precision (32), half precision (16).
Can be used on CPU, GPU or TPUs.

If used on TPU will use torch.bfloat16 but tensor printing
will still show torch.float32.

Example::

    # default used by the Trainer
    trainer = Trainer(precision=32)

    # 16-bit precision
    trainer = Trainer(precision=16)

    # one day
    trainer = Trainer(precision=8|4|2)

print_nan_grads
^^^^^^^^^^^^^^^

.. warning:: .. deprecated:: 0.7.2.

    Has no effect. When detected, NaN grads will be printed automatically.
    Will remove 0.9.0.


process_position
^^^^^^^^^^^^^^^^
Orders the progress bar. Useful when running multiple trainers on the same node.

Example::

    # default used by the Trainer
    trainer = Trainer(process_position=0)

Note:
    This argument is ignored if a custom callback is passed to :paramref:`~Trainer.callbacks`.

profiler
^^^^^^^^
To profile individual steps during training and assist in identifying bottlenecks.

See the `profiler documentation <profiler.rst>`_. for more details.

Example::

    from pytorch_lightning.profiler import Profiler, AdvancedProfiler

    # default used by the Trainer
    trainer = Trainer(profiler=None)

    # to profile standard training events
    trainer = Trainer(profiler=True)

    # equivalent to profiler=True
    profiler = Profiler()
    trainer = Trainer(profiler=profiler)

    # advanced profiler for function-level stats
    profiler = AdvancedProfiler()
    trainer = Trainer(profiler=profiler)

progress_bar_refresh_rate
^^^^^^^^^^^^^^^^^^^^^^^^^
How often to refresh progress bar (in steps).
In notebooks, faster refresh rates (lower number) is known to crash them
because of their screen refresh rates, so raise it to 50 or more.

Example::

    # default used by the Trainer
    trainer = Trainer(progress_bar_refresh_rate=1)

    # disable progress bar
    trainer = Trainer(progress_bar_refresh_rate=0)

Note:
    This argument is ignored if a custom callback is passed to :paramref:`~Trainer.callbacks`.

reload_dataloaders_every_epoch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

replace_sampler_ddp
^^^^^^^^^^^^^^^^^^^
Enables auto adding of distributed sampler.

Example::

    # default used by the Trainer
    trainer = Trainer(replace_sampler_ddp=True)

By setting to False, you have to add your own distributed sampler:

Example::

    # default used by the Trainer
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

resume_from_checkpoint
^^^^^^^^^^^^^^^^^^^^^^
To resume training from a specific checkpoint pass in the path here.

Example::

    # default used by the Trainer
    trainer = Trainer(resume_from_checkpoint=None)

    # resume from a specific checkpoint
    trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

row_log_interval
^^^^^^^^^^^^^^^^

How often to add logging rows (does not write to disk)

Example::

    # default used by the Trainer
    trainer = Trainer(row_log_interval=10)


add_row_log_interval:

.. warning:: .. deprecated:: 0.5.0

    Use `row_log_interval` instead. Will remove 0.8.0.

use_amp:

.. warning:: .. deprecated:: 0.7.0

    Use `precision` instead. Will remove 0.9.0.

show_progress_bar
^^^^^^^^^^^^^^^^^

.. warning:: .. deprecated:: 0.7.2

    Set `progress_bar_refresh_rate` to 0 instead. Will remove 0.9.0.

test_percent_check
^^^^^^^^^^^^^^^^^^

How much of test dataset to check.

Example::

    # default used by the Trainer
    trainer = Trainer(test_percent_check=1.0)

    # run through only 25% of the test set each epoch
    trainer = Trainer(test_percent_check=0.25)

val_check_interval
^^^^^^^^^^^^^^^^^^

How often within one training epoch to check the validation set.
Can specify as float or int.

- use (float) to check within a training epoch
- use (int) to check every n steps (batches)

.. code-block:: python

    # default used by the Trainer
    trainer = Trainer(val_check_interval=1.0)

Example::

    # check validation set 4 times during a training epoch
    trainer = Trainer(val_check_interval=0.25)

    # check validation set every 1000 training batches
    # use this when using iterableDataset and your dataset has no length
    # (ie: production cases with streaming data)
    trainer = Trainer(val_check_interval=1000)

track_grad_norm
^^^^^^^^^^^^^^^

- no tracking (-1)
- Otherwise tracks that norm (2 for 2-norm)

.. code-block:: python

    # default used by the Trainer
    trainer = Trainer(track_grad_norm=-1)

Example::

    # track the 2-norm
    trainer = Trainer(track_grad_norm=2)

train_percent_check
^^^^^^^^^^^^^^^^^^^

How much of training dataset to check.
Useful when debugging or testing something that happens at the end of an epoch.

.. code-block::python

    # default used by the Trainer
    trainer = Trainer(train_percent_check=1.0)

Example::

    # default used by the Trainer
    trainer = Trainer(train_percent_check=1.0)

    # run through only 25% of the training set each epoch
    trainer = Trainer(train_percent_check=0.25)

truncated_bptt_steps
^^^^^^^^^^^^^^^^^^^^

Truncated back prop breaks performs backprop every k steps of
a much longer sequence.

If this is enabled, your batches will automatically get truncated
and the trainer will apply Truncated Backprop to it.

(`Williams et al. "An efficient gradient-based algorithm for on-line training of
recurrent network trajectories."
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.7941&rep=rep1&type=pdf>`_)

Example::

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
                "hiddens": hiddens  # remember to detach() this
            }

To modify how the batch is split,
override :meth:`pytorch_lightning.core.LightningModule.tbptt_split_batch`:

.. code-block:: python

        class LitMNIST(pl.LightningModule):
            def tbptt_split_batch(self, batch, split_size):
                # do your own splitting on the batch
                return splits


val_percent_check
^^^^^^^^^^^^^^^^^

How much of validation dataset to check.
Useful when debugging or testing something that happens at the end of an epoch.

Example::

    # default used by the Trainer
    trainer = Trainer(val_percent_check=1.0)

    # run through only 25% of the validation set each epoch
    trainer = Trainer(val_percent_check=0.25)

weights_save_path
^^^^^^^^^^^^^^^^^
Directory of where to save weights if specified.

.. code-block:: python

    # default used by the Trainer
    trainer = Trainer(weights_save_path=os.getcwd())

Example::

    # save to your custom path
    trainer = Trainer(weights_save_path='my/path')

    # if checkpoint callback used, then overrides the weights path
    # **NOTE: this saves weights to some/path NOT my/path
    checkpoint_callback = ModelCheckpoint(filepath='some/path')
    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        weights_save_path='my/path'
    )

weights_summary
^^^^^^^^^^^^^^^
Prints a summary of the weights when training begins.
Options: 'full', 'top', None.

Example::

    # default used by the Trainer (ie: print all weights)
    trainer = Trainer(weights_summary='full')

    # print only the top level modules
    trainer = Trainer(weights_summary='top')

    # don't print a summary
    trainer = Trainer(weights_summary=None)

Trainer class
-------------

"""

from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.trainer.seed import seed_everything

__all__ = ['Trainer', 'seed_everything']
