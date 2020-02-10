Checkpointing
==============

.. _model-saving:

Model saving
-------------------
To save a LightningModule, provide a :meth:`pytorch_lightning.callbacks.ModelCheckpoint` callback.

The Lightning checkpoint also saves the hparams (hyperparams) passed into the LightningModule init.

.. note:: hparams is a `Namespace <https://docs.python.org/2/library/argparse.html#argparse.Namespace>`_ or dictionary.

.. code-block:: python
   :emphasize-lines: 8

   from argparse import Namespace

   # usually these come from command line args
   args = Namespace(**{'learning_rate':0.001})

   # define you module to have hparams as the first arg
   # this means your checkpoint will have everything that went into making
   # this model (in this case, learning rate)
   class MyLightningModule(pl.LightningModule):

       def __init__(self, hparams, ...):
           self.hparams = hparams

   my_model = MyLightningModule(args)

   # auto-saves checkpoint
   checkpoint_callback = ModelCheckpoint(filepath='my_path')
   Trainer(checkpoint_callback=checkpoint_callback)


Model loading
-----------------------------------

To load a model, use :meth:`pytorch_lightning.core.LightningModule.load_from_checkpoint`

.. note:: If lightning created your checkpoint, your model will receive all the hyperparameters used
   to create the checkpoint. (See: :ref:`model-saving`).

.. code-block:: python

    # load weights without mapping
    MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

    # load weights mapping all weights from GPU 1 to GPU 0
    map_location = {'cuda:1':'cuda:0'}
    MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt', map_location=map_location)

Restoring training session
-----------------------------------

If you want to pick up training from where you left off, you have a few options.

1. Pass in a logger with the same experiment version to continue training.

.. code-block:: python

   # train the first time and set the version number
   logger = TensorboardLogger(version=10)
   trainer = Trainer(logger=logger)
   trainer.fit(model)

   # when you init another logger with that same version, the model
   # will continue where it left off
   logger = TensorboardLogger(version=10)
   trainer = Trainer(logger=logger)
   trainer.fit(model)

2. A second option is to pass in a path to a checkpoint (see: :ref:`pytorch_lightning.trainer`).

.. code-block:: python

   # train the first time and set the version number
   trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
   trainer.fit(model)


Computing cluster (SLURM)
==========================

Lightning automates job the details behind  training on a SLURM powered cluster.

Multi-node training
--------------------
To train a model using multiple-nodes do the following:

1. Design your LightningModule.

2. Add `torch.DistributedSampler <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`_
   which enables access to a subset of your full dataset to each GPU.

3. Enable ddp in the trainer

.. code-block:: python

   # train on 32 GPUs across 4 nodes
   trainer = Trainer(gpus=8, num_nodes=4, distributed_backend='ddp')

4. It's a good idea to structure your train.py file like this:

.. code-block:: python

    # train.py
    def main(hparams):
        model = LightningTemplateModel(hparams)

        trainer = pl.Trainer(
            gpus=8,
            num_nodes=4,
            distributed_backend='ddp'
        )

        trainer.fit(model)


    if __name__ == '__main__':
        root_dir = os.path.dirname(os.path.realpath(__file__))
        parent_parser = ArgumentParser(add_help=False)
        hyperparams = parser.parse_args()

       # TRAIN
        main(hyperparams)

4. Submit the appropriate SLURM job

.. code-block:: bash

    #!/bin/bash -l

    # SLURM SUBMIT SCRIPT
    #SBATCH --nodes=4
    #SBATCH --gres=gpu:8
    #SBATCH --ntasks-per-node=8
    #SBATCH --mem=0
    #SBATCH --time=0-02:00:00

    # activate conda env
    source activate $1

    # -------------------------
    # debugging flags (optional)
     export NCCL_DEBUG=INFO
     export PYTHONFAULTHANDLER=1

    # on your cluster you might need these:
    # set the network interface
    # export NCCL_SOCKET_IFNAME=^docker0,lo

    # might need the latest cuda
    # module load NCCL/2.4.7-1-cuda.10.0
    # -------------------------

    # run script from above
    srun python3 train.py


Walltime auto-resubmit
-----------------------------------
When you use Lightning in a SLURM cluster, lightning automatically detects when it is about
to run into the walltime, and it does the following:

1. Saves a temporary checkpoint.
2. Requeues the job.
3. When the job starts, it loads the temporary checkpoint.

.. note:: To get this behavior you have to do nothing.

Debugging
==========
The following are flags that make debugging much easier.

Fast dev run
-------------------
This flag runs a "unit test" by running 1 training batch and 1 validation batch.
The point is to detect any bugs in the training/validation loop without having to wait for
a full epoch to crash.

.. code-block:: python

    trainer = pl.Trainer(fast_dev_run=True)

Inspect gradient norms
-----------------------------------
Logs (to a logger), the norm of each weight matrix.

.. code-block:: python

    # the 2-norm
    trainer = pl.Trainer(track_grad_norm=2)

Log GPU usage
-----------------------------------
Logs (to a logger) the GPU usage for each GPU on the master machine.

(See: :ref:`trainer`)

.. code-block:: python

    trainer = pl.Trainer(log_gpu_memory=True)

Make model overfit on subset of data
-----------------------------------

A good debugging technique is to take a tiny portion of your data (say 2 samples per class),
and try to get your model to overfit. If it can't, it's a sign it won't work with large datasets.

(See: :ref:`trainer`)

.. code-block:: python

    trainer = pl.Trainer(overfit_pct=0.01)

Print the parameter count by layer
-----------------------------------
Whenever the .fit() function gets called, the Trainer will print the weights summary for the lightningModule.
To disable this behavior, turn off this flag:

(See: :ref:`trainer.weights_summary`)

.. code-block:: python

    trainer = pl.Trainer(weights_summary=None)

Print which gradients are nan
------------------------------
Prints the tensors with nan gradients.

(See: :ref:`trainer.print_nan_grads`)

.. code-block:: python

    trainer = pl.Trainer(print_nan_grads=False)

Distributed training
=====================

Implement Your Own Distributed (DDP) training
----------------------------------------------

16-bit mixed precision
----------------------------------------------

Multi-GPU
----------------------------------------------

Multi-node
----------------------------------------------

Single GPU
----------------------------------------------

Experiment Logging
====================

Display metrics in progress bar
----------------------------------------------

Log metric row every k batches
----------------------------------------------

Tensorboard support
----------------------------------------------

Test Tube support
----------------------------------------------

Comet.ml support
----------------------------------------------

Neptune support
----------------------------------------------

Wandb support
----------------------------------------------

Save a snapshot of all hyperparameters
----------------------------------------------

Snapshot code for a training run
----------------------------------------------

Write logs file to csv every k batches
----------------------------------------------

Training loop
===============
Accumulate gradients
-------------------------------------

Force training for min or max epochs
-------------------------------------

Early stopping callback
-------------------------------------

Force disable early stop
-------------------------------------

Gradient Clipping
-------------------------------------

Hooks
-------------------------------------

Learning rate scheduling
-------------------------------------

Use multiple optimizers (like GANs)
-------------------------------------

Set how much of the training set to check (1-100%)
-------------------------------------

Step optimizers at arbitrary intervals
-------------------------------------

Validation loop
================

Check validation every n epochs
-------------------------------------

Hooks
-------------------------------------

Set how much of the validation set to check
-------------------------------------

Set how much of the test set to check
-------------------------------------

Set validation check frequency within 1 training epoch
-------------------------------------

Set the number of validation sanity steps
-------------------------------------

Testing loop
=============

Run test set
-------------------------------------

Examples
=============================

.. toctree::
   :maxdepth: 3

   pl_examples


profiling a training run
========================
.. toctree::
    :maxdepth: 1

    profiler