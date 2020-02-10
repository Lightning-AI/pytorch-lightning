Checkpointing
==============

Checkpoint callback
-------------------
This is a callback

Model saving
-------------------

Model loading
-------------------

Restoring training session
-------------------


Computing cluster (SLURM)
==========================

Walltime auto-resubmit
-------------------

Debugging
==========

Fast dev run
-------------------

Inspect gradient norms
-------------------

Log GPU usage
-------------------

Make model overfit on subset of data
-------------------

Print the parameter count by layer
-------------------

Print which gradients are nan
-------------------

Print input and output size of every module in system
-------------------

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



Checkpoint Saving/Loading
==============================

Multi-gpu (same node) training
==============================

Multi-node training
====================

16-bit precision
=================

gradient clipping
=================

modifying training via hooks
=============================

.. toctree::
   :maxdepth: 3

   pl_examples


profiling a training run
========================
.. toctree::
    :maxdepth: 1

    profiler