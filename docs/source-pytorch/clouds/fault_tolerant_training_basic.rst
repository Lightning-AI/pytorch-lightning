:orphan:

###############################
Fault-tolerant Training (basic)
###############################
**Audience:** User who want to run on the cloud or a cluster environment.

**Pre-requisites**: Users must have first read :doc:`Run on the cloud (basic) <run_basic>`

----

********************************
What is fault-tolerant training?
********************************
When developing models on the cloud or cluster environments, you may be forced to restart from scratch in the event of a software or hardware failure (ie: a *fault*). Lightning models can run fault-proof.

With Fault Tolerant Training, when ``Trainer.fit()`` fails in the middle of an epoch during training or validation,
Lightning will restart exactly where it failed, and everything will be restored (down to the batch it was on even if the dataset was shuffled).

.. warning:: Fault-tolerant Training is currently an experimental feature within Lightning.

----

***************************************************
Use fault-tolerance to save money on cloud training
***************************************************
Cloud providers offer pre-emptible machines which can be priced as low as 1/10th the cost but can be shut-down automatically at any time.
Because fault-tolerant training can automatically recover from an interruption, you can train models for many weeks/months at a time for the pre-emptible prices.

To easily run on the cloud with fault-tolerance with lightning-grid, use the following arguments:

.. code-block:: bash

    grid run --use_spot --auto_resume lightning_script.py

The ``--use_spot`` argument enables cheap preemptible pricing (but the machines that can be interrupted).
If the machine is interrupted, the ``--auto_resume`` argument automatically restarts the machine.

As long as you are running a script that runs a lightning model, the model will restore itself and handle all the details of fault tolerance.

----

.. include:: grid_costs.rst
