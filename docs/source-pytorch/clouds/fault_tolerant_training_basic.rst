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
