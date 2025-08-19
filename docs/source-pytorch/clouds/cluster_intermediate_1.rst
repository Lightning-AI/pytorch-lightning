:orphan:

########################################
Run on an on-prem cluster (intermediate)
########################################
**Audience**: Users who need to run on an academic or enterprise private cluster.


----


.. _non-slurm:

******************
Set up the cluster
******************
This guide shows how to run a training job on a general purpose cluster. We recommend beginners to try this method
first because it requires the least amount of configuration and changes to the code.
To setup a multi-node computing cluster you need:

1) Multiple computers with PyTorch Lightning installed
2) A network connectivity between them with firewall rules that allow traffic flow on a specified *MASTER_PORT*.
3) Defined environment variables on each node required for the PyTorch Lightning multi-node distributed training

PyTorch Lightning follows the design of `PyTorch distributed communication package <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`_. and requires the following environment variables to be defined on each node:

- *MASTER_PORT* - required; has to be a free port on machine with NODE_RANK 0
- *MASTER_ADDR* - required (except for NODE_RANK 0); address of NODE_RANK 0 node
- *WORLD_SIZE* - required; the total number of GPUs/processes that you will use
- *NODE_RANK* - required; id of the node in the cluster

.. _training_script_setup:


----


**************************
Set up the training script
**************************
To train a model using multiple nodes, do the following:

1.  Design your :ref:`lightning_module` (no need to add anything specific here).

2.  Enable DDP in the trainer

    .. code-block:: python

       # train on 32 GPUs across 4 nodes
       trainer = Trainer(accelerator="gpu", devices=8, num_nodes=4, strategy="ddp")


----


***************************
Submit a job to the cluster
***************************
To submit a training job to the cluster you need to run the same training script on each node of the cluster.
This means that you need to:

1. Copy all third-party libraries to each node (usually means - distribute requirements.txt file and install it).
2. Copy all your import dependencies and the script itself to each node.
3. Run the script on each node.


----


******************
Debug on a cluster
******************
When running in DDP mode, some errors in your code can show up as an NCCL issue.
Set the ``NCCL_DEBUG=INFO`` environment variable to see the ACTUAL error.

.. code-block:: bash

    NCCL_DEBUG=INFO python train.py ...
