
.. _non-slurm:

Computing cluster
=================

With Lightning it is easy to run your training script on a computing cluster without almost any modifications to the script.
This guide shows how to run a training job on a general purpose cluster.

Also, check :doc:`../extensions/accelerators` as a new and more general approach to a cluster setup.

--------


Cluster setup
-------------

To setup a multi-node computing cluster you need:

1) Multiple computers with PyTorch Lightning installed
2) A network connectivity between them with firewall rules that allow traffic flow on a specified *MASTER_PORT*.
3) Defined environment variables on each node required for the PyTorch Lightning multi-node distributed training

PyTorch Lightning follows the design of `PyTorch distributed communication package <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`_. and requires the following environment variables to be defined on each node:

- *MASTER_PORT* - required; has to be a free port on machine with NODE_RANK 0
- *MASTER_ADDR* - required (except for NODE_RANK 0); address of NODE_RANK 0 node
- *WORLD_SIZE* - required; how many nodes are in the cluster
- *NODE_RANK* - required; id of the node in the cluster


Training script design
----------------------

To train a model using multiple nodes, do the following:

1.  Design your :ref:`lightning_module` (no need to add anything specific here).

2.  Enable DDP in the trainer

    .. code-block:: python

       # train on 32 GPUs across 4 nodes
       trainer = Trainer(gpus=8, num_nodes=4, accelerator='ddp')


Submit a job to the cluster
---------------------------

To submit a training job to the cluster you need to run the same training script on each node of the cluster.
This means that you need to:

1. Copy all third-party libraries to each node (usually means - distribute requirements.txt file and install it).

2. Copy all your import dependencies and the script itself to each node.

3. Run the script on each node.
