Multi-GPU training
=====================

Lightning supports multiple ways of doing distributed training.

Data Parallel (dp)
-------------------
`DataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel>`_ splits a batch across k GPUs. That is, if you have a batch of 32 and use dp with 2 gpus,
each GPU will process 16 samples, after which the root node will aggregate the results.

.. code-block:: python

    # train on 1 GPU (using dp mode)
    trainer = pl.Trainer(gpus=2, distributed_backend='dp')

Distributed Data Parallel
---------------------------
`DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#distributeddataparallel>`_ works as follows.

1. Each GPU across every node gets its own process.

2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.

3. Each process inits the model.

.. note:: Make sure  to set the random seed so that each model inits  with the same weights

4. Each process performs a full forward and backward pass in parallel.

5. The gradients are synced and averaged across all processes.

6. Each process updates its optimizer.

.. code-block:: python

    # train on 8 GPUs (same machine (ie: node))
    trainer = pl.Trainer(gpus=8, distributed_backend='ddp')

    # train on 32 GPUs (4 nodes)
    trainer = pl.Trainer(gpus=8, distributed_backend='ddp', num_nodes=4)

Distributed Data Parallel 2
-----------------------------
In certain cases, it's advantageous to use all batches on the same machine instead of a subset.
For instance you might want to compute a NCE loss where it pays  to have more negative samples.

In  this case, we can use ddp2 which behaves like dp in a machine and ddp across nodes. DDP2 does the following:

1. Copies a subset of the  data to each node.

2. Inits a model on each node.

3. Runs a forward and backward pass using DP.

4. Syncs gradients across nodes.

5. Applies the optimizer updates.

.. code-block:: python

    # train on 32 GPUs (4 nodes)
    trainer = pl.Trainer(gpus=8, distributed_backend='ddp2', num_nodes=4)


Implement Your Own Distributed (DDP) training
----------------------------------------------
If you need your own way to init PyTorch DDP you can override :meth:`pytorch_lightning.core.LightningModule.init_ddp_connection`.

If you also need to use your own DDP implementation, override:  :meth:`pytorch_lightning.core.LightningModule.configure_ddp`.
