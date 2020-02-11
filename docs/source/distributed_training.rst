Distributed training
=====================

Implement Your Own Distributed (DDP) training
----------------------------------------------
If you need your own way to init PyTorch DDP you can override :meth:`pytorch_lightning.core.LightningModule.init_ddp_connection`.

If you also need to use your own DDP implementation, override:  :meth:`pytorch_lightning.core.LightningModule.configure_ddp`.

Multi-GPU
----------------------------------------------
To train on multiple GPUs make sure you are running lightning on a machine with GPUs. Lightning handles
all the NVIDIA flags for you, there's no need to set them yourself.

There are three options for multi-GPU training:

1. DataParallel (dp) - Splits a batch across GPUs on a single machine.

2. DistributedDataParallel (ddp) - Splits data across each GPU and only syncs gradients.

3. ddp2 - Acts like dp on a single machine but syncs gradients across machines like ddp.

.. code-block:: python

    # train on 1 GPU (using dp mode)
    trainer = pl.Trainer(gpus=1, distributed_backend='dp')

    # train on 2 GPUs (using dp mode)
    trainer = pl.Trainer(gpus=2, distributed_backend='dp')

    # train on 2 GPUs (using ddp mode)
    trainer = pl.Trainer(gpus=2, distributed_backend='ddp')

    # train on 0 GPUs
    trainer = pl.Trainer()


Multi-node
----------------------------------------------
See :ref:`multi-node`

Single GPU
----------------------------------------------
Make sure you are running on a machine that has at least one GPU. Lightning handles all the NVIDIA flags for you,
there's no need to set them yourself.

.. code-block:: python

    # train on 1 GPU (using dp mode)
    trainer = pl.Trainer(gpus=1)