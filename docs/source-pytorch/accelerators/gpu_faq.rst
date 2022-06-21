:orphan:

.. _gpu_faq:

GPU training (FAQ)
==================

******************************************************************
How should I adjust the learning rate when using multiple devices?
******************************************************************

When using distributed training make sure to modify your learning rate according to your effective
batch size.

Let's say you have a batch size of 7 in your dataloader.

.. testcode::

    class LitModel(LightningModule):
        def train_dataloader(self):
            return Dataset(..., batch_size=7)

In DDP, DDP_SPAWN, Deepspeed, DDP_SHARDED, or Horovod your effective batch size will be 7 * devices * num_nodes.

.. code-block:: python

    # effective batch size = 7 * 8
    Trainer(accelerator="gpu", devices=8, strategy="ddp")
    Trainer(accelerator="gpu", devices=8, strategy="ddp_spawn")
    Trainer(accelerator="gpu", devices=8, strategy="ddp_sharded")
    Trainer(accelerator="gpu", devices=8, strategy="horovod")

    # effective batch size = 7 * 8 * 10
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="ddp")
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="ddp_spawn")
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="ddp_sharded")
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="horovod")

In DDP2 or DP, your effective batch size will be 7 * num_nodes.
The reason is that the full batch is visible to all GPUs on the node when using DDP2.

.. code-block:: python

    # effective batch size = 7
    Trainer(accelerator="gpu", devices=8, strategy="ddp2")
    Trainer(accelerator="gpu", devices=8, strategy="dp")

    # effective batch size = 7 * 10
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy="ddp2")
    Trainer(accelerator="gpu", devices=8, strategy="dp")


.. note:: Huge batch sizes are actually really bad for convergence. Check out:
        `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>`_

----

*********************************************************
How do I use multiple GPUs on Jupyter or Colab notebooks?
*********************************************************

To use multiple GPUs on notebooks, use the *DP* mode.

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="dp")

If you want to use other models, please launch your training via the command-shell.

.. note:: Learn how to :ref:`access a cloud machine with multiple GPUs <grid_cloud_session_basic>` in this guide.

----

*****************************************************
I'm getting errors related to Pickling. What do I do?
*****************************************************

Pickle is Python's mechanism for serializing and unserializing data. A majority of distributed modes require that your code is fully pickle compliant. If you run into an issue with pickling try the following to figure out the issue

.. code-block:: python

    import pickle

    model = YourModel()
    pickle.dumps(model)

If you `ddp` your code doesn't need to be pickled.

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="ddp")

If you use `ddp_spawn` the pickling requirement remains. This is a limitation of Python.

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="ddp_spawn")
