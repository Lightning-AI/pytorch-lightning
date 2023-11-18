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

Whenever you use multiple devices and/or nodes, your effective batch size will be 7 * devices * num_nodes.

.. code-block:: python

    # effective batch size = 7 * 8
    Trainer(accelerator="gpu", devices=8, strategy=...)

    # effective batch size = 7 * 8 * 10
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy=...)


.. note:: Huge batch sizes are actually really bad for convergence. Check out:
        `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>`_

----


*********************************************************
How do I use multiple GPUs on Jupyter or Colab notebooks?
*********************************************************

To use multiple GPUs on notebooks, use the *DDP_NOTEBOOK* mode.

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="ddp_notebook")

If you want to use other strategies, please launch your training via the command-shell.
See also: :doc:`../../common/notebooks`

----

*****************************************************
I'm getting errors related to Pickling. What do I do?
*****************************************************

Pickle is Python's mechanism for serializing and unserializing data. Some distributed modes require that your code is fully pickle compliant. If you run into an issue with pickling, try the following to figure out the issue.

.. code-block:: python

    import pickle

    model = YourModel()
    pickle.dumps(model)

For example, the `ddp_spawn` strategy has the pickling requirement. This is a limitation of Python.

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="ddp_spawn")

If you use `ddp`, your code doesn't need to be pickled:

.. code-block:: python

    Trainer(accelerator="gpu", devices=4, strategy="ddp")
