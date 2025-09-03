:orphan:

.. _gpu_faq:

GPU training (FAQ)
==================

***************************************************************
How should I adjust the batch size when using multiple devices?
***************************************************************

Lightning automatically shards your data across multiple GPUs, meaning that each device only sees a unique subset of your
data, but the `batch_size` in your DataLoader remains the same. This means that the effective batch size e.g. the
total number of samples processed in one forward/backward pass is

.. math::

    \text{Effective Batch Size} = \text{DataLoader Batch Size} \times \text{Number of Devices} \times \text{Number of Nodes}

A couple of examples to illustrate this:

.. code-block:: python

    dataloader = DataLoader(..., batch_size=7)

    # Single GPU: effective batch size = 7
    Trainer(accelerator="gpu", devices=1)

    # Multi-GPU: effective batch size = 7 * 8 = 56
    Trainer(accelerator="gpu", devices=8, strategy=...)

    # Multi-node: effective batch size = 7 * 8 * 10 = 560
    Trainer(accelerator="gpu", devices=8, num_nodes=10, strategy=...)

In general you should be able to use the same `batch_size` in your DataLoader regardless of the number of devices you are
using.

.. note::

    If you want distributed training to work exactly the same as single GPU training, you need to set the `batch_size`
    in your DataLoader to `original_batch_size / num_devices` to maintain the same effective batch size. However, this
    can lead to poor GPU utilization.

----

******************************************************************
How should I adjust the learning rate when using multiple devices?
******************************************************************

Because the effective batch size is larger when using multiple devices, you need to adjust your learning rate
accordingly. Because the learning rate is a hyperparameter that controls how much to change the model in response to
the estimated error each time the model weights are updated, it is important to scale it with the effective batch size.

In general, there are two common scaling rules:

1. **Linear scaling**: Increase the learning rate linearly with the number of devices.

    .. code-block:: python

        # Example: Linear scaling
        base_lr = 1e-3
        num_devices = 8
        scaled_lr = base_lr * num_devices  # 8e-3

2. **Square root scaling**: Increase the learning rate by the square root of the number of devices.

    .. code-block:: python

        # Example: Square root scaling
        base_lr = 1e-3
        num_devices = 8
        scaled_lr = base_lr * (num_devices ** 0.5)  # 2.83e-3

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
