:orphan:

.. _gpu_basic:

GPU training (Basic)
====================
**Audience:** Users looking to save money and run large models faster using single or multiple

----

What is a GPU?
--------------
A Graphics Processing Unit (GPU), is a specialized hardware accelerator designed to speed up mathematical computations used in gaming and deep learning.

----

Train on 1 GPU
--------------

Make sure you're running on a machine with at least one GPU. There's no need to specify any NVIDIA flags
as Lightning will do it for you.

.. testcode::
    :skipif: torch.cuda.device_count() < 1

    trainer = Trainer(accelerator="gpu", devices=1)

----------------


.. _multi_gpu:

Train on multiple GPUs
----------------------

To use multiple GPUs, set the number of devices in the Trainer or the index of the GPUs.

.. code::

    trainer = Trainer(accelerator="gpu", devices=4)

Choosing GPU devices
^^^^^^^^^^^^^^^^^^^^

You can select the GPU devices using ranges, a list of indices or a string containing
a comma separated list of GPU ids:

.. testsetup::

    k = 1

.. testcode::
    :skipif: torch.cuda.device_count() < 2

    # DEFAULT (int) specifies how many GPUs to use per node
    Trainer(accelerator="gpu", devices=k)

    # Above is equivalent to
    Trainer(accelerator="gpu", devices=list(range(k)))

    # Specify which GPUs to use (don't use when running on cluster)
    Trainer(accelerator="gpu", devices=[0, 1])

    # Equivalent using a string
    Trainer(accelerator="gpu", devices="0, 1")

    # To use all available GPUs put -1 or '-1'
    # equivalent to list(range(torch.cuda.device_count()))
    Trainer(accelerator="gpu", devices=-1)

The table below lists examples of possible input formats and how they are interpreted by Lightning.

+------------------+-----------+---------------------+---------------------------------+
| `devices`        | Type      | Parsed              | Meaning                         |
+==================+===========+=====================+=================================+
| 3                | int       | [0, 1, 2]           | first 3 GPUs                    |
+------------------+-----------+---------------------+---------------------------------+
| -1               | int       | [0, 1, 2, ...]      | all available GPUs              |
+------------------+-----------+---------------------+---------------------------------+
| [0]              | list      | [0]                 | GPU 0                           |
+------------------+-----------+---------------------+---------------------------------+
| [1, 3]           | list      | [1, 3]              | GPUs 1 and 3                    |
+------------------+-----------+---------------------+---------------------------------+
| "3"              | str       | [0, 1, 2]           | first 3 GPUs                    |
+------------------+-----------+---------------------+---------------------------------+
| "1, 3"           | str       | [1, 3]              | GPUs 1 and 3                    |
+------------------+-----------+---------------------+---------------------------------+
| "-1"             | str       | [0, 1, 2, ...]      | all available GPUs              |
+------------------+-----------+---------------------+---------------------------------+

.. note::

    When specifying number of ``devices`` as an integer ``devices=k``, setting the trainer flag
    ``auto_select_gpus=True`` will automatically help you find ``k`` GPUs that are not
    occupied by other processes. This is especially useful when GPUs are configured
    to be in "exclusive mode", such that only one process at a time can access them.
    For more details see the :doc:`trainer guide <../common/trainer>`.
