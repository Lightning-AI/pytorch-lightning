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

.. _multi_gpu:

Train on GPUs
-------------

The Trainer will run on all available GPUs by default. Make sure you're running on a machine with at least one GPU.
There's no need to specify any NVIDIA flags as Lightning will do it for you.

.. code-block:: python

    # run on as many GPUs as available by default
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
    # equivalent to
    trainer = Trainer()

    # run on one GPU
    trainer = Trainer(accelerator="gpu", devices=1)
    # run on multiple GPUs
    trainer = Trainer(accelerator="gpu", devices=8)
    # choose the number of devices automatically
    trainer = Trainer(accelerator="gpu", devices="auto")

.. note::
    Setting ``accelerator="gpu"`` will also automatically choose the "mps" device on Apple sillicon GPUs.
    If you want to avoid this, you can set ``accelerator="cuda"`` instead.

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
    # equivalent to `list(range(torch.cuda.device_count())) and `"auto"`
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
| [1, 3]           | list      | [1, 3]              | GPU index 1 and 3 (0-based)     |
+------------------+-----------+---------------------+---------------------------------+
| "3"              | str       | [0, 1, 2]           | first 3 GPUs                    |
+------------------+-----------+---------------------+---------------------------------+
| "1, 3"           | str       | [1, 3]              | GPU index 1 and 3 (0-based)     |
+------------------+-----------+---------------------+---------------------------------+
| "-1"             | str       | [0, 1, 2, ...]      | all available GPUs              |
+------------------+-----------+---------------------+---------------------------------+


Find usable CUDA devices
^^^^^^^^^^^^^^^^^^^^^^^^

If you want to run several experiments at the same time on your machine, for example for a hyperparameter sweep, then you can
use the following utility function to pick GPU indices that are "accessible", without having to change your code every time.

.. code-block:: python

    from lightning.pytorch.accelerators import find_usable_cuda_devices

    # Find two GPUs on the system that are not already occupied
    trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(2))

    from lightning.fabric.accelerators import find_usable_cuda_devices

    # Works with Fabric too
    fabric = Fabric(accelerator="cuda", devices=find_usable_cuda_devices(2))


This is especially useful when GPUs are configured to be in "exclusive compute mode", such that only one process at a time is allowed access to the device.
This special mode is often enabled on server GPUs or systems shared among multiple users.
