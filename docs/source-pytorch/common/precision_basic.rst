:orphan:

.. _precision_basic:

#######################
N-Bit Precision (Basic)
#######################
**Audience:** Users looking to train models faster and consume less memory.

----

If you're looking to run models faster or consume less memory, consider tweaking the precision settings of your models.

Lower precision, such as 16-bit floating-point, requires less memory and enables training and deploying larger models.
Higher precision, such as the 64-bit floating-point, can be used for highly sensitive use-cases.

----

****************
16-bit Precision
****************

Use 16-bit mixed precision to speed up training and inference.
If your GPUs are [`Tensor Core <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>`_] GPUs, you can expect a ~3x speed improvement.

.. code-block:: python

    Trainer(precision='16-mixed')


With true 16-bit precision you can additionally lower your memory consumption by up to half so that you can train and deploy larger models.
However, this setting can sometimes lead to unstable training.

.. code-block:: python

    Trainer(precision='16-true')


----


****************
32-bit Precision
****************

32-bit precision is the default used across all models and research. This precision is known to be stable in contrast to lower precision settings.

.. testcode::

    Trainer(precision="32-true")

    # or
    Trainer(precision="32")

    # or
    Trainer(precision=32)

----

****************
64-bit Precision
****************

For certain scientific computations, 64-bit precision enables more accurate models. However, doubling the precision from 32 to 64 bit also doubles the memory requirements.

.. testcode::

    Trainer(precision="64-true")

    # or
    Trainer(precision="64")

    # or
    Trainer(precision=64)

.. note::

    Since in deep learning, memory is always a bottleneck, especially when dealing with a large volume of data and with limited resources.
    It is recommended using single precision for better speed. Although you can still use it if you want for your particular use-case.

----

********************************
Precision support by accelerator
********************************

.. list-table:: Precision with Accelerators
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Precision
     - CPU
     - GPU
     - TPU
     - IPU
   * - 16 Mixed
     - No
     - Yes
     - No
     - Yes
   * - BFloat16 Mixed
     - Yes
     - Yes
     - Yes
     - No
   * - 32 True
     - Yes
     - Yes
     - Yes
     - Yes
   * - 64 True
     - Yes
     - Yes
     - No
     - No
