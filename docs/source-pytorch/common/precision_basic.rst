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

Use 16-bit precision to cut your memory consumption in half so that you can train and deploy larger models. If your GPUs are [`Tensor Core <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>`_] GPUs, you can also get a ~3x speed improvement. Half precision can sometimes lead to unstable training.

.. code::

    Trainer(precision=16)

----

****************
32-bit Precision
****************

32-bit precision is the default used across all models and research. This precision is known to be stable in contrast to lower precision settings.

.. testcode::

    Trainer(precision=32)

----

****************
64-bit Precision
****************

For certain scientific computations, 64-bit precision enables more accurate models. However, doubling the precision from 32 to 64 bit also doubles the memory requirements.

.. testcode::

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
   * - 16
     - No
     - Yes
     - No
     - Yes
   * - BFloat16
     - Yes
     - Yes
     - Yes
     - No
   * - 32
     - Yes
     - Yes
     - Yes
     - Yes
   * - 64
     - Yes
     - Yes
     - No
     - No
