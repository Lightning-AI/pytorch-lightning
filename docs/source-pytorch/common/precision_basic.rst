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

    Trainer(precision="16-mixed")


In most cases, mixed precision uses FP16. Supported `PyTorch operations <https://pytorch.org/docs/stable/amp.html#op-specific-behavior>`__ automatically run in FP16, saving memory and improving throughput on the supported accelerators.
Since computation happens in FP16, which has a very limited "dynamic range", there is a chance of numerical instability during training. This is handled internally by a dynamic grad scaler which skips invalid steps and adjusts the scaler to ensure subsequent steps fall within a finite range. For more information `see the autocast docs <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`__.


With true 16-bit precision you can additionally lower your memory consumption by up to half so that you can train and deploy larger models.
However, this setting can sometimes lead to unstable training.

.. code-block:: python

    Trainer(precision="16-true")


----


****************
32-bit Precision
****************

32-bit precision is the default used across all models and research. This precision is known to be stable in contrast to lower precision settings.

.. testcode::

    Trainer(precision="32-true")

    # or (legacy)
    Trainer(precision="32")

    # or (legacy)
    Trainer(precision=32)

----

****************
64-bit Precision
****************

For certain scientific computations, 64-bit precision enables more accurate models. However, doubling the precision from 32 to 64 bit also doubles the memory requirements.

.. testcode::

    Trainer(precision="64-true")

    # or (legacy)
    Trainer(precision="64")

    # or (legacy)
    Trainer(precision=64)

Since in deep learning, memory is always a bottleneck, especially when dealing with a large volume of data and with limited resources.
It is recommended using single precision for better speed. Although you can still use it if you want for your particular use-case.

When working with complex numbers, instantiation of complex tensors should be done in the
:meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` hook or under the
:meth:`~lightning.pytorch.trainer.trainer.Trainer.init_module` context manager so that the `complex128` dtype
is properly selected.

.. code-block:: python

    trainer = Trainer(precision="64-true")

    # init the model directly on the device and with parameters in full-precision
    with trainer.init_module():
        model = MyModel()

    trainer.fit(model)


----

********************************
Precision support by accelerator
********************************

.. list-table:: Precision with Accelerators
   :widths: 20 20 20 20
   :header-rows: 1

   * - Precision
     - CPU
     - GPU
     - TPU
   * - 16 Mixed
     - No
     - Yes
     - No
   * - BFloat16 Mixed
     - Yes
     - Yes
     - Yes
   * - 32 True
     - Yes
     - Yes
     - Yes
   * - 64 True
     - Yes
     - Yes
     - No
