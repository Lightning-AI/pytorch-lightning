.. testsetup:: *

    from pytorch_lightning import Trainer


.. _amp:


#########
Precision
#########

There are numerous benefits to using numerical formats with lower precision than the 32-bit floating-point or higher precision such as 64-bit floating-point.

Lower precision, such as the 16-bit floating-point, requires less memory, enabling the training and deployment of larger neural networks. Second, they require less memory bandwidth, thereby speeding up data transfer operations. Third, math operations run much faster in reduced precision, especially on GPUs with Tensor Core support for that precision.

Higher precision, such as the 64-bit floating-point, offers <TODO add something>

Following are the precisions available in Lightning along with their supported Accelerator:

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


***************
Mixed Precision
***************

Most deep learning frameworks, including PyTorch, train with 32-bit floating-point (FP32) arithmetic by default. However, this is not essential for many deep learning models to achieve complete accuracy. Mixed precision training offers significant computational speedup
by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of Tensor Cores in the Volta and Turing architectures, significant
training speedups are experienced by switching to mixed precision. It combines both FP32 and lower bit floating-points (such as FP16) to reduce memory footprint during model
training/evaluation, resulting in improved performance. It does so by identifying the steps that require full precision and using a 32-bit floating-point for only those steps while using a 16-bit floating-point everywhere else. Mixed precision training achieves all these benefits
while ensuring that no task-specific accuracy is lost compared to full precision training.

.. note::

    In some cases, it is essential to remain in FP32 for numerical stability, so keep this in mind when using mixed precision.
    For example, when running scatter operations during the forward (such as torchpoint3d), computation must remain in FP32.


FP16 Mixed Precision
====================

In most cases, mixed precision uses FP16. Supported `PyTorch operations <https://pytorch.org/docs/stable/amp.html#op-specific-behavior>`__ automatically run in FP16, saving memory and improving throughput on the supported accelerators.


.. note::

    When using TPUs, setting ``precision=16`` will enable bfloat16, the only supported precision type on TPUs.

.. testcode::
    :skipif: not torch.cuda.is_available()

    Trainer(gpus=1, precision=16)


PyTorch Native
--------------

PyTorch 1.6 release introduced mixed precision functionality into their core as the AMP package, `torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`__. It is more flexible and intuitive compared to `NVIDIA APEX <https://github.com/NVIDIA/apex>`__.
Since computation happens in FP16, there is a chance of numerical instability during training. This is handled internally by a dynamic grad scaler which skips invalid steps and adjusts the scaler to ensure subsequent steps fall within a finite range. For more information `see the autocast docs <https://pytorch.org/docs/stable/amp.html#gradient-scaling>`__.
Lightning uses native amp by default with ``precision=16|"bf16"``. You can also set it using:

.. testcode::

    Trainer(precision=16, amp_backend="native")


NVIDIA APEX
-----------

.. warning::

    We strongly recommend using the above native mixed precision rather than NVIDIA APEX unless you require more refined control.

`NVIDIA APEX <https://github.com/NVIDIA/apex>`__ offers additional flexibility in setting mixed precision. This can be useful when trying out different precision configurations, such as keeping most of your weights in FP16 and running computation in FP16.

.. testcode::
    :skipif: not _APEX_AVAILABLE or not torch.cuda.is_available()

    Trainer(gpus=1, amp_backend="apex")

Set the `NVIDIA optimization level <https://nvidia.github.io/apex/amp.html#opt-levels>`__ via the trainer.

.. testcode::
    :skipif: not _APEX_AVAILABLE or not torch.cuda.is_available()

    Trainer(gpus=1, amp_backend="apex", amp_level="O2")


BFloat16 Mixed Precision
========================

.. warning::

    BFloat16 requires PyTorch 1.10 or later and is only supported with PyTorch Native AMP.

    BFloat16 is also experimental and may not provide significant speedups or memory improvements, offering better numerical stability.

    Do note for GPUs, the most significant benefits require `Ampere <https://en.wikipedia.org/wiki/Ampere_(microarchitecture)>`__ based GPUs, such as A100s or 3090s.

BFloat16 Mixed precision is similar to FP16 mixed precision, however, it maintains more of the "dynamic range" that FP32 offers. This means it is able to improve numerical stability than FP16 mixed precision. For more information, see `this TPU performance blogpost <https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus>`__.

Under the hood, we use `torch.autocast <https://pytorch.org/docs/stable/amp.html>`__ with the dtype set to ``bfloat16``, with no gradient scaling.

.. testcode::
    :skipif: not _TORCH_GREATER_EQUAL_1_10 or not torch.cuda.is_available()

    Trainer(gpus=1, precision="bf16")

It is also possible to use BFloat16 mixed precision on the CPU, relying on MKLDNN under the hood.

.. testcode::
    :skipif: not _TORCH_GREATER_EQUAL_1_10

    Trainer(precision="bf16")


****************
Single Precision
****************

PyTorch models train with 32-bit floating-point (FP32) arithmetic by default.
Lightning uses 32-bit by default. You can also set it using:

.. testcode::

    Trainer(precision=32)


****************
Double Precision
****************

<TODO add more stuff here>
Lightning supports training models with double precision/64-bit. You can also set it using:

.. testcode::

    Trainer(precision=64)

.. note::

    Since in deep learning, memory is always a bottleneck, especially when dealing with a large volume of data and using a GPU with limited memory.
    It is recommended using single precision for better speed with the price of a possible negligible loss of precision.


*****************
Precision Plugins
*****************

You can also customize and pass your own Precision Plugin by subclassing the :class:`~pytorch_lightning.plugins.precision.precision_plugin.PrecisionPlugin` class.

- Perform pre and post backward/optimizer step operations such as scaling gradients.
- Provide context managers for forward, training_step, etc.
- Configure Gradient clipping.

.. code-block:: python

    class CustomPrecisionPlugin(PrecisionPlugin):
        precision = 16

        ...


    trainer = Trainer(plugins=[CustomPrecisionPlugin()])
