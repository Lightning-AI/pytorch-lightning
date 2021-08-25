.. testsetup:: *

    from pytorch_lightning import Trainer


.. _amp:

Mixed Precision Training
========================

Mixed precision combines the use of both FP32 and lower bit floating points (such as FP16) to reduce memory footprint during model training, resulting in improved performance.

Lightning offers mixed precision training for GPUs, CPUs and TPUs.

.. note::

    In some cases it is important to remain in FP32 for numerical stability, so keep this in mind when using mixed precision.

FP16 Mixed Precision
--------------------

.. testcode::
    :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

    Trainer(gpus=1, precision=16)

BFloat16 Mixed Precision
------------------------

BFloat16 Mixed precision is similar to FP16 mixed precision, however we maintain more of the "dynamic range" that FP32 has to offer. This means we are able to improve numerical stability, compared to FP16 mixed precision. For more information see `this TPU performance blog post <https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus>`__.

Since BFloat16 is more stable than FP16 during training, we do not need to worry about any gradient scaling or nan gradient values that comes with using FP16 mixed precision.

.. note::

    BFloat16 requires PyTorch 1.10 or later.

.. testcode::
    :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

    Trainer(gpus=1, precision="bf16")

It is also possible to use BFloat16 Mixed Precision on the CPU, relying on MKLDNN under the hood.

.. testcode::

    Trainer(precision="bf16")
