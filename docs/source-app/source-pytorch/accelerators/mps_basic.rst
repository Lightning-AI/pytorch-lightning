:orphan:

.. _mps_basic:

MPS training (basic)
====================
**Audience:** Users looking to train on their Apple silicon GPUs.

.. warning::

   Both the MPS accelerator and the PyTorch backend are still experimental.
   As such, not all operations are currently supported. However, with ongoing development from the PyTorch team, an increasingly large number of operations are becoming available.
   You can use ``PYTORCH_ENABLE_MPS_FALLBACK=1 python your_script.py`` to fall back to cpu for unsupported operations.


----

What is Apple silicon?
----------------------
Apple silicon chips are a unified system on a chip (SoC) developed by Apple based on the ARM design.
Among other things, they feature CPU-cores, GPU-cores, a neural engine and shared memory between all of these features.

----

So it's a CPU?
--------------
Apple silicon includes CPU-cores among several other features. However, the full potential for the hardware acceleration of which the M-Socs are capable is unavailable when running on the ``CPUAccelerator``. This is because they also feature a GPU and a neural engine.

To use them, Lightning supports the ``MPSAccelerator``.

----

Run on Apple silicon gpus
-------------------------
Enable the following Trainer arguments to run on Apple silicon gpus (MPS devices).

.. code::

   trainer = Trainer(accelerator="mps", devices=1)

.. note::
   The ``MPSAccelerator`` only supports 1 device at a time. Currently there are no machines with multiple MPS-capable GPUs.

----

What does MPS stand for?
------------------------
MPS is short for `Metal Performance Shaders <https://developer.apple.com/metal/>`_  which is the technology used in the back for gpu communication and computing.
