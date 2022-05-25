:orphan:

.. _mps_basic:

MPS training (basic)
====================
**Audience:** Users looking to train on their Apple silicon GPUs.

.. warning::

   The MPS accelerator as well as the PyTorch backend are still very experimental.
   So far not all operations are supported, but more ops are coming every day due to development from the PyTorch Team.
   You can use ``PYTORCH_ENABLE_MPS_FALLBACK=1 python your_script.py`` to fall back to cpu for unsupported operations.


----

What is Apple silicon?
----------------------
Apple silicon chips are a unified system on a chip (SoC) developed by Apple based on the ARM design.
Among other things, they feature a CPU-cores, GPU-cores, a neural engine and shared memory between all of those.

----

So it's a CPU?
--------------
Among other things yes, it includes CPU-cores. However, when running on the ``CPUAccelerator``, not the full potential of hardware acceleration the M-Socs are capable of, is used because they also feature a GPU and a neural engine.

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
MPS is short for Metal Performance Shaders which is the technology used in the back for gpu communication and computing.
