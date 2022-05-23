:orphan:

.. _mps_basic:

MPS training (basic)
=======================
**Audience:** Users looking to train on their M-SoC GPUs.

----

What is a M-SoC?
----------------
M-SoCs are a unified system of a cirquit (SoC) developed by Apple based on the ARM design.
Among other things, they feature a CPU-cores, GPU-cores a neural engine and shared memory between all of those.

----

So it's a CPU?
--------------
Among other things yes, it includes CPU-cores. However, when running on the ``CPUAccelerator``, not the full potential of hardware acceleration the M-Socs are capable of, is used because they also feature a GPU and a neural engine.

To use them, Lightning supports the ``MPSAccelerator``.

----

Run on MPS
----------
Enable the following Trainer arguments to run on MPS devices.

.. code::

   trainer = Trainer(accelerator="mps", devices=1)

