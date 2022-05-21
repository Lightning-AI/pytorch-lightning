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

Run on multiple MPS
-------------------
From a deep learning perspective, every Apple Machine only has 1 MPS device (no matter the actual amount of cores). To enable multiple MPS devices, you therefore need multiple Macs.
Multi-Node :ref:`_gpu_intermediate` training with MPS devices should work just as multi-node training with gpus does (except setting the ``accelerator="mps"`` in the trainer).

.. note::
   This has not been tested so far!
