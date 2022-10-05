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

.. code-block:: python

   trainer = Trainer(accelerator="mps", devices=1)

.. note::
   The ``MPSAccelerator`` only supports 1 device at a time. Currently there are no machines with multiple MPS-capable GPUs.

----

What does MPS stand for?
------------------------
MPS is short for `Metal Performance Shaders <https://developer.apple.com/metal/>`_  which is the technology used in the back for gpu communication and computing.

----

Troubleshooting
---------------


If Lightning can't detect the Apple Silicon hardware, it will raise this exception:

.. code::

   MisconfigurationException: `MPSAccelerator` can not run on your system since the accelerator is not available.

If you are seeing this despite running on an ARM-enabled Mac, the most likely cause is that your Python is being emulated and thinks it is running on an Intel CPU.
To solve this, re-install your python executable (and if using environment managers like conda, you have to reinstall these as well) by downloading the Apple M1/M2 build (not Intel!), for example `here <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`_.
