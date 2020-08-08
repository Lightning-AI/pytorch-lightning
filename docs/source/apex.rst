.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer


16-bit training
=================
Lightning offers 16-bit training for CPUs, GPUs and TPUs.

----------

GPU 16-bit
----------
16 bit precision can cut your memory footprint by half.
If using volta architecture GPUs it can give a dramatic training speed-up as well.

.. note:: PyTorch 1.6+ is recommended for 16-bit

Native torch
^^^^^^^^^^^^
When using PyTorch 1.6+ Lightning uses the native amp implementation to support 16-bit.

.. testcode::
    :skipif: not is_apex_available() and not is_native_amp_available()

    # turn on 16-bit
    trainer = Trainer(precision=16)

Apex 16-bit
^^^^^^^^^^^
If you are using an earlier version of PyTorch Lightning uses Apex to support 16-bit.

Follow these instructions to install Apex.
To use 16-bit precision, do two things:

1. Install Apex
2. Set the "precision" trainer flag.

.. code-block:: bash

    $ git clone https://github.com/NVIDIA/apex
    $ cd apex

    # ------------------------
    # OPTIONAL: on your cluster you might need to load cuda 10 or 9
    # depending on how you installed PyTorch

    # see available modules
    module avail

    # load correct cuda before install
    module load cuda-10.0
    # ------------------------

    # make sure you've loaded a cuda version > 4.0 and < 7.0
    module load gcc-6.1.0

    $ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

.. warning:: NVIDIA Apex and DDP have instability problems. We recommend native 16-bit in PyTorch 1.6+

Enable 16-bit
^^^^^^^^^^^^^

.. testcode::
    :skipif: not is_apex_available() and not is_native_amp_available()

    # turn on 16-bit
    trainer = Trainer(amp_level='O2', precision=16)

If you need to configure the apex init for your particular use case or want to use a different way of doing
16-bit training, override   :meth:`pytorch_lightning.core.LightningModule.configure_apex`.

----------

TPU 16-bit
----------
16-bit on TPus is much simpler. To use 16-bit with TPUs set precision to 16 when using the tpu flag

.. testcode::
    :skipif: not is_xla_available()

    # DEFAULT
    trainer = Trainer(tpu_cores=8, precision=32)

    # turn on 16-bit
    trainer = Trainer(tpu_cores=8, precision=16)
