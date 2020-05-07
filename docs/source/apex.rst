.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer


16-bit training
=================
Lightning offers 16-bit training for CPUs, GPUs and TPUs.

GPU 16-bit
-----------
Lightning uses NVIDIA apex to handle 16-bit precision training.

To use 16-bit precision, do two things:

1. Install Apex
2. Set the "precision" trainer flag.

Install apex
^^^^^^^^^^^^
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


Enable 16-bit
^^^^^^^^^^^^^

.. testcode::

    # turn on 16-bit
    trainer = Trainer(amp_level='O1', precision=16)

If you need to configure the apex init for your particular use case or want to use a different way of doing
16-bit training, override   :meth:`pytorch_lightning.core.LightningModule.configure_apex`.

TPU 16-bit
----------
16-bit on TPus is much simpler. To use 16-bit with TPUs set precision to 16 when using the tpu flag

.. testcode::

    # DEFAULT
    trainer = Trainer(tpu_cores=8, precision=32)

    # turn on 16-bit
    trainer = Trainer(tpu_cores=8, precision=16)
