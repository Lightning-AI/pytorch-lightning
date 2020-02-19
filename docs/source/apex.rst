16-bit training
=================
Lightning uses NVIDIA apex to handle 16-bit precision training.


To use 16-bit precision, do two things:
1. Install Apex
2. Set the amp trainer flag.

Install apex
----------------------------------------------
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
--------------

.. code-block:: python

    # DEFAULT
    trainer = Trainer(amp_level='O1', use_amp=False)

If you need to configure the apex init for your particular use case or want to use a different way of doing
16-bit training, override   :meth:`pytorch_lightning.core.LightningModule.configure_apex`.
