Single GPU Training
====================
Make sure you are running on a machine that has at least one GPU. Lightning handles all the NVIDIA flags for you,
there's no need to set them yourself.

.. code-block:: python

    # train on 1 GPU (using dp mode)
    trainer = pl.Trainer(gpus=1)