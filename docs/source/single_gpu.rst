.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

Single GPU Training
====================
Make sure you are running on a machine that has at least one GPU. Lightning handles all the NVIDIA flags for you,
there's no need to set them yourself.

.. testcode::
    :skipif: torch.cuda.device_count() < 1

    # train on 1 GPU (using dp mode)
    trainer = Trainer(gpus=1)