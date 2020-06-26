.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

Single GPU Training
===================
Make sure you are running on a machine that has at least one GPU. Lightning handles all the NVIDIA flags for you,
there's no need to set them yourself.

.. testcode::
    :skipif: torch.cuda.device_count() < 1

    # train on 1 GPU (using dp mode)
    trainer = Trainer(gpus=1)

Note that each step will load the batch from the CPU to the GPU.
If your entire dataset fits in memory, you can manually save your
whole dataset's data as a GPU Tensor inside the dataset object when
you create it (dataset's `__init__`). If you return it as a GPU
tensor from the dataset, you will save the time required to load
it every batch.
