.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

.. _speed:

***********************
Speed up model training
***********************

There are multiple ways you can speed up your model.

Mixed precision (16-bit) training
=================================

Mixed precision is the combined use of both 32 and 16 bit floating points during model training, which reduced memory requirements and improves performance significantly, achiving over 3X speedups on modern GPUs.

Lightning offers mixed precision or 16-bit training for CPUs, GPUs, and TPUs. 

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_precision.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+9+-+precision_1.mp4"></video>

|


----------

16-bit precision on GPUs
------------------------
Mixed or 16-bit precision can cut your memory footprint by half.
If using volta architecture GPUs it can give a dramatic training speed-up as well.

When using PyTorch 1.6+, Lightning uses the native AMP implementation to support 16-bit precision.

.. testcode::
    :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

    # turn on 16-bit precision
    trainer = Trainer(precision=16, gpus=1)


PyTorch 1.6+ is recommended for 16-bit

.. admonition:: Using 16-bit precision with PyTorch < 1.6 is not recommended, but supported using apex.
   :class: dropdown, warning

    NVIDIA Apex and DDP have instability problems. We recommend upgrading to PyTorch 1.6+ to use the native AMP 16-bit precision.

    If you are using an earlier version of PyTorch (before 1.6), Lightning uses `Apex <https://github.com/NVIDIA/apex>`_ to support 16-bit training.

    To use Apex 16-bit training:

    1. Install Apex

    .. code-block:: bash

        # ------------------------
        # OPTIONAL: on your cluster you might need to load CUDA 10 or 9
        # depending on how you installed PyTorch

        # see available modules
        module avail

        # load correct CUDA before install
        module load cuda-10.0
        # ------------------------

        # make sure you've loaded a cuda version > 4.0 and < 7.0
        module load gcc-6.1.0

        $ pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex

    2. Set the `precision` trainer flag to 16. You can customize the `Apex optimization level <https://nvidia.github.io/apex/amp.html#opt-levels>`_ by setting the `amp_level` flag.

    .. testcode::
        :skipif: not _APEX_AVAILABLE and not _NATIVE_AMP_AVAILABLE or not torch.cuda.is_available()

        # turn on 16-bit
        trainer = Trainer(amp_level='O2', precision=16)

    If you need to configure the apex init for your particular use case, or want to ucustumize the
    16-bit training behviour, override :meth:`pytorch_lightning.core.LightningModule.configure_apex`.

----------

16-bit precision on TPUs
------------------------
To use 16-bit precision on TPUs simply set the number of tpu cores, and set `precision` trainer flag to 16.

.. testcode::
    :skipif: not _TPU_AVAILABLE

    # DEFAULT
    trainer = Trainer(tpu_cores=8, precision=32)

    # turn on 16-bit
    trainer = Trainer(tpu_cores=8, precision=16)
