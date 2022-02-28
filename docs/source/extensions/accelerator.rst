.. _accelerator:

############
Accelerators
############
Accelerators connect a Lightning Trainer to arbitrary accelerators (CPUs, GPUs, TPUs, IPUs). Accelerators
also manage distributed communication through :ref:`Plugins` (like DP, DDP, HPC cluster) and
can also be configured to run on arbitrary clusters or to link up to arbitrary
computational strategies like 16-bit precision via AMP and Apex.

An Accelerator is meant to deal with one type of hardware.
Currently there are accelerators for:

- CPU
- GPU
- TPU
- IPU

Each Accelerator gets two plugins upon initialization:
One to handle differences from the training routine and one to handle different precisions.

.. testcode::
    :skipif: torch.cuda.device_count() < 2

    from pytorch_lightning import Trainer
    from pytorch_lightning.accelerators import GPUAccelerator
    from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
    from pytorch_lightning.strategies import DDPStrategy

    accelerator = GPUAccelerator()
    precision_plugin = NativeMixedPrecisionPlugin(precision=16, device="cuda")
    training_strategy = DDPStrategy(accelerator=accelerator, precision_plugin=precision_plugin)
    trainer = Trainer(strategy=training_strategy, devices=2)


We expose Accelerators and Plugins mainly for expert users who want to extend Lightning to work with new
hardware and distributed training or clusters.


.. image:: ../_static/images/accelerator/overview.svg


.. warning:: The Accelerator API is in beta and subject to change.
    For help setting up custom plugins/accelerators, please reach out to us at **support@pytorchlightning.ai**

----------


Accelerator API
---------------

.. currentmodule:: pytorch_lightning.accelerators

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    GPUAccelerator
    TPUAccelerator
    IPUAccelerator
