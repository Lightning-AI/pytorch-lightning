.. _accelerator:

############
Accelerators
############

Accelerators connect a Lightning Trainer to arbitrary hardware (CPUs, GPUs, TPUs, IPUs, ...).
Currently there are accelerators for:

- CPU
- :doc:`GPU <../accelerators/gpu>`
- :doc:`TPU <../accelerators/tpu>`
- :doc:`IPU <../accelerators/ipu>`
- :doc:`HPU <../accelerators/hpu>`

The Accelerator is part of the Strategy which manages communication across multiple devices (distributed communication).
Whenever the Trainer, the loops or any other component in Lightning needs to talk to hardware, it calls into the Strategy and the Strategy calls into the Accelerator.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/strategies/overview.jpeg
    :alt: Illustration of the Strategy as a composition of the Accelerator and several plugins

We expose Accelerators and Strategies mainly for expert users who want to extend Lightning to work with new
hardware and distributed training or clusters.

Here is how you extend an existing Accelerator:

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


:doc:`Learn more about Strategies and how they interact with the Accelerator <../extensions/strategy>`.

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
    HPUAccelerator
    IPUAccelerator
    TPUAccelerator
